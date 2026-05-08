import numpy as np
import torch
from torch import nn
from typing import Callable, Optional, Tuple
from sklearn.base import BaseEstimator, RegressorMixin
from tqdm import tqdm
from typing import Callable, Tuple, Optional, Union, Dict, Any

import os
import sys
# Get the path of the current script, go up one level to the root
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
    
from torch_utils.scalers import TorchIdentityTransformer
from torch_utils.linear_algebra import make_psd, columns_mean_excluding_self

import copy
from torch_utils.kernel_utils import RBF

def _as_2d_tensor(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = x.to(device)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x


def _ix(K: torch.Tensor, rows, cols) -> torch.Tensor:
    rows = torch.as_tensor(rows, dtype=torch.long, device=K.device)
    cols = torch.as_tensor(cols, dtype=torch.long, device=K.device)
    return K.index_select(0, rows).index_select(1, cols)


def _take_rows(X: torch.Tensor, rows) -> torch.Tensor:
    rows = torch.as_tensor(rows, dtype=torch.long, device=X.device)
    return X.index_select(0, rows)


class KernelAlternativeProxyDoseResponseNystrom(BaseEstimator, RegressorMixin, nn.Module):
    """
    Nyström version of KernelAlternativeProxyDoseResponse.

    Approximation is used in:
      - Stage 1 conditional embedding solve.
      - Third-stage KRR solve.

    The main alpha / eta second-stage system is kept exact over the Stage-2 sample,
    matching your stated plan.
    """

    def __init__(
        self,
        kernel_A: Callable,
        kernel_W: Callable,
        kernel_Z: Callable,
        kernel_X: Optional[Callable] = None,
        lambda1_: float = 0.1,
        eta: float = 0.1,
        lambda2_: float = 0.1,
        nystrom_first_stage_m: int = 500,
        nystrom_third_stage_m: int = 500,
        device=None,
        **kwargs,
    ) -> None:
        nn.Module.__init__(self)

        self.kernel_A = kernel_A
        self.kernel_W = kernel_W
        self.kernel_Z = kernel_Z
        self.kernel_X = kernel_X if kernel_X is not None else RBF()

        self.lambda1_ = lambda1_
        self.eta = eta
        self.lambda2_ = lambda2_

        self.nystrom_first_stage_m = nystrom_first_stage_m
        self.nystrom_third_stage_m = nystrom_third_stage_m

        self.optimize_lambda_parameters = kwargs.pop("optimize_lambda_parameters", False)
        self.optimize_eta_parameter = kwargs.pop("optimize_eta_parameter", False)
        self.lambda_optimization_range = kwargs.pop("lambda_optimization_range", (1e-7, 1.0))
        self.eta_optimization_range = kwargs.pop("eta_optimization_range", (1e-7, 1.0))
        self.regularization_grid_points = kwargs.pop("regularization_grid_points", 25)
        self.make_psd_eps = kwargs.pop("make_psd_eps", 1e-7)
        self.stage1_perc = kwargs.pop("stage1_perc", 0.5)
        self.model_seed = kwargs.pop("model_seed", 0)
        self.label_variance_in_lambda_opt = kwargs.pop("label_variance_in_lambda_opt", 0.0)
        self.label_variance_in_eta_opt = kwargs.pop("label_variance_in_eta_opt", 0.0)

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.alpha = None
        self.B = None
        self.B_bar = None
        self.third_stage_KRR_weights = None
        self.ones_divided_by_m = None

        self.ATrain = None
        self.WTrain = None
        self.ZTrain = None
        self.XTrain = None

        self.K_ZZ = None
        self.train_indices = None
        self.stage1_idx = None
        self.stage2_idx = None

        self.nystrom_stage1_landmarks = None
        self.K_AWX_nm = None
        self.stage1_ridge_weights = None

    def sample_landmarks(self, original_size: int, m: int, seed: int = 0):
        if m > original_size:
            m = original_size
        rng = np.random.default_rng(seed)
        return rng.choice(original_size, m, replace=False)

    @staticmethod
    def _eta_objective(
        eta,
        L,
        L_sub,
        M,
        N,
        L2,
        M2,
        stage1_data_size,
        label_variance_in_eta_opt=0.0,
        make_psd_eps=1e-9,
    ):
        stage2_data_size = L.shape[0] - 1
        alpha = torch.linalg.solve(make_psd(L / stage2_data_size + eta * N, make_psd_eps), M)
        cost = (1.0 / stage1_data_size) * (alpha.T @ make_psd(L2, make_psd_eps) @ alpha) - 2.0 * (alpha.T @ M2)
        cost += label_variance_in_eta_opt * (2.0 / stage2_data_size) * torch.trace(
            torch.linalg.solve(make_psd(L + stage2_data_size * eta * N, make_psd_eps), L)
        )
        return cost.reshape(())

    @staticmethod
    def _predict_structural_function(
        alpha: torch.Tensor,
        B: torch.Tensor,
        B_bar: torch.Tensor,
        third_stage_KRR_weights: torch.Tensor,
        K_ATraina: torch.Tensor,
        K_ATildea: torch.Tensor,
        ones_divided_by_m: torch.Tensor,
    ) -> torch.Tensor:
        K_ATraina = K_ATraina.reshape(-1, 1)
        K_ATildea = K_ATildea.reshape(-1, 1)

        u = third_stage_KRR_weights @ K_ATraina

        term1 = (B.T @ u) * K_ATildea
        pred = alpha[:-1].T @ term1

        term2 = (B_bar.T @ u) * K_ATildea
        pred = pred + alpha[-1].reshape(1, 1) * (term2.T @ ones_divided_by_m)

        return pred.reshape(())

    def fit(
        self,
        AZWX: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        Y: torch.Tensor,
    ) -> None:
        if len(AZWX) == 4:
            A, Z, W, X = AZWX
        elif len(AZWX) == 3:
            A, Z, W = AZWX
            X = None
        else:
            raise ValueError("AZWX must be either (A, Z, W) or (A, Z, W, X).")

        A = _as_2d_tensor(A, self.device)
        Z = _as_2d_tensor(Z, self.device)
        W = _as_2d_tensor(W, self.device)
        Y = _as_2d_tensor(Y, self.device)
        X = _as_2d_tensor(X, self.device) if X is not None else None

        dtype = torch.get_default_dtype()

        K_ATrainATrain = self.kernel_A(A, A)
        K_WTrainWTrain = self.kernel_W(W, W)
        K_ZTrainZTrain = self.kernel_Z(Z, Z)

        if X is None:
            K_XTrainXTrain = torch.ones((W.shape[0], W.shape[0]), dtype=K_ATrainATrain.dtype, device=self.device)
        else:
            K_XTrainXTrain = make_psd(self.kernel_X(X, X), self.make_psd_eps)

        for kernel_ in [self.kernel_A, self.kernel_W, self.kernel_Z, self.kernel_X]:
            if hasattr(kernel_, "use_length_scale_heuristic"):
                kernel_.use_length_scale_heuristic = False

        train_data_size = A.shape[0]
        train_indices = np.random.permutation(train_data_size)

        if 0.0 < self.stage1_perc < 1.0:
            stage1_data_size = int(train_data_size * self.stage1_perc)
            stage2_data_size = train_data_size - stage1_data_size
            stage1_idx = train_indices[:stage1_data_size]
            stage2_idx = train_indices[stage1_data_size:]
        else:
            stage1_data_size = train_data_size
            stage2_data_size = train_data_size
            stage1_idx = train_indices
            stage2_idx = train_indices

        nystrom_stage1_landmarks = self.sample_landmarks(
            stage1_data_size,
            self.nystrom_first_stage_m,
            self.model_seed,
        )
        nystrom_stage3_landmarks = self.sample_landmarks(
            train_data_size,
            self.nystrom_third_stage_m,
            self.model_seed + 1,
        )

        K_AA = _ix(K_ATrainATrain, stage1_idx, stage1_idx)
        K_AATilde = _ix(K_ATrainATrain, stage1_idx, stage2_idx)
        K_ATildeATilde = _ix(K_ATrainATrain, stage2_idx, stage2_idx)

        K_WW = _ix(K_WTrainWTrain, stage1_idx, stage1_idx)
        K_WWTilde = _ix(K_WTrainWTrain, stage1_idx, stage2_idx)

        K_ZZ = _ix(K_ZTrainZTrain, stage1_idx, stage1_idx)

        K_XX = _ix(K_XTrainXTrain, stage1_idx, stage1_idx)
        K_XXTilde = _ix(K_XTrainXTrain, stage1_idx, stage2_idx)

        K_AWX = K_AA * K_WW * K_XX

        if self.optimize_lambda_parameters:
            l_min, l_max = self.lambda_optimization_range
            lambda_list = torch.exp(
                torch.linspace(
                    torch.log(torch.tensor(l_min, dtype=dtype, device=self.device)),
                    torch.log(torch.tensor(l_max, dtype=dtype, device=self.device)),
                    self.regularization_grid_points,
                    device=self.device,
                )
            )
            objective_list = torch.zeros(self.regularization_grid_points, dtype=dtype, device=self.device)
            with torch.no_grad():
                for i, lambda_val in tqdm(enumerate(lambda_list), total=len(lambda_list), desc="Tuning Lambda 1"):
                    objective_list[i] = lambda_objective_loocv(
                        lambda_val,
                        K_AWX,
                        K_ZZ,
                        self.label_variance_in_lambda_opt,
                        self.make_psd_eps,
                    )
            self.lambda1_ = lambda_list[torch.argmin(objective_list)].item()

        K_AWX_nm = K_AWX[:, nystrom_stage1_landmarks]
        K_AWX_mm = K_AWX[nystrom_stage1_landmarks][:, nystrom_stage1_landmarks]

        stage1_ridge_weights = K_AWX_nm.T @ K_AWX_nm + stage1_data_size * self.lambda1_ * K_AWX_mm
        stage1_ridge_weights = make_psd(stage1_ridge_weights, self.make_psd_eps)

        K_A_m_ATilde = K_AATilde[nystrom_stage1_landmarks, :]
        K_WX_m_WXTilde = (K_WWTilde * K_XXTilde)[nystrom_stage1_landmarks, :]

        B = K_AWX_nm @ torch.linalg.solve(stage1_ridge_weights, K_WX_m_WXTilde * K_A_m_ATilde)
        B_bar = K_AWX_nm @ torch.linalg.solve(
            stage1_ridge_weights,
            columns_mean_excluding_self(K_WX_m_WXTilde) * K_A_m_ATilde,
        )

        block_component1 = (B.T @ K_ZZ @ B) * K_ATildeATilde
        block_component2 = (B.T @ K_ZZ @ B_bar) * K_ATildeATilde
        block_component4 = (B_bar.T @ K_ZZ @ B_bar) * K_ATildeATilde

        ones_divided_by_m = torch.ones((stage2_data_size, 1), dtype=K_AWX.dtype, device=self.device) / stage2_data_size

        L_sub = torch.vstack((block_component1, ones_divided_by_m.T @ block_component2.T))
        L = L_sub @ L_sub.T

        M = torch.vstack(
            (
                (block_component2 @ ones_divided_by_m).reshape(-1, 1),
                (ones_divided_by_m.T @ block_component4 @ ones_divided_by_m).reshape(-1, 1),
            )
        )

        P = torch.hstack((block_component1, (block_component2 @ ones_divided_by_m).reshape(-1, 1)))
        R = torch.hstack(
            (
                (ones_divided_by_m.T @ block_component2.T).reshape(1, -1),
                (ones_divided_by_m.T @ block_component4 @ ones_divided_by_m).reshape(-1, 1),
            )
        )
        N = torch.vstack((P, R))

        if self.optimize_eta_parameter:
            K_ATildeA = K_AATilde.T

            B2 = K_AWX_nm @ torch.linalg.solve(stage1_ridge_weights, K_AWX[nystrom_stage1_landmarks, :])

            K_WX = K_WW * K_XX
            B2_bar_target = columns_mean_excluding_self(K_WX) * K_AA
            B2_bar = K_AWX_nm @ torch.linalg.solve(stage1_ridge_weights, B2_bar_target[nystrom_stage1_landmarks, :])

            ones_divided_by_n = torch.ones((stage1_data_size, 1), dtype=K_AWX.dtype, device=self.device) / stage1_data_size

            block_component12 = (B2.T @ K_ZZ @ B) * K_AATilde
            block_component22 = (B2.T @ K_ZZ @ B_bar) * K_AATilde
            block_component32 = (B.T @ K_ZZ @ B2_bar) * K_ATildeA
            block_component42 = (B_bar.T @ K_ZZ @ B2_bar) * K_ATildeA

            L2_sub = torch.vstack((block_component12.T, ones_divided_by_m.T @ block_component22.T))
            L2 = L2_sub @ L2_sub.T
            M2 = torch.vstack(
                (
                    (block_component32 @ ones_divided_by_n).reshape(-1, 1),
                    (ones_divided_by_m.T @ block_component42 @ ones_divided_by_n).reshape(-1, 1),
                )
            )

            e_min, e_max = self.eta_optimization_range
            eta_list = torch.exp(
                torch.linspace(
                    torch.log(torch.tensor(e_min, dtype=dtype, device=self.device)),
                    torch.log(torch.tensor(e_max, dtype=dtype, device=self.device)),
                    self.regularization_grid_points,
                    device=self.device,
                )
            )
            objective_list = torch.zeros(self.regularization_grid_points, dtype=dtype, device=self.device)

            with torch.no_grad():
                for i, eta_val in tqdm(enumerate(eta_list), total=len(eta_list), desc="Tuning Eta"):
                    objective_list[i] = self._eta_objective(
                        eta_val,
                        L,
                        L_sub,
                        M,
                        N,
                        L2,
                        M2,
                        stage1_data_size,
                        self.label_variance_in_eta_opt,
                        self.make_psd_eps,
                    )
            self.eta = eta_list[torch.argmin(objective_list)].item()

        alpha = torch.linalg.solve(make_psd(L / stage2_data_size + self.eta * N, self.make_psd_eps), M)

        if self.optimize_lambda_parameters:
            objective_list = torch.zeros(self.regularization_grid_points, dtype=dtype, device=self.device)
            K_Y = Y @ Y.T
            with torch.no_grad():
                for i, lambda_val in tqdm(enumerate(lambda_list), total=len(lambda_list), desc="Tuning Lambda 2"):
                    objective_list[i] = lambda_objective_loocv(
                        lambda_val,
                        K_ATrainATrain,
                        K_ZTrainZTrain * K_Y,
                        self.label_variance_in_lambda_opt,
                        self.make_psd_eps,
                    )
            self.lambda2_ = lambda_list[torch.argmin(objective_list)].item()

        K_ZZTrain = _ix(K_ZTrainZTrain, stage1_idx, train_indices)
        K_ATrainATrain_ = _ix(K_ATrainATrain, train_indices, train_indices)
        K_ATrainATrain_nm = K_ATrainATrain_[:, nystrom_stage3_landmarks]
        K_ATrainATrain_mm = K_ATrainATrain_[nystrom_stage3_landmarks][:, nystrom_stage3_landmarks]

        KRR_LHS = make_psd(
            K_ATrainATrain_nm.T @ K_ATrainATrain_nm
            + train_data_size * self.lambda2_ * K_ATrainATrain_mm,
            self.make_psd_eps,
        )
        KRR_RHS = K_ATrainATrain_nm.T @ (K_ZZTrain.T * Y[train_indices])
        third_stage_KRR_weights = torch.linalg.solve(KRR_LHS, KRR_RHS).T

        self.alpha = alpha
        self.B = B
        self.B_bar = B_bar
        self.third_stage_KRR_weights = third_stage_KRR_weights
        self.ones_divided_by_m = ones_divided_by_m

        self.ATrain = A
        self.WTrain = W
        self.ZTrain = Z
        self.XTrain = X

        self.K_ZZ = K_ZZ
        self.train_indices = train_indices[nystrom_stage3_landmarks]
        self.stage1_idx = stage1_idx
        self.stage2_idx = stage2_idx

        self.nystrom_stage1_landmarks = nystrom_stage1_landmarks
        self.K_AWX_nm = K_AWX_nm
        self.stage1_ridge_weights = stage1_ridge_weights

    def predict(self, A: torch.Tensor, outcome_transformer: nn.Module = TorchIdentityTransformer()) -> torch.Tensor:
        A_test = _as_2d_tensor(A, self.device)

        K_ATrainATest = self.kernel_A(self.ATrain, A_test)
        K_ATrainATest_ = _ix(K_ATrainATest, self.train_indices, np.arange(A_test.shape[0]))
        K_ATildeATest = _ix(K_ATrainATest, self.stage2_idx, np.arange(A_test.shape[0]))

        f_struct_pred = torch.vstack(
            [
                self._predict_structural_function(
                    self.alpha,
                    self.B,
                    self.B_bar,
                    self.third_stage_KRR_weights,
                    K_ATrainATest_[:, jj],
                    K_ATildeATest[:, jj],
                    self.ones_divided_by_m,
                )
                for jj in range(K_ATildeATest.shape[1])
            ]
        )

        f_struct_pred = outcome_transformer.inverse_transform(f_struct_pred.squeeze(-1))
        return f_struct_pred

    def _predict_bridge_func(self, A_test: torch.Tensor, Z_test: torch.Tensor, X_test=None):
        A_test = _as_2d_tensor(A_test, self.device)
        Z_test = _as_2d_tensor(Z_test, self.device)

        K_ZZTest = self.kernel_Z(_take_rows(self.ZTrain, self.stage1_idx), Z_test)
        K_ATildeATest = self.kernel_A(_take_rows(self.ATrain, self.stage2_idx), A_test)

        bridge_function = torch.vstack(
            [
                self.alpha[:-1].T @ ((self.B.T @ K_ZZTest) * K_ATildeATest[:, jj].reshape(-1, 1))
                + self.alpha[-1].reshape(1, 1)
                * (self.ones_divided_by_m.T @ ((self.B_bar.T @ K_ZZTest) * K_ATildeATest[:, jj].reshape(-1, 1)))
                for jj in range(K_ATildeATest.shape[1])
            ]
        )
        return bridge_function

    def _predict_density_ratio(self, A_test: torch.Tensor, W_test: torch.Tensor, X_test: Optional[torch.Tensor] = None):
        A_test = _as_2d_tensor(A_test, self.device)
        W_test = _as_2d_tensor(W_test, self.device)
        X_test = _as_2d_tensor(X_test, self.device) if X_test is not None else None

        A_stage1 = _take_rows(self.ATrain, self.stage1_idx)
        W_stage1 = _take_rows(self.WTrain, self.stage1_idx)

        K_WWTest = self.kernel_W(W_stage1, W_test)
        K_AATest = self.kernel_A(A_stage1, A_test)

        if X_test is None or self.XTrain is None:
            K_XXTest = torch.ones_like(K_AATest)
        else:
            X_stage1 = _take_rows(self.XTrain, self.stage1_idx)
            K_XXTest = self.kernel_X(X_stage1, X_test)

        target = K_WWTest * K_XXTest * K_AATest
        B_test = self.K_AWX_nm @ torch.linalg.solve(
            self.stage1_ridge_weights,
            target[self.nystrom_stage1_landmarks, :],
        )

        K_ATildeATest = self.kernel_A(_take_rows(self.ATrain, self.stage2_idx), A_test)

        dens_ratio = (
            self.alpha[:-1].T @ ((self.B.T @ self.K_ZZ @ B_test) * K_ATildeATest)
            + self.alpha[-1].reshape(1, 1)
            * (self.ones_divided_by_m.T @ ((self.B_bar.T @ self.K_ZZ @ B_test) * K_ATildeATest))
        )
        return dens_ratio.T


class KernelProxyVariableDoseResponseNystrom(BaseEstimator, RegressorMixin, nn.Module):
    """
    Nyström version of KernelProxyVariableDoseResponse.

    Approximation is used in:
      - Stage 1 conditional mean embedding solve.
      - Stage 2 outcome bridge solve.
    """

    def __init__(
        self,
        kernel_A: Callable,
        kernel_W: Callable,
        kernel_Z: Callable,
        kernel_X: Optional[Callable] = None,
        lambda1_: float = 0.1,
        lambda2_: float = 0.1,
        nystrom_first_stage_m: int = 500,
        nystrom_second_stage_m: int = 500,
        optimize_lambda1_parameter: bool = False,
        optimize_lambda2_parameter: bool = False,
        lambda1_optimization_range: Tuple[float, float] = (1e-5, 1.0),
        lambda2_optimization_range: Tuple[float, float] = (1e-5, 1.0),
        device=None,
        **kwargs,
    ) -> None:
        nn.Module.__init__(self)

        self.kernel_A = kernel_A
        self.kernel_W = kernel_W
        self.kernel_Z = kernel_Z
        self.kernel_X = kernel_X if kernel_X is not None else RBF()

        self.lambda1_ = lambda1_
        self.lambda2_ = lambda2_

        self.nystrom_first_stage_m = nystrom_first_stage_m
        self.nystrom_second_stage_m = nystrom_second_stage_m

        self.optimize_lambda1_parameter = optimize_lambda1_parameter
        self.optimize_lambda2_parameter = optimize_lambda2_parameter
        self.lambda1_optimization_range = lambda1_optimization_range
        self.lambda2_optimization_range = lambda2_optimization_range

        self.stage1_perc = kwargs.pop("stage1_perc", 0.5)
        self.regularization_grid_points = kwargs.pop("regularization_grid_points", 25)
        self.make_psd_eps = kwargs.pop("make_psd_eps", 1e-9)
        self.label_variance_in_lambda_opt = kwargs.pop("label_variance_in_lambda_opt", 0.0)
        self.model_seed = kwargs.pop("model_seed", 0)

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.alpha = None
        self.B = None
        self.w_mean_vec = None
        self.x_mean_vec = None
        self.upweight = None

        self.stage1_idx = None
        self.stage2_idx = None
        self.nystrom_stage2_landmarks = None

        self.ATilde = None
        self.XTilde = None
        self.W = None

    def sample_landmarks(self, original_size: int, m: int, seed: int = 0):
        if m > original_size:
            m = original_size
        rng = np.random.default_rng(seed)
        return rng.choice(original_size, m, replace=False)

    def fit(
        self,
        AZWX: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        Y: torch.Tensor,
    ) -> None:
        if len(AZWX) == 4:
            A, Z, W, X = AZWX
        elif len(AZWX) == 3:
            A, Z, W = AZWX
            X = None
        else:
            raise ValueError("AZWX must be either (A, Z, W) or (A, Z, W, X).")

        A = _as_2d_tensor(A, self.device)
        Z = _as_2d_tensor(Z, self.device)
        W = _as_2d_tensor(W, self.device)
        Y = _as_2d_tensor(Y, self.device)
        X = _as_2d_tensor(X, self.device) if X is not None else None

        dtype = torch.get_default_dtype()

        K_ATrainATrain = self.kernel_A(A, A)
        K_WTrainWTrain = self.kernel_W(W, W)
        K_ZTrainZTrain = self.kernel_Z(Z, Z)

        if X is None:
            K_XTrainXTrain = torch.ones((W.shape[0], W.shape[0]), dtype=K_ATrainATrain.dtype, device=self.device)
        else:
            K_XTrainXTrain = make_psd(self.kernel_X(X, X), self.make_psd_eps)

        for kernel_ in [self.kernel_A, self.kernel_W, self.kernel_Z, self.kernel_X]:
            if hasattr(kernel_, "use_length_scale_heuristic"):
                kernel_.use_length_scale_heuristic = False

        n_total = A.shape[0]
        indices = np.random.permutation(n_total)

        if 0.0 < self.stage1_perc < 1.0:
            n1 = int(n_total * self.stage1_perc)
            stage1_idx = indices[:n1]
            stage2_idx = indices[n1:]
        else:
            n1 = n_total
            stage1_idx = indices
            stage2_idx = indices

        n2 = len(stage2_idx)

        nystrom_stage1_landmarks = self.sample_landmarks(n1, self.nystrom_first_stage_m, self.model_seed)
        nystrom_stage2_landmarks = self.sample_landmarks(n2, self.nystrom_second_stage_m, self.model_seed + 1)

        K_AATilde = _ix(K_ATrainATrain, stage1_idx, stage2_idx)
        K_ATildeATilde = _ix(K_ATrainATrain, stage2_idx, stage2_idx)

        K_WW = _ix(K_WTrainWTrain, stage1_idx, stage1_idx)

        K_ZZTilde = _ix(K_ZTrainZTrain, stage1_idx, stage2_idx)

        K_XXTilde = _ix(K_XTrainXTrain, stage1_idx, stage2_idx)
        K_XTildeXTilde = _ix(K_XTrainXTrain, stage2_idx, stage2_idx)

        K_ZAX = _ix(K_ZTrainZTrain * K_ATrainATrain * K_XTrainXTrain, stage1_idx, stage1_idx)

        if self.optimize_lambda1_parameter:
            l_min, l_max = self.lambda1_optimization_range
            lambda_list = torch.exp(
                torch.linspace(
                    torch.log(torch.tensor(l_min, dtype=dtype, device=self.device)),
                    torch.log(torch.tensor(l_max, dtype=dtype, device=self.device)),
                    self.regularization_grid_points,
                    device=self.device,
                )
            )
            obj_list = torch.zeros(self.regularization_grid_points, dtype=dtype, device=self.device)
            with torch.no_grad():
                for i, l_val in tqdm(enumerate(lambda_list), total=len(lambda_list), desc="Tuning Lambda 1"):
                    obj_list[i] = lambda_objective_loocv(
                        l_val,
                        K_ZAX,
                        K_WW,
                        self.label_variance_in_lambda_opt,
                        self.make_psd_eps,
                    )
            self.lambda1_ = lambda_list[torch.argmin(obj_list)].item()

        K_ZAX_nm = K_ZAX[:, nystrom_stage1_landmarks]
        K_ZAX_mm = K_ZAX[nystrom_stage1_landmarks][:, nystrom_stage1_landmarks]

        stage1_ridge_weights = make_psd(
            K_ZAX_nm.T @ K_ZAX_nm + n1 * self.lambda1_ * K_ZAX_mm,
            self.make_psd_eps,
        )

        K_ZAX_Tilde = K_ZZTilde * K_AATilde * K_XXTilde
        K_ZAX_m_ZAXTilde = K_ZAX_Tilde[nystrom_stage1_landmarks, :]

        B = K_ZAX_nm @ torch.linalg.solve(stage1_ridge_weights, K_ZAX_m_ZAXTilde)
        self.B = B

        stage2_ridge_weights = K_ATildeATilde * (B.T @ K_WW @ B) * K_XTildeXTilde
        YTilde = Y[stage2_idx]

        if self.optimize_lambda2_parameter:
            l_min, l_max = self.lambda2_optimization_range
            lambda_list = torch.exp(
                torch.linspace(
                    torch.log(torch.tensor(l_min, dtype=dtype, device=self.device)),
                    torch.log(torch.tensor(l_max, dtype=dtype, device=self.device)),
                    self.regularization_grid_points,
                    device=self.device,
                )
            )
            K_YTilde = YTilde @ YTilde.T
            obj_list = torch.zeros(self.regularization_grid_points, dtype=dtype, device=self.device)
            with torch.no_grad():
                for i, l_val in tqdm(enumerate(lambda_list), total=len(lambda_list), desc="Tuning Lambda 2"):
                    obj_list[i] = lambda_objective_loocv(
                        l_val,
                        stage2_ridge_weights,
                        K_YTilde,
                        self.label_variance_in_lambda_opt,
                        self.make_psd_eps,
                    )
            self.lambda2_ = lambda_list[torch.argmin(obj_list)].item()

        stage2_ridge_weights_mn = stage2_ridge_weights[nystrom_stage2_landmarks, :]
        stage2_ridge_weights_mm = stage2_ridge_weights[nystrom_stage2_landmarks][:, nystrom_stage2_landmarks]

        alpha = torch.linalg.solve(
            make_psd(
                stage2_ridge_weights_mn @ stage2_ridge_weights_mn.T
                + n2 * self.lambda2_ * stage2_ridge_weights_mm,
                self.make_psd_eps,
            ),
            stage2_ridge_weights_mn @ YTilde,
        )

        self.alpha = alpha
        self.w_mean_vec = torch.mean(K_WW, dim=0, keepdim=True).T
        self.x_mean_vec = torch.mean(K_XXTilde, dim=0, keepdim=True).T

        self.ATilde = _take_rows(A[stage2_idx], nystrom_stage2_landmarks)
        self.XTilde = _take_rows(X[stage2_idx], nystrom_stage2_landmarks) if X is not None else None
        self.W = A.new_tensor(W[stage1_idx].detach().cpu().numpy()).to(self.device)

        self.upweight = torch.mean(
            (B[:, nystrom_stage2_landmarks].T @ K_WW) * K_XXTilde[:, nystrom_stage2_landmarks].T,
            dim=1,
            keepdim=True,
        )

        self.stage1_idx = stage1_idx
        self.stage2_idx = stage2_idx
        self.nystrom_stage2_landmarks = nystrom_stage2_landmarks

        self.kernel_A = self.kernel_A
        self.kernel_W = self.kernel_W
        self.kernel_Z = self.kernel_Z
        self.kernel_X = self.kernel_X

    def predict(self, A: torch.Tensor, outcome_transformer: nn.Module = TorchIdentityTransformer()) -> torch.Tensor:
        A_test = _as_2d_tensor(A, self.device)

        K_ATildeATest = self.kernel_A(self.ATilde, A_test)
        pred = (K_ATildeATest * self.upweight).T @ self.alpha

        pred = outcome_transformer.inverse_transform(pred.squeeze(-1))
        return pred

    def _predict_bridge_func(
        self,
        A_test: torch.Tensor,
        W_test: torch.Tensor,
        X_test: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        A_test = _as_2d_tensor(A_test, self.device)
        W_test = _as_2d_tensor(W_test, self.device)
        X_test = _as_2d_tensor(X_test, self.device) if X_test is not None else None

        K_ATildeATest = self.kernel_A(self.ATilde, A_test)
        K_WWTest = self.kernel_W(self.W, W_test)

        if X_test is not None and self.XTilde is not None:
            K_XTildeXTest = self.kernel_X(self.XTilde, X_test)
        else:
            K_XTildeXTest = torch.ones((self.ATilde.shape[0], W_test.shape[0]), dtype=K_ATildeATest.dtype, device=self.device)

        B_sub = self.B[:, self.nystrom_stage2_landmarks]

        bridge_preds = []
        for jj in range(A_test.shape[0]):
            feat = K_ATildeATest[:, jj].reshape(-1, 1) * ((B_sub.T @ K_WWTest) * K_XTildeXTest)
            bridge_preds.append(feat.T @ self.alpha)

        return torch.stack(bridge_preds).squeeze(-1)


class DoublyRobustKernelProxyDoseResponseNystrom(BaseEstimator, RegressorMixin, nn.Module):
    """
    Nyström doubly robust kernel proxy estimator for dose-response estimation.

    This estimator combines:
      - KernelAlternativeProxyDoseResponseNystrom for the treatment bridge;
      - KernelProxyVariableDoseResponseNystrom for the outcome bridge;
      - a Nyström KRR smoother for the interaction/slack term.

    The fit tuple follows the torch convention:
        (A, Z, W) or (A, Z, W, X).

    Regularization parameters are fixed by the user. No tuning is performed.
    """

    def __init__(
        self,
        treatment_bridge_params: Optional[Dict[str, Any]] = None,
        outcome_bridge_params: Optional[Dict[str, Any]] = None,
        lambda_DR: float = 1e-3,
        nystrom_m: int = 500,
        device=None,
        **kwargs,
    ) -> None:
        nn.Module.__init__(self)

        self.lambda_DR = lambda_DR
        self.nystrom_m = nystrom_m
        self.model_seed = kwargs.pop("model_seed", 0)
        self.make_psd_eps = kwargs.pop("make_psd_eps", 5e-9)

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if treatment_bridge_params is None:
            treatment_bridge_params = {
                "kernel_A": RBF(use_length_scale_heuristic=True),
                "kernel_W": RBF(use_length_scale_heuristic=True),
                "kernel_Z": RBF(use_length_scale_heuristic=True),
                "lambda1_": 1e-3,
                "eta": 1e-3,
                "lambda2_": 1e-3,
                "nystrom_first_stage_m": 500,
                "nystrom_third_stage_m": 500,
                "stage1_perc": 0.5,
                "model_seed": self.model_seed,
                "make_psd_eps": self.make_psd_eps,
                "device": self.device,
            }

        if outcome_bridge_params is None:
            outcome_bridge_params = {
                "algorithm_name": "Kernel_Proxy_Variable",
                "kernel_A": RBF(use_length_scale_heuristic=True),
                "kernel_W": RBF(use_length_scale_heuristic=True),
                "kernel_Z": RBF(use_length_scale_heuristic=True),
                "lambda1_": 0.1,
                "lambda2_": 0.1,
                "nystrom_first_stage_m": 500,
                "nystrom_second_stage_m": 500,
                "stage1_perc": 0.5,
                "model_seed": self.model_seed + 1,
                "make_psd_eps": self.make_psd_eps,
                "device": self.device,
            }

        self.treatment_bridge_params = self._clean_treatment_params(treatment_bridge_params)
        self.outcome_bridge_params = self._clean_outcome_params(outcome_bridge_params)

        self.treatment_bridge_algo = KernelAlternativeProxyDoseResponseNystrom(
            **self.treatment_bridge_params
        )

        algorithm_name = outcome_bridge_params.get("algorithm_name", "Kernel_Proxy_Variable")
        if algorithm_name == "Kernel_Proxy_Variable":
            self.outcome_bridge_algo = KernelProxyVariableDoseResponseNystrom(
                **self.outcome_bridge_params
            )
        else:
            raise NotImplementedError(
                f"Only Kernel_Proxy_Variable is implemented for the outcome bridge, got {algorithm_name}."
            )

        self.A = None
        self.Z = None
        self.W = None
        self.X = None
        self.Y = None
        self.kernel_A = None

        self.nystrom_landmarks = None
        self.DR_KRR_weights = None
        self.slack_prediction_raw = None
        self.treatment_bridge_algo_pred_raw = None
        self.outcome_bridge_algo_pred_raw = None

    @staticmethod
    def _clean_treatment_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove tuning-related keys and normalize parameter names.

        The older JAX code used lambda_, while the torch KAP class uses lambda1_.
        """
        params = copy.deepcopy(params)

        # Map old JAX-style name to current torch-style name.
        if "lambda_" in params and "lambda1_" not in params:
            params["lambda1_"] = params.pop("lambda_")
        else:
            params.pop("lambda_", None)

        # Remove all tuning-related keys. Regularization is fixed.
        for key in [
            "optimize_lambda_parameters",
            "optimize_eta_parameter",
            "lambda_optimization_range",
            "eta_optimization_range",
            "regularization_grid_points",
            "label_variance_in_lambda_opt",
            "label_variance_in_eta_opt",
            "large_eta_option",
            "selecting_biggest_eta_tol",
            "zeta",
            "kernel_V",
        ]:
            params.pop(key, None)

        return params

    @staticmethod
    def _clean_outcome_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove tuning-related keys and metadata. Regularization is fixed.
        """
        params = copy.deepcopy(params)

        for key in [
            "algorithm_name",
            "optimize_lambda1_parameter",
            "optimize_lambda2_parameter",
            "lambda1_optimization_range",
            "lambda2_optimization_range",
            "regularization_grid_points",
            "label_variance_in_lambda_opt",
            "zeta",
            "kernel_V",
        ]:
            params.pop(key, None)

        return params

    def sample_landmarks(self, original_size: int, m: int, seed: int = 0):
        if m > original_size:
            m = original_size
        rng = np.random.default_rng(seed)
        return rng.choice(original_size, m, replace=False)

    @staticmethod
    def _ensure_bridge_shape(
        bridge_values: torch.Tensor,
        n_test: int,
        n_train: int,
        name: str,
    ) -> torch.Tensor:
        """
        Ensure bridge predictions have shape (n_test, n_train).
        """
        bridge_values = bridge_values.squeeze()

        if bridge_values.ndim == 1:
            if n_test == 1 and bridge_values.shape[0] == n_train:
                bridge_values = bridge_values.reshape(1, n_train)
            else:
                raise ValueError(
                    f"{name} bridge prediction has invalid 1D shape {bridge_values.shape}; "
                    f"expected ({n_test}, {n_train})."
                )

        if bridge_values.shape == (n_train, n_test):
            bridge_values = bridge_values.T

        if bridge_values.shape != (n_test, n_train):
            raise ValueError(
                f"{name} bridge prediction has shape {bridge_values.shape}; "
                f"expected ({n_test}, {n_train})."
            )

        return bridge_values

    def fit(
        self,
        AZWX: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        Y: torch.Tensor,
    ):
        if len(AZWX) == 4:
            A, Z, W, X = AZWX
        elif len(AZWX) == 3:
            A, Z, W = AZWX
            X = None
        else:
            raise ValueError("AZWX must be either (A, Z, W) or (A, Z, W, X).")

        A = _as_2d_tensor(A, self.device)
        Z = _as_2d_tensor(Z, self.device)
        W = _as_2d_tensor(W, self.device)
        Y = _as_2d_tensor(Y, self.device)
        X = _as_2d_tensor(X, self.device) if X is not None else None

        self.treatment_bridge_algo.fit((A, Z, W) if X is None else (A, Z, W, X), Y)
        self.outcome_bridge_algo.fit((A, Z, W) if X is None else (A, Z, W, X), Y)

        self.A = A
        self.Z = Z
        self.W = W
        self.X = X
        self.Y = Y
        self.kernel_A = self.treatment_bridge_algo.kernel_A

        return self

    def _compute_dr_krr_weights(self, A_test: torch.Tensor) -> torch.Tensor:
        """
        Compute Nyström KRR smoothing weights for E[phi(A_i) | A=a].

        Returns
        -------
        DR_KRR_weights:
            Tensor of shape (n_train, n_test).
        """
        n_train = self.A.shape[0]

        nystrom_landmarks = self.sample_landmarks(
            original_size=n_train,
            m=self.nystrom_m,
            seed=self.model_seed,
        )

        A_landmarks = _take_rows(self.A, nystrom_landmarks)

        K_AA_nm = self.kernel_A(self.A, A_landmarks)
        K_AA_mm = self.kernel_A(A_landmarks, A_landmarks)
        K_Aland_A_test = self.kernel_A(A_landmarks, A_test)

        lhs = make_psd(
            K_AA_nm.T @ K_AA_nm + n_train * self.lambda_DR * K_AA_mm,
            self.make_psd_eps,
        )

        DR_KRR_weights = K_AA_nm @ torch.linalg.solve(lhs, K_Aland_A_test)

        self.nystrom_landmarks = nystrom_landmarks
        self.DR_KRR_weights = DR_KRR_weights

        return DR_KRR_weights

    def predict(
        self,
        A: torch.Tensor,
        outcome_transformer: nn.Module = TorchIdentityTransformer(),
    ) -> torch.Tensor:
        """
        Predict the doubly robust dose-response curve.

        Important:
        The component predictions and the slack term are first computed in the
        scale on which the bridge algorithms were fitted. The final DR curve is
        then inverse-transformed once.
        """
        A_test = _as_2d_tensor(A, self.device)
        n_test = A_test.shape[0]
        n_train = self.A.shape[0]

        # Component predictions in the fitted outcome scale.
        treatment_bridge_algo_pred_raw = self.treatment_bridge_algo.predict(
            A_test,
            TorchIdentityTransformer(),
        ).reshape(-1)

        outcome_bridge_algo_pred_raw = self.outcome_bridge_algo.predict(
            A_test,
            TorchIdentityTransformer(),
        ).reshape(-1)

        self.treatment_bridge_algo_pred_raw = treatment_bridge_algo_pred_raw
        self.outcome_bridge_algo_pred_raw = outcome_bridge_algo_pred_raw

        # Bridge values on the training data.
        treatment_bridge_values = self.treatment_bridge_algo._predict_bridge_func(
            A_test,
            self.Z,
            self.X,
        )
        outcome_bridge_values = self.outcome_bridge_algo._predict_bridge_func(
            A_test,
            self.W,
            self.X,
        )

        treatment_bridge_values = self._ensure_bridge_shape(
            treatment_bridge_values,
            n_test=n_test,
            n_train=n_train,
            name="Treatment",
        )
        outcome_bridge_values = self._ensure_bridge_shape(
            outcome_bridge_values,
            n_test=n_test,
            n_train=n_train,
            name="Outcome",
        )

        # Nyström KRR weights for the DR interaction/slack term.
        DR_KRR_weights = self._compute_dr_krr_weights(A_test)  # (n_train, n_test)

        slack_prediction_raw = (
            treatment_bridge_values
            * outcome_bridge_values
            * DR_KRR_weights.T
        ).sum(dim=1)

        self.slack_prediction_raw = slack_prediction_raw

        f_struct_pred_raw = (
            treatment_bridge_algo_pred_raw
            + outcome_bridge_algo_pred_raw
            - slack_prediction_raw
        )

        f_struct_pred = outcome_transformer.inverse_transform(f_struct_pred_raw)

        return f_struct_pred

    def predict_components(
        self,
        A: torch.Tensor,
        outcome_transformer: nn.Module = TorchIdentityTransformer(),
    ) -> Dict[str, torch.Tensor]:
        """
        Return DR prediction and its three components.

        Components are returned after applying the same final inverse-transform
        convention only to the final DR prediction. The raw components are also
        included for diagnostic use.
        """
        A_test = _as_2d_tensor(A, self.device)

        f_dr = self.predict(A_test, outcome_transformer=outcome_transformer)

        return {
            "dr_prediction": f_dr,
            "treatment_prediction_raw": self.treatment_bridge_algo_pred_raw,
            "outcome_prediction_raw": self.outcome_bridge_algo_pred_raw,
            "slack_prediction_raw": self.slack_prediction_raw,
            "dr_krr_weights": self.DR_KRR_weights,
        }
