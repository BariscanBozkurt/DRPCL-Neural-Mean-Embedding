# causal_models/kernel_dose_response.py (A-notation, Streamlined)

import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from tqdm import tqdm

from typing import Callable, Tuple, Optional, Union, Dict
import numpy as np

from torch_utils.ml_utils import lambda_objective_loocv, cme_lambda_objective_loocv
from torch_utils.linear_algebra import make_psd


class KernelDoseResponse(BaseEstimator, RegressorMixin, nn.Module):
    """
    Estimator for the Average Treatment Effect (Dose Response Curve, theta_0^ATE(a))
    based on Kernel Ridge Regression in a Tensor Product RKHS.

    See Algorithm 4.1-(2) in https://arxiv.org/pdf/2010.04855
    Kernel Methods for Causal Functions: Dose, Heterogeneous, and Incremental Response Curves (Rahul Singh et al.)
    """
    def __init__(self,
                 kernel_A: Callable, # Kernel for Treatment/Action (A)
                 kernel_X: Callable, # Kernel for Covariates (X)
                 lambda_: float = 1e-3,
                 optimize_regularization_parameters: bool = True,
                 lambda_optimization_range: Tuple[float, float] = (1e-5, 1.0),
                 **kwargs) -> None:
        
        nn.Module.__init__(self)
        
        # ... (Parameter assignments remain the same: self.lambda_, self.kernel_X, etc.)
        self.kernel_X = kernel_X
        self.kernel_A = kernel_A
        self.lambda_ = lambda_
        self.optimize_regularization_parameters = optimize_regularization_parameters
        self.lambda_optimization_range = lambda_optimization_range
        self.regularization_grid_points = kwargs.pop('regularization_grid_points', 150)
        self.make_psd_eps = kwargs.pop('make_psd_eps', 1e-9)

        # ... (Kernel setting logic remains the same, assuming it checks/sets params) ...
        # NOTE: Using a simplified check for callable kernels now:
        if not isinstance(kernel_X, Callable) or not isinstance(kernel_A, Callable):
            raise TypeError("Kernels for A and X must be callable (nn.Module).")
        
        # --- Stored Fitted Variables ---
        self.W_matrix_ = None       # The inverse term W = Y^T @ (K_WW + n*lambda*I)^-1
        self.K_XX_mean_ = None      # Proxy for mean embedding mu_x = E[phi(X)]
        self.A_train = None         
        self.Y_train = None         
        

    def fit(self, 
            AX: Tuple[torch.Tensor, torch.Tensor], 
            Y: torch.Tensor) -> 'KernelDoseResponse':
        """
        Fit the KernelDoseResponse model by calculating the optimal KRR weight matrix.
        """
        A, X = AX
        n = A.shape[0]

        # 1. Data and Type Setup
        device = X.device
        dtype = torch.get_default_dtype()
        A, X, Y = A.to(dtype), X.to(dtype), Y.to(dtype)
        Y_reshaped = Y.reshape(-1, 1) # NEW: Reshape Y to (1, n) for prediction matrix math

        # 2. Kernel Matrix Computation
        K_XX = self.kernel_X(X)
        K_AA = self.kernel_A(A)
        K_WW = K_XX * K_AA # Tensor Product Kernel Matrix

        # 3. Optimization of Regularization Parameter (lambda)
        if self.optimize_regularization_parameters:
            l_range = self.lambda_optimization_range
            grid_points = self.regularization_grid_points
            
            log_min = torch.log(torch.tensor(l_range[0], dtype=dtype, device=device))
            log_max = torch.log(torch.tensor(l_range[1], dtype=dtype, device=device))
            lambda_list = torch.exp(torch.linspace(log_min, log_max, grid_points, device=device))
            
            objective_list = torch.zeros(grid_points, dtype=dtype, device=device)
            
            with torch.no_grad():
                for i, lambda_val in tqdm(enumerate(lambda_list), total=grid_points, desc="Tuning Lambda"):
                    objective_list[i] = lambda_objective_loocv(lambda_val, K_WW, Y, make_psd_eps = self.make_psd_eps)
            
            best_lambda = lambda_list[torch.argmin(objective_list).item()]
            self.lambda_ = best_lambda.item()

        # 4. Final Solution (Calculate and store the weight matrix)
        
        # A_matrix = K_WW + n * lambda * I
        lambda_term = n * self.lambda_
        A_matrix = K_WW + lambda_term * torch.eye(n, dtype=dtype, device=device)
        A_psd = make_psd(A_matrix, eps=self.make_psd_eps)

        # W_matrix_ = Y^T @ inv(A) (The full set of KRR coefficients projected onto Y)
        # self.W_matrix_ = (Y_reshaped @ torch.linalg.pinv(A_psd)).detach() # Shape (1, n)
        self.W_matrix_ = torch.linalg.solve(A_psd, Y_reshaped).T.detach()
        # K_XX_mean_ = E[phi(X)] projection (mu_x proxy)
        self.K_XX_mean_ = torch.mean(K_XX, dim=0).detach() # Shape (n,)

        # 5. Store Fitted Data
        self.A_train = A.detach()
        
        return self

    def predict(self, 
                A_test: torch.Tensor) -> torch.Tensor:
        """
        Predict outcomes (Dose Response) for new treatment values a.

        The prediction is: W_matrix @ K_Wa_test
        """
        if self.W_matrix_ is None:
            raise RuntimeError("Model must be fitted before calling predict.")
            
        A_train = self.A_train
        
        # 1. Ensure input is on the correct device/dtype and is (n_test, 1)
        A_test = A_test.to(A_train.device).to(A_train.dtype).reshape(-1, 1)

        # 2. Compute cross-kernel K_Aa = K_A(A_train, A_test) -> Shape (n_train, n_test)
        K_Aa = self.kernel_A(A_train, A_test) 
        
        # 3. Construct the integral term K_Wa_test
        # K_Wa_test (n_train, n_test) = K_Aa (n_train, n_test) * K_XX_mean_vector (n_train, 1)
        K_XX_mean_vector = self.K_XX_mean_.unsqueeze(-1) 
        K_Wa_test = K_Aa * K_XX_mean_vector 
        
        # 4. Final Prediction: W_matrix @ K_Wa_test 
        # Result: (1, n_train) @ (n_train, n_test) = (1, n_test)
        Y_a_pred = self.W_matrix_ @ K_Wa_test
        
        # Return predicted values of shape (n_test,)
        return torch.t(Y_a_pred)

    def fit_predict(self, 
                    AX: Tuple[torch.Tensor, torch.Tensor], 
                    Y: torch.Tensor) -> torch.Tensor:
        """
        Fit the model and predict outcomes for the training treatment values.
        """
        self.fit(AX, Y)
        A, _ = AX
        return self.predict(A)
    

class KernelConditionalDoseResponse(BaseEstimator, RegressorMixin, nn.Module):
    """
    Estimator for the Conditional Dose Response Curve (theta_0^ATT(a, a')).
    
    This is a two-stage estimator:
    1. KRR for the full regression (Y on (A, X)).
    2. KRR for the Conditional Mean Embedding (CME) (phi(X) on phi(A)).

    See Algorithm 4.1-(2) in https://arxiv.org/pdf/2010.04855
    Kernel Methods for Causal Functions: Dose, Heterogeneous, and Incremental Response Curves (Rahul Singh et al.)
    """
    def __init__(self,
                 kernel_A: Callable, # Kernel for Treatment/Action (A)
                 kernel_X: Callable, # Kernel for Covariates (X)
                 lambda1_: float = 1e-3, # Regularization for the main KRR (lambda)
                 lambda2_: float = 1e-3, # Regularization for the CME (lambda1/lambda2)
                 optimize_regularization_parameters: bool = True,
                 lambda_optimization_range: Tuple[float, float] = (1e-5, 1.0),
                 **kwargs) -> None:
        
        nn.Module.__init__(self)
        
        self.kernel_X = kernel_X
        self.kernel_A = kernel_A
        self.lambda1_ = lambda1_
        self.lambda2_ = lambda2_
        self.optimize_regularization_parameters = optimize_regularization_parameters
        self.lambda_optimization_range = lambda_optimization_range
        
        # ... (Parameter extraction and error checking remain the same)
        self.regularization_grid_points = kwargs.pop('regularization_grid_points', 150)
        self.make_psd_eps = kwargs.pop('make_psd_eps', 1e-9)

        if not isinstance(kernel_X, Callable) or not isinstance(kernel_A, Callable):
            raise TypeError("Kernels for A and X must be callable (nn.Module).")
        
        # Set kernel params (omitted for brevity, assume similar logic to KernelDoseResponse)
        
        # --- Stored Fitted Variables ---
        self.ridge_term_inv_ = None # (K_WW + n*lambda*I)^-1
        self.cme_term_inv_ = None   # K_XX @ (K_AA + n*lambda2*I)^-1
        self.A_train = None         
        self.Y_train = None         


    def fit(self, 
            AX: Tuple[torch.Tensor, torch.Tensor], 
            Y: torch.Tensor) -> 'KernelConditionalDoseResponse':
        """
        Fit the model by calculating the two KRR weight matrices (for gamma_0 and mu_x(a)).
        """
        A, X = AX
        n = A.shape[0]

        # 1. Data and Type Setup
        device = X.device
        dtype = torch.get_default_dtype()
        A, X, Y = A.to(dtype), X.to(dtype), Y.to(dtype)
        Y_reshaped = Y.reshape(1, -1) # Y^T is shape (1, n)

        # 2. Kernel Matrix Computation
        K_XX = self.kernel_X(X)
        K_AA = self.kernel_A(A)
        K_WW = K_XX * K_AA # Tensor Product Kernel Matrix K((A,X), (A',X'))
        
        # 3. Optimization of Regularization Parameters (lambda_ and lambda2_)
        if self.optimize_regularization_parameters:
            
            l_range = self.lambda_optimization_range
            grid_points = self.regularization_grid_points
            
            log_min = torch.log(torch.tensor(l_range[0], dtype=dtype, device=device))
            log_max = torch.log(torch.tensor(l_range[1], dtype=dtype, device=device))
            lambda_list = torch.exp(torch.linspace(log_min, log_max, grid_points, device=device))
            
            # --- Tune lambda_ (Main KRR) ---
            ridge_obj_list = torch.zeros(grid_points, dtype=dtype, device=device)
            cme_obj_list = torch.zeros(grid_points, dtype=dtype, device=device)

            with torch.no_grad():
                for i, lambda_val in tqdm(enumerate(lambda_list), total=grid_points, desc="Tuning Lambdas"):
                    ridge_obj_list[i] = lambda_objective_loocv(lambda_val, K_WW, Y, self.make_psd_eps)
                    
                    # --- Tune lambda2_ (CME KRR: phi(X) on phi(A)) ---
                    # K_AA is the input kernel, K_XX is the output kernel (K_YY in the CME formula)
                    cme_obj_list[i] = cme_lambda_objective_loocv(lambda_val, K_AA, K_XX, self.make_psd_eps)
            
            self.lambda1_ = lambda_list[torch.argmin(ridge_obj_list).item()].item()
            self.lambda2_ = lambda_list[torch.argmin(cme_obj_list).item()].item()


        # 4. Final Solution (Closed Form KRR Terms)
        
        # Term 1: Ridge Regression for gamma_0 (W_matrix)
        # W_matrix = Y^T @ inv(K_WW + n*lambda*I)
        lambda1_term = n * self.lambda1_
        A_matrix = K_WW + lambda1_term * torch.eye(n, dtype=dtype, device=device)
        A_psd = make_psd(A_matrix, eps=self.make_psd_eps)
        # Use solve/transpose equivalent of Y^T @ pinv(A)
        self.W_matrix_ = torch.linalg.solve(A_psd.T, Y_reshaped.T).T.detach() # Shape (1, n)
        
        # Term 2: Conditional Mean Embedding (CME) weights (cme_term_inv_)
        # CME_weights = K_XX @ inv(K_AA + n*lambda2*I)
        lambda2_term = n * self.lambda2_
        B_matrix = K_AA + lambda2_term * torch.eye(n, dtype=dtype, device=device)
        B_psd = make_psd(B_matrix, eps=self.make_psd_eps)
        
        # Solving for X in B @ X = K_XX^T (then transposing) is one way, but simpler to use pinv
        self.cme_term_inv_ = (K_XX @ torch.linalg.pinv(B_psd)).detach() # Shape (n, n)
        
        # 5. Store Fitted Data
        self.A_train = A.detach()
        
        return self

    def predict(self, 
                A_counterfactual: torch.Tensor,
                A_subpopulation: torch.Tensor, # The subpopulation treatment (D=d)
                ) -> torch.Tensor: # The counterfactual treatment (D=d')
        """
        Predict the Conditional Dose Response (theta_0^ATT(a, a')).
        
        Prediction = W_matrix @ [K_A a @ (K_XX @ inv(K_AA + n*lambda2*I)) @ K_A a']
        """
        if self.W_matrix_ is None:
            raise RuntimeError("Model must be fitted before calling predict.")
            
        A_train = self.A_train
        
        # 1. Ensure input is on the correct device/dtype and is (n_test, 1)
        A_subpopulation = A_subpopulation.to(A_train.device).to(A_train.dtype).reshape(-1, 1) # a
        A_counterfactual = A_counterfactual.to(A_train.device).to(A_train.dtype).reshape(-1, 1) # a'

        # 2. Compute cross-kernel K_A a (for the subpopulation) -> (n_train, n_test)
        K_Aaprime_subpopulation = self.kernel_A(A_train, A_subpopulation) 

        # 3. Compute cross-kernel K_A a' (for the counterfactual) -> (n_train, n_test)
        K_Aa_counterfactual = self.kernel_A(A_train, A_counterfactual) 
        
        # 4. Final Prediction Term Assembly (Matrix form of the integral)
        
        # Step 4a: Estimate mu_x(a) = K_XX @ inv(K_AA + n*lambda2*I) @ K_Aa (CME term)
        # This is the conditional mean embedding mu_x(a) for the subpopulation A_subpopulation.
        # K_X_mu_x_a = cme_term_inv_ @ K_Aa_subpopulation
        K_X_mu_x_a = self.cme_term_inv_ @ K_Aaprime_subpopulation # Shape (n_train, n_test)
        
        # Step 4b: The final tensor product kernel K_W(a', mu_x(a))
        # K_Wd' = K_A a' * K_X_mu_x_a (Element-wise product for the tensor product RKHS)
        K_Wa_test = K_Aa_counterfactual * K_X_mu_x_a 
        
        # 5. Final Prediction: W_matrix @ K_Wa_test 
        # Result: (1, n_train) @ (n_train, n_test) = (1, n_test)
        Y_pred = self.W_matrix_ @ K_Wa_test
        
        # Return predicted values of shape (n_test,1)
        return torch.t(Y_pred)

    def fit_predict(self, 
                    AX: Tuple[torch.Tensor, torch.Tensor], 
                    Y: torch.Tensor,
                    A_subpopulation: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Fit the model and predict outcomes.
        """
        self.fit(AX, Y)
        A_train, _ = AX
        
        return self.predict(A_train, A_subpopulation)