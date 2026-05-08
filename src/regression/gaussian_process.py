import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from typing import Callable, Tuple, Optional, Union, Dict

from torch_utils.kernel_utils import RBF


class GaussianProcessRegressor(BaseEstimator, RegressorMixin, nn.Module):
    """
    Gaussian Process Regression Model (PyTorch implementation).

    This model performs hyperparameter optimization by minimizing the 
    negative Log Marginal Likelihood (-LML).
    """
    def __init__(
        self,
        noise_variance: float = 1e-6, 
        kernel: Union[Callable, str] = 'RBF',
        kernel_params: Optional[dict] = None,
        **kwargs,
    ) -> None:
        nn.Module.__init__(self)

        self.noise_variance = noise_variance
        
        # --- Optimization Parameters ---
        self.normalize_labels = kwargs.pop('normalize_labels', True)
        self.optimize_kernel_params = kwargs.pop('optimize_kernel_params', True)
        self.max_iter_optimizer = kwargs.pop('max_iter_optimizer', 100)
        self.lr_optimizer = kwargs.pop('lr_optimizer', 1e-3)
        self.kernel_params_optimization_tol = kwargs.pop('kernel_params_optimization_tol', 1e-9)

        # --- Kernel Initialization (Matching KRR) ---
        if isinstance(kernel, Callable):
            self.kernel = kernel
            if kernel_params:
                if hasattr(self.kernel, 'set_params'):
                    self.kernel.set_params(**kernel_params)
        elif isinstance(kernel, str):
            if kernel == "RBF":
                # Assumes RBF is a class inheriting from Kernel/nn.Module
                from utils.kernel_utils import RBF 
                self.kernel = RBF(**(kernel_params if kernel_params else {}))
            else:
                raise NotImplementedError("Possible Kernels: RBF")
        else:
             raise TypeError("Kernel must be callable (nn.Module) or string.")

        # --- Stored Fitted Variables ---
        self.coefs_ = None  # (K + noise*I)^-1 @ Y
        self.X_fit_ = None
        self.L_ = None      # Cholesky decomposition L
        self.y_mean_ = 0.0
        self.y_std_ = 1.0


    def _optimize_kernel_params(self, X: torch.Tensor, Y: torch.Tensor):
        """
        Optimize kernel parameters by minimizing -LML using the Adam optimizer.
        """
        # Collect parameters that require gradients (i.e., kernel hyperparameters)
        trainable_params = [p for p in self.kernel.parameters() if p.requires_grad]
        
        if not trainable_params:
            print("Warning: No kernel parameters found for optimization.")
            return

        optimizer = optim.Adam(trainable_params, lr=self.lr_optimizer)
        
        # Initial likelihood for comparison
        nll_old = torch.tensor(np.inf, dtype=X.dtype, device=X.device)
        
        # Optimization loop
        for j in tqdm(range(self.max_iter_optimizer)):
            optimizer.zero_grad()
            
            # Compute negative log marginal likelihood
            nll = _log_marginal_likelihood(self.kernel, X, Y, self.noise_variance)
            
            # Backpropagation
            nll.backward()
            
            # Optimization step
            optimizer.step()

            nll_val = nll.item()
            
            # Check convergence tolerance
            if abs(nll_val - nll_old.item()) < self.kernel_params_optimization_tol:
                print(f"Optimization converged after {j+1} iterations. NLL: {nll_val:.4f}")
                break
            
            nll_old = torch.tensor(nll_val)
            
        print(f"Optimization finished. Final NLL: {nll_val:.4f}")


    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> 'GaussianProcessRegressor':
        """
        Fit the Gaussian Process Regression model.
        """
        # 1. Data and Device Setup (Ensure targets are [n, 1])
        device = X.device
        dtype = torch.get_default_dtype()
        if hasattr(self.kernel, 'log_length_scale'):
            device = self.kernel.log_length_scale.device
            dtype = self.kernel.log_length_scale.dtype

        X = X.to(device).to(dtype)
        Y = Y.to(device).to(dtype).reshape(-1, 1) # Ensure Y is (n, 1)
        n = X.shape[0]

        # 2. Label Normalization
        if self.normalize_labels:
            self.y_mean_ = torch.mean(Y, dim=0, keepdim=True)
            self.y_std_ = torch.std(Y, dim=0, keepdim=True)
            Y = (Y - self.y_mean_) / self.y_std_

        # 3. Kernel Parameter Optimization (Hyperparameter Tuning)
        if self.optimize_kernel_params:
            self._optimize_kernel_params(X, Y.squeeze()) # Pass Y as (n,) to the optimizer

        # 4. Final Solution Calculation (Save L and coefs)
        # K_XX + noise_variance * I
        K_XX = self.kernel(X)
        K_XX_plus_noise = K_XX + self.noise_variance * torch.eye(n, dtype=dtype, device=device)
        
        # Cholesky Decomposition (L)
        self.L_ = torch.linalg.cholesky(K_XX_plus_noise)
        
        # Calculate alpha_coefs: alpha = (K + noise*I)^-1 @ Y
        v = torch.linalg.solve_triangular(self.L_, Y, upper=False)
        self.coefs_ = torch.linalg.solve_triangular(self.L_.T, v, upper=True).detach() # (n, 1)
        
        self.X_fit_ = X.detach()
        
        return self


    def predict(self, X: torch.Tensor, return_std: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict mean and standard deviation for the given input data.
        """
        with torch.no_grad():
            if self.coefs_ is None:
                raise RuntimeError("Model must be fitted before calling predict.")
                
            # Ensure input is on the same device/dtype as the fitted data
            X = X.to(self.X_fit_.device).to(self.X_fit_.dtype)
            n_test = X.shape[0]
            
            # K_Xx = K(X_fit_, X) (Kernel matrix between training and test points)
            K_Xx = self.kernel(self.X_fit_, X) # (n_train, n_test)
            
            # 1. Mean Prediction: mu = K_Xx.T @ alpha
            # Result is (n_test, 1)
            y_pred_normalized = K_Xx.T @ self.coefs_ 
            
            # Rescale the prediction
            y_pred = self.y_std_ * y_pred_normalized + self.y_mean_
            
            if not return_std:
                return y_pred.squeeze(-1) # Return (n_test,)

            # 2. Variance Prediction: var = diag(K_xx) - diag(v.T @ v)
            # v = L \ K_Xx = torch.linalg.solve_triangular(L, K_Xx)
            v_mat = torch.linalg.solve_triangular(self.L_, K_Xx, upper=False) # (n_train, n_test)
            
            # Sum squares of v along the training dimension (axis 0)
            # sum(v*v, axis=0) -> (n_test,)
            v_sum_squares = torch.sum(v_mat * v_mat, dim=0)

            # K_xx is the test-test kernel matrix diagonal (K(x*, x*))
            # Note: K(X, X) call needs to be modified if K(X, Y) is used for test-test.
            K_xx_diag = self.kernel(X, X).diag() # K(X, X) gives matrix, diag() extracts diagonal
            
            # var = K_xx_diag - v_sum_squares
            pred_var = K_xx_diag.reshape(n_test, 1) - v_sum_squares.reshape(n_test, 1)
            
            # Ensure variance is non-negative due to numerical stability
            pred_var = torch.clamp(pred_var, min=0.0) 
            
            # Standard Deviation
            pred_std = torch.sqrt(pred_var) * self.y_std_
            
            # Return Mean (n_test,) and Std (n_test,)
        return y_pred.squeeze(-1), pred_std.squeeze(-1)


    def fit_predict(self, X: torch.Tensor, Y: torch.Tensor, return_std: bool = True):
        """Fit the model and predict target values and optional standard deviation."""
        self.fit(X, Y)
        return self.predict(X, return_std=return_std)

def _log_marginal_likelihood(kernel: torch.nn.Module, X: torch.Tensor, Y: torch.Tensor, noise_variance: float) -> torch.Tensor:
    """
    Computes the negative Log Marginal Likelihood (-LML) of the Gaussian Process.
    
    Parameters
    ----------
    kernel : torch.nn.Module
        Kernel function instance.
    X : torch.Tensor
        Input data (n_samples, n_features).
    Y : torch.Tensor
        Target values (n_samples,).
    noise_variance : float
        Noise variance (alpha/lambda added to the diagonal).

    Returns
    -------
    torch.Tensor (scalar)
        Negative Log Marginal Likelihood.
    """
    n = X.shape[0]
    device = X.device
    dtype = X.dtype

    # 1. Compute Kernel Matrix K_XX + noise_variance * I
    K_XX = kernel(X)
    
    # Add noise variance to the diagonal
    K_XX_plus_noise = K_XX + noise_variance * torch.eye(n, dtype=dtype, device=device)
    
    # 2. Cholesky Decomposition
    try:
        L = torch.linalg.cholesky(K_XX_plus_noise)
    except RuntimeError:
        # Handle non-positive definite case (numerical instability)
        # Often occurs during optimization. Add jitter and retry.
        jitter = 1e-6
        K_XX_plus_noise = K_XX_plus_noise + jitter * torch.eye(n, dtype=dtype, device=device)
        L = torch.linalg.cholesky(K_XX_plus_noise)
        
    # 3. Calculate LML components
    
    # Solve for alpha: alpha = (K + noise*I)^-1 @ Y
    # alpha = L.T \ (L \ Y) using torch.linalg.solve_triangular
    # L \ Y is solved by: L @ x = Y  =>  x = torch.linalg.solve_triangular(L, Y, upper=False)
    # L.T \ x is solved by: L.T @ alpha = x => alpha = torch.linalg.solve_triangular(L.T, x, upper=True)

    # Ensure Y is (n, 1) for matrix operations
    Y_reshaped = Y.reshape(-1, 1) 
    
    v = torch.linalg.solve_triangular(L, Y_reshaped, upper=False)
    alpha_coefs = torch.linalg.solve_triangular(L.T, v, upper=True) # Renamed coefs to avoid conflict

    # LML = -0.5 * Y.T @ alpha - sum(log(diag(L))) - (n/2) * log(2*pi)
    
    # Term 1: -0.5 * Y.T @ alpha
    term1 = -0.5 * Y_reshaped.T @ alpha_coefs

    # Term 2: - sum(log(diag(L)))
    term2 = - torch.sum(torch.log(torch.diag(L)))

    # Term 3: - (n/2) * log(2*pi)
    term3 = -(n / 2) * torch.log(torch.tensor(2 * np.pi, dtype=dtype, device=device))

    log_likelihood = term1 + term2 + term3
    
    # PyTorch convention is often to minimize NEGATIVE log likelihood
    return -log_likelihood.squeeze()