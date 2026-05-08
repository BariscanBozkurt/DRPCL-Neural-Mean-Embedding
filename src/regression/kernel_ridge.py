import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin

from typing import Callable, Tuple, Optional, Union, Dict
import numpy as np

import sys
sys.path.append("..")
from torch_utils.ml_utils import lambda_objective_loocv
from torch_utils.kernel_utils import RBF
from torch_utils.linear_algebra import make_psd


class KernelRidgeRegression(BaseEstimator, RegressorMixin, nn.Module):
    """
    Kernel Ridge Regression (KRR) model implemented using PyTorch.

    The model finds the closed-form solution for the coefficients (dual form).
    Regularization parameter (lambda_) is optimized via LOOCV grid search if enabled.
    """
    def __init__(
        self,
        kernel: Union[Callable, str],
        lambda_: float = 1e-3, # Regularization parameter
        optimize_regularization_parameters: bool = True,
        lambda_optimization_range: Tuple[float, float] = (1e-9, 1.0),
        **kwargs
    ) -> None:
        # Initialize nn.Module and BaseEstimator/RegressorMixin (Order is important)
        nn.Module.__init__(self) 
        
        # --- Parameter Assignment (using lambda_) ---
        self.lambda_ = lambda_
        self.optimize_regularization_parameters = optimize_regularization_parameters
        self.lambda_optimization_range = lambda_optimization_range

        # Parse kwargs
        kernel_params = kwargs.pop('kernel_params', {})
        self.regularization_grid_points = kwargs.pop('regularization_grid_points', 150)
        self.make_psd_eps = kwargs.pop('make_psd_eps', 1e-9)
        self.kernel_params = kernel_params
        
        # --- Kernel Initialization ---
        if isinstance(kernel, Callable):
            self.kernel = kernel
            if kernel_params:
                 # Attempt to set parameters if kernel supports it (e.g., RBF)
                 if hasattr(self.kernel, 'set_params'):
                     self.kernel.set_params(**kernel_params)
        
        elif isinstance(kernel, str):
            if kernel == "RBF":
                # Assumes RBF is a class inheriting from Kernel/nn.Module
                self.kernel = RBF(**kernel_params)
            else:
                raise NotImplementedError("Possible Kernels: RBF")
        
        else:
             raise TypeError("Kernel must be callable (nn.Module) or string.")

        # Non-PyTorch state variables (coefficients and training data)
        self.coefs_ = None
        self.X_fit_ = None


    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "KernelRidgeRegression":
        """
        Fit the KRR model by solving the closed-form dual problem.
        """
        # 1. Data and Device Setup
        device = X.device
        dtype = torch.get_default_dtype()
        if hasattr(self.kernel, 'log_length_scale'):
            device = self.kernel.log_length_scale.device
            dtype = self.kernel.log_length_scale.dtype

        X = X.to(device).to(dtype)
        y = y.to(device).to(dtype)
        n = X.shape[0]
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        # 2. Compute training kernel matrix K(X, X)
        K_XX = self.kernel(X)

        # 3. Optimization of Regularization Parameter (lambda_)
        if self.optimize_regularization_parameters:
            l_range = self.lambda_optimization_range
            grid_points = self.regularization_grid_points
            
            # Create logarithmic grid of lambda values
            log_min = torch.log(torch.tensor(l_range[0], dtype=dtype, device=device))
            log_max = torch.log(torch.tensor(l_range[1], dtype=dtype, device=device))
            lambda_list = torch.exp(torch.linspace(log_min, log_max, grid_points, device=device)) # Using lambda_list
            
            objective_list = torch.zeros(grid_points, dtype=dtype, device=device)
            
            # Use torch.no_grad() for efficient grid search
            with torch.no_grad():
                for i, lambda_val in enumerate(lambda_list):
                    # Call the external LOOCV utility function
                    objective_list[i] = lambda_objective_loocv(lambda_val, K_XX, y, make_psd_eps = self.make_psd_eps)
            
            # Find the optimal lambda and update the parameter
            best_lambda = lambda_list[torch.argmin(objective_list).item()]
            self.lambda_ = best_lambda.item() # Storing the final value


        # 4. Solve for Coefficients (closed-form: coefs = (K + n*lambda*I)^-1 @ y)
        lambda_term = n * self.lambda_
        
        # A = K_XX + n * lambda_ * I
        identity_matrix = torch.eye(n, dtype=dtype, device=device)
        A = K_XX + lambda_term * identity_matrix

        # Use make_psd for stability before inversion
        A_psd = make_psd(A, eps=self.make_psd_eps)

        # Solve for coefficients (pinv for stability)
        self.coefs_ = torch.linalg.pinv(A_psd) @ y
        self.X_fit_ = X
        
        return self


    def predict(self, X: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict target values for the given input data.
        """
        if self.coefs_ is None:
            raise RuntimeError("Model must be fitted before calling predict.")
            
        # Ensure input is on the same device/dtype as the fitted data
        X = X.to(self.X_fit_.device).to(self.X_fit_.dtype)
        
        # K_Xx = K(X_fit_, X) (Kernel matrix between training and test points)
        K_Xx = self.kernel(self.X_fit_, X)
        
        # Prediction: K_Xx.T @ coefs
        return (K_Xx.T @ self.coefs_).squeeze(-1)


    def fit_predict(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Fit the model and predict target values."""
        self.fit(X, y)
        return self.predict(X)