import torch
import yaml
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from datetime import datetime

import sys
# sys.path.append("../../src/other_methods/DRKPV")
sys.path.append("../..")

# from utils.kernel_utils import Kernel, ColumnwiseRBF, RBF
# from causal_models.doubly_robust_pcl import DoublyRobustKernelProxyATE
# from utils.ml_utils import data_transform
# from utils.linalg_utils import cartesian_product, make_psd

from src.simulation_utils.data_generating_processes import generate_synthetic_PCL_ATE_data
from src.torch_utils.scalers import TorchStandardScaler, TorchIdentityTransformer
from src.python_utils.visualization_utils import plot_observational_vs_causal_effect
from src.torch_utils.kernel_utils import RBF, ColumnwiseRBF
from src.causal_learning.doubly_robust_kernel_proxy_causal_learning import DoublyRobustKernelProxyATE
from src.python_utils.visualization_utils import plot_regression_dashboard, plot_observational_vs_causal_effect, plot_causal_effect_estimation

if not os.path.exists("../Results"):
    os.mkdir("../Results")

if not os.path.exists("../Results/Synthetic_Low_Dim_Experiment"):
    os.mkdir("../Results/Synthetic_Low_Dim_Experiment")

data_size_list = [2000, 5000, 10000, 15000, 20000]
seed_list = np.arange(0, 3000, 100)

# Get current date and time in YYYY-MM-DD_HH-MM-SS format
current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
data_sizes_str = "_".join(map(str, data_size_list))

csv_name_for_results = f"../Results/Synthetic_Low_Dim_Experiment/DRKPV_Synthetic_Low_Dim_Results_{data_sizes_str}_{current_date}.csv"

print(f"Running simulation for sizes: {data_size_list}")

print("Running the script Synthetic_Low_Dim_Experiment to generate results and save at {}".format(csv_name_for_results))

df_results = pd.DataFrame(columns = ["Algorithm", "Data_Size", "Seed", "Causal_MSE"])
# # Data generating Process

# In[2]:

for n_plus_m in data_size_list:
    for seed in seed_list:
        np.random.seed(seed)

        n_sample = n_plus_m
        U, W, Z, A, Y, do_A, EY_do_A = generate_synthetic_PCL_ATE_data(size = n_sample, seed = seed, do_A_range = (-1, 2))

        device = "cuda"
        Z_tensor = torch.tensor(Z, dtype=torch.float32).to(device)
        W_tensor = torch.tensor(W, dtype=torch.float32).to(device)
        A_tensor = torch.tensor(A, dtype=torch.float32).to(device)
        Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)
        A_test_tensor = torch.tensor(do_A, dtype=torch.float32).to(device)

        transformers = [TorchStandardScaler(), TorchStandardScaler(), TorchStandardScaler(), TorchStandardScaler()]

        A_tensor_transformed = transformers[0].fit_transform(A_tensor)
        Y_tensor_transformed = transformers[1].fit_transform(Y_tensor)
        Z_tensor_transformed = transformers[2].fit_transform(Z_tensor)
        W_tensor_transformed = transformers[3].fit_transform(W_tensor)
        A_test_tensor_transformed = transformers[0].transform(A_test_tensor)


        treatment_bridge_algo_param_dict_default = {
                                                    "kernel_A" : RBF(use_length_scale_heuristic = True),
                                                    "kernel_W" : ColumnwiseRBF(use_length_scale_heuristic = True), 
                                                    "kernel_Z" : RBF(use_length_scale_heuristic = True),
                                                    # "kernel_X" : RBF(use_length_scale_heuristic = True),
                                                    "lambda_" : 1e-3,
                                                    "eta" : 1e-3,
                                                    "lambda2_" : 1e-3,
                                                    "optimize_lambda_parameters" : True,
                                                    "optimize_eta_parameter" : True,
                                                    "lambda_optimization_range" : (1e-5, 1.0),
                                                    "eta_optimization_range" : (5e-4, 1e-1),
                                                    "stage1_perc" : 0.5,
                                                    "regularization_grid_points" : 25, 
                                                    "make_psd_eps" : 5e-6,
                                                    "label_variance_in_lambda_opt" : 0.0,
                                                    "label_variance_in_eta_opt" : 3.0,
                                                    }
        outcome_bridge_kpv_algo_param_dict_default = {
                                                    "algorithm_name" : "Kernel_Proxy_Variable",
                                                    "kernel_A" : RBF(use_length_scale_heuristic = True),
                                                    "kernel_W" : RBF(use_length_scale_heuristic = True),
                                                    "kernel_Z" : RBF(use_length_scale_heuristic = True),
                                                    # "kernel_X" : RBF(use_length_scale_heuristic = True),      
                                                    "lambda1_" : 0.1,
                                                    "lambda2_" : 0.1,
                                                    "optimize_lambda1_parameter" : True,
                                                    "optimize_lambda2_parameter" : True,
                                                    "lambda1_optimization_range" : (1e-5, 1.0),
                                                    "lambda2_optimization_range" : (1e-5, 1.0),
                                                    "stage1_perc" : 0.5,
                                                    "regularization_grid_points" : 25, 
                                                    "make_psd_eps" : 5e-6,
                                                        }

        model_DR = DoublyRobustKernelProxyATE(
                                            treatment_bridge_params = treatment_bridge_algo_param_dict_default,
                                            outcome_bridge_params = outcome_bridge_kpv_algo_param_dict_default,
                                            lambda_DR_optimization_range = (1e-5, 1.0),
                                            regularization_grid_points = 25, 
                                            device = device
                                            )

        model_DR.fit((A_tensor_transformed, Z_tensor_transformed, W_tensor_transformed), Y_tensor_transformed)
        do_A_size = do_A.shape[0]
        f_struct_pred_DR = model_DR.predict(A_test_tensor_transformed, transformers[1]).reshape(do_A_size, -1).detach().cpu().numpy()

        structured_pred_mse = (np.mean((f_struct_pred_DR.reshape(-1, 1) - EY_do_A.reshape(-1, 1)) ** 2)).item()
        structured_pred_mae = (np.mean(np.abs(f_struct_pred_DR.reshape(-1, 1) - EY_do_A.reshape(-1, 1)))).item()
        print("Structured function test set MSE: {}".format(structured_pred_mse))
        print("Structured function test set MAE: {}".format(structured_pred_mae))

        Result_Dict = {
                        "Algorithm" : "DRKPV",
                        "Data_Size" : n_plus_m,
                        "Seed" : seed,
                        "Causal_MSE" : structured_pred_mse,
                    }

        df_results = pd.concat([df_results, pd.DataFrame([Result_Dict])], ignore_index=True)

        df_results.to_csv(csv_name_for_results, index=False)