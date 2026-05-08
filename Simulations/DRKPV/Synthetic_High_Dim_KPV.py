import torch
import yaml
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from datetime import datetime

import sys
sys.path.append("../..")

from src.simulation_utils.data_generating_processes import PCL_Synthetic_High_DimNew as PCL_Synthetic_High_Dim
from src.torch_utils.scalers import TorchStandardScaler, TorchIdentityTransformer
from src.python_utils.visualization_utils import plot_observational_vs_causal_effect
from src.torch_utils.kernel_utils import RBF, ColumnwiseRBF
from src.causal_learning.kernel_proxy_causal_learning import KernelProxyVariableDoseResponse
from src.python_utils.visualization_utils import plot_regression_dashboard, plot_observational_vs_causal_effect, plot_causal_effect_estimation
from src.torch_utils.densratio import HighDimCausalDensityRatioEstimator

if not os.path.exists("../Results"):
    os.mkdir("../Results")

if not os.path.exists("../Results/Synthetic_High_Dim_Experiment_NewVersion"):
    os.mkdir("../Results/Synthetic_High_Dim_Experiment_NewVersion")

data_size_list = [2000, 5000, 10000, 15000, 20000]
seed_list = np.arange(0, 3000, 100)

# Get current date and time in YYYY-MM-DD_HH-MM-SS format
current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
data_sizes_str = "_".join(map(str, data_size_list))

csv_name_for_results = f"../Results/Synthetic_High_Dim_Experiment_NewVersion/KPV_Synthetic_High_Dim_Results_{data_sizes_str}_{current_date}.csv"

print(f"Running simulation for sizes: {data_size_list}")

print("Running the script Synthetic_High_Dim_KPV to generate results and save at {}".format(csv_name_for_results))

df_results = pd.DataFrame(columns = ["Algorithm", "Data_Size", "Seed", "Causal_MSE"])
# # Data generating Process

# In[2]:

dim_z = 10
dim_w = 10
dim_x = 100
type_ = "quadratic"
do_A_range = (0.0, 1.0)
do_A_size = 100

for n_plus_m in data_size_list:
    for seed in seed_list:
        np.random.seed(seed)
        torch.manual_seed(seed)
        print("Running KPV with data size {} and seed {}".format(n_plus_m, seed))
        n_sample = n_plus_m

        data_generator = PCL_Synthetic_High_Dim(seed, n_sample, dim_z = dim_z, dim_w = dim_w, dim_x = dim_x, type_ = type_)

        A, Z, W, Y, X = data_generator.generatate_high()

        do_A, EY_do_A = PCL_Synthetic_High_Dim.generate_test_effect(do_A_range[0],
                                                                    do_A_range[1],
                                                                    do_A_size, 
                                                                    type_,
                                                                    dim_z,
                                                                    dim_w,
                                                                    dim_x)

        # device = "cpu"
        # Z_tensor = torch.tensor(Z, dtype=torch.float32).to(device)
        # W_tensor = torch.tensor(W, dtype=torch.float32).to(device)
        # A_tensor = torch.tensor(A, dtype=torch.float32).to(device)
        # Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)
        # X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        # do_A_tensor = torch.tensor(do_A, dtype=torch.float32).to(device)

        # # transformers = [TorchStandardScaler(), TorchStandardScaler(), TorchStandardScaler(), TorchStandardScaler()]
        # transformers = [TorchStandardScaler(), TorchIdentityTransformer(), TorchStandardScaler(), TorchStandardScaler(), TorchStandardScaler()]
        # # transformers = [TorchIdentityTransformer(), TorchIdentityTransformer(), TorchIdentityTransformer(), TorchIdentityTransformer()]

        # A_tensor_transformed = transformers[0].fit_transform(A_tensor).detach().cpu().numpy()
        # Y_tensor_transformed = transformers[1].fit_transform(Y_tensor).detach().cpu().numpy()
        # Z_tensor_transformed = transformers[2].fit_transform(Z_tensor).detach().cpu().numpy()
        # W_tensor_transformed = transformers[3].fit_transform(W_tensor).detach().cpu().numpy()
        # X_tensor_transformed = transformers[4].fit_transform(X_tensor).detach().cpu().numpy()
        # do_A_tensor_transformed = transformers[0].transform(do_A_tensor).detach().cpu().numpy()

        # #####################################################################################
        # ############## The following part is not necessary step for PKDR algorithm
        # ############## Nonetheless, we run this density ratio part to apply inlier filtering
        # ############## so that for simulations it is going to use the same set of data with 
        # ############## our proposed method.
        # #####################################################################################

        # # MAF Config for High Dim
        # ratio_estimator = HighDimCausalDensityRatioEstimator(
        #     features_dim=1,  # A is 1D
        #     context_dim=110,   # W and X are 110-D
        #     hidden_features=(128,64,32), # Keep it small for N=500
        #     transforms=4,    # Shallow flow avoids overfitting
        #     flow_type="maf", 
        #     activation=nn.Tanh,
        #     lr=1e-4,
        #     n_epochs=100,    
        #     batch_size=512,  
        #     device=device,
        #     verbose = False,
        # )

        # ratio_estimator.fit(A_tensor_transformed, W_tensor_transformed, X_tensor_transformed)
        # dens_ratio_tensor = ratio_estimator.predict_ratio(A_tensor_transformed, W_tensor_transformed, X_tensor_transformed).to("cpu")
        # inlier_dens_ratio_range = (1e-2, 5.0)
        # inlier_indices = ((dens_ratio_tensor > inlier_dens_ratio_range[0]) & (dens_ratio_tensor < inlier_dens_ratio_range[1])).view(-1)
        # n_sample = inlier_indices.sum().item()
        # print("Ratio of inlier data picked according to density ratio estimation is {}".format(n_sample / A_tensor_transformed.shape[0]))
        # ##### Filter to get inlier data
        # A_tensor, Y_tensor, Z_tensor, W_tensor, X_tensor = A_tensor[inlier_indices], Y_tensor[inlier_indices], Z_tensor[inlier_indices], W_tensor[inlier_indices], X_tensor[inlier_indices]

        # transformers = [TorchStandardScaler(), TorchIdentityTransformer(), TorchStandardScaler(), TorchStandardScaler(), TorchStandardScaler()]

        device = "cuda"
        Z_tensor = torch.tensor(Z, dtype=torch.float32).to(device)
        W_tensor = torch.tensor(W, dtype=torch.float32).to(device)
        A_tensor = torch.tensor(A, dtype=torch.float32).to(device)
        Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        A_test_tensor = torch.tensor(do_A, dtype=torch.float32).to(device)

        transformers = [TorchStandardScaler(), TorchStandardScaler(), TorchStandardScaler(), TorchStandardScaler(), TorchStandardScaler()]

        A_tensor_transformed = transformers[0].fit_transform(A_tensor)
        Y_tensor_transformed = transformers[1].fit_transform(Y_tensor)
        Z_tensor_transformed = transformers[2].fit_transform(Z_tensor)
        W_tensor_transformed = transformers[3].fit_transform(W_tensor)
        X_tensor_transformed = transformers[4].fit_transform(X_tensor)
        A_test_tensor_transformed = transformers[0].transform(A_test_tensor)

        outcome_bridge_kpv_algo_param_dict_default = {
                                                    "algorithm_name" : "Kernel_Proxy_Variable",
                                                    "kernel_A" : RBF(use_length_scale_heuristic = True),
                                                    "kernel_W" : ColumnwiseRBF(use_length_scale_heuristic = True),
                                                    "kernel_Z" : RBF(use_length_scale_heuristic = True),
                                                    "kernel_X" : RBF(use_length_scale_heuristic = True),      
                                                    "lambda1_" : 0.1,
                                                    "lambda2_" : 0.1,
                                                    "optimize_lambda1_parameter" : True,
                                                    "optimize_lambda2_parameter" : True,
                                                    "lambda1_optimization_range" : (1e-4, 1.0),
                                                    "lambda2_optimization_range" : (1e-4, 1.0),
                                                    "stage1_perc" : 0.5,
                                                    "regularization_grid_points" : 25, 
                                                    "make_psd_eps" : 5e-6,
                                                        }

        model = KernelProxyVariableDoseResponse(**outcome_bridge_kpv_algo_param_dict_default,
                                        device = device
                                            )

        model.fit((A_tensor_transformed, Z_tensor_transformed, W_tensor_transformed), Y_tensor_transformed)
        f_struct_pred = model.predict(A_test_tensor_transformed, transformers[1]).reshape(do_A_size, -1).detach().cpu().numpy()

        structured_pred_mse = (np.mean((f_struct_pred.reshape(-1, 1) - EY_do_A.reshape(-1, 1)) ** 2)).item()
        structured_pred_mae = (np.mean(np.abs(f_struct_pred.reshape(-1, 1) - EY_do_A.reshape(-1, 1)))).item()
        print("Structured function test set MSE: {}".format(structured_pred_mse))
        print("Structured function test set MAE: {}".format(structured_pred_mae))

        Result_Dict = {
                        "Algorithm" : "KPV",
                        "Data_Size" : n_plus_m,
                        "Seed" : seed,
                        "Causal_MSE" : structured_pred_mse,
                    }

        df_results = pd.concat([df_results, pd.DataFrame([Result_Dict])], ignore_index=True)

        df_results.to_csv(csv_name_for_results, index=False)