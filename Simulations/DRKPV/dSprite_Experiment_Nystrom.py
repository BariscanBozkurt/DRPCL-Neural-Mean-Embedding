#!/usr/bin/env python
# coding: utf-8

import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import sys
sys.path.append("../..")

from src.simulation_utils.data_generating_processes import dSprite_ProxyVariable_DatasetV2
from src.torch_utils.scalers import TorchStandardScaler, TorchIdentityTransformer
from src.python_utils.visualization_utils import plot_observational_vs_causal_effect
from src.torch_utils.kernel_utils import RBF, ColumnwiseRBF
from src.causal_learning.approximate_kernel_proxy_causal_learning import DoublyRobustKernelProxyDoseResponseNystrom


# Get current date and time in YYYY-MM-DD_HH-MM-SS format
current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

if not os.path.exists("../Results"):
    os.mkdir("../Results")

if not os.path.exists("../Results/dSprite_Experiment"):
    os.mkdir("../Results/dSprite_Experiment")


# data_size_list = [2000, 5000, 10000, 15000]
data_size_list = [20000]
seed_list = np.arange(0, 3000, 100)

data_sizes_str = "_".join(map(str, data_size_list))
csv_name_for_results = (
    f"../Results/dSprite_Experiment/"
    f"DRKPV_Nystrom_dSprite_Results_{current_date}_{data_sizes_str}data.csv"
)

print("Running the script dSprite_Experiment_Nystrom_DRKPV to generate results and save at {}".format(csv_name_for_results))

df_results = pd.DataFrame(columns=["Algorithm", "Data_Size", "Seed", "Causal_MSE"])


# # Data generating Process

data_path = "../../data/dsprite"

for n_plus_m in data_size_list:
    for seed in seed_list:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        n_sample = n_plus_m

        dsprite_data_generator = dSprite_ProxyVariable_DatasetV2()
        A, Y, Z, W, do_A, EY_do_A = dsprite_data_generator.generate_dsprite_pv(
            data_path,
            n_sample=n_sample,
            generate_test=True,
            rand_seed=seed,
        )

        # The notebook for the Nyström estimator uses CPU.
        # This is usually safer for kernel matrices and avoids GPU memory issues.
        device = "cuda"

        Z_tensor = torch.tensor(Z, dtype=torch.float32).to(device)
        W_tensor = torch.tensor(W, dtype=torch.float32).to(device)
        A_tensor = torch.tensor(A, dtype=torch.float32).to(device)
        Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)
        A_test_tensor = torch.tensor(do_A, dtype=torch.float32).to(device)

        transformers = [
            TorchStandardScaler(),
            TorchStandardScaler(),
            TorchStandardScaler(),
            TorchStandardScaler(),
        ]

        A_tensor_transformed = transformers[0].fit_transform(A_tensor)
        Y_tensor_transformed = transformers[1].fit_transform(Y_tensor)
        Z_tensor_transformed = transformers[2].fit_transform(Z_tensor)
        W_tensor_transformed = transformers[3].fit_transform(W_tensor)
        A_test_tensor_transformed = transformers[0].transform(A_test_tensor)


        treatment_bridge_algo_param_dict_default = {
            "kernel_A": RBF(use_length_scale_heuristic=True, length_scale_heuristic_quantile=0.95),
            "kernel_W": RBF(use_length_scale_heuristic=True, length_scale_heuristic_quantile=0.95),
            "kernel_Z": RBF(use_length_scale_heuristic=True),
            # "kernel_X": RBF(use_length_scale_heuristic=True),
            "lambda1_": 1e-3,
            "eta": 1e-3,
            "lambda2_": 1e-3,
            "nystrom_first_stage_m": 500,
            "nystrom_third_stage_m": 500,
            "stage1_perc": 0.5,
            "model_seed": seed,
            "make_psd_eps": 5e-6,
            "device": device,
        }

        outcome_bridge_kpv_algo_param_dict_default = {
            "algorithm_name": "Kernel_Proxy_Variable",
            "kernel_A": RBF(use_length_scale_heuristic=True),
            "kernel_W": RBF(use_length_scale_heuristic=True),
            "kernel_Z": RBF(use_length_scale_heuristic=True),
            # "kernel_X": RBF(use_length_scale_heuristic=True),
            "lambda1_": 1e-3,
            "lambda2_": 1e-3,
            "stage1_perc": 0.5,
            "nystrom_first_stage_m": 500,
            "nystrom_second_stage_m": 500,
            "model_seed": seed + 1,
            "make_psd_eps": 5e-6,
            "device": device,
        }

        model_DR = DoublyRobustKernelProxyDoseResponseNystrom(
            treatment_bridge_params=treatment_bridge_algo_param_dict_default,
            outcome_bridge_params=outcome_bridge_kpv_algo_param_dict_default,
            lambda_DR=1e-3,
            nystrom_m=500,
            model_seed=seed + 2,
            make_psd_eps=5e-6,
            device=device,
        )

        model_DR.fit(
            (A_tensor_transformed, Z_tensor_transformed, W_tensor_transformed),
            Y_tensor_transformed,
        )

        do_A_size = do_A.shape[0]

        # # Doubly Robust Estimation

        f_struct_pred_DR = model_DR.predict(
            A_test_tensor_transformed,
            transformers[1],
        ).reshape(do_A_size, -1).detach().cpu().numpy()

        structured_pred_mse = (
            np.mean((f_struct_pred_DR.reshape(-1, 1) - EY_do_A.reshape(-1, 1)) ** 2)
        ).item()
        structured_pred_mae = (
            np.mean(np.abs(f_struct_pred_DR.reshape(-1, 1) - EY_do_A.reshape(-1, 1)))
        ).item()

        print("Data size: {}, Seed: {}, Algorithm: DRKPV_Nystrom".format(n_plus_m, seed))
        print("Structured function test set MSE: {}".format(structured_pred_mse))
        print("Structured function test set MAE: {}".format(structured_pred_mae))

        Result_Dict = {
            "Algorithm": "DRKPV_Nystrom",
            "Data_Size": n_plus_m,
            "Seed": seed,
            "Causal_MSE": structured_pred_mse,
        }

        df_results = pd.concat([df_results, pd.DataFrame([Result_Dict])], ignore_index=True)

        # # Outcome Bridge-based Estimation: KPV Nyström

        f_struct_pred_om = model_DR.outcome_bridge_algo.predict(
            A_test_tensor_transformed,
            transformers[1],
        ).reshape(do_A_size, -1).detach().cpu().numpy()

        structured_pred_mse = (
            np.mean((f_struct_pred_om.reshape(-1, 1) - EY_do_A.reshape(-1, 1)) ** 2)
        ).item()
        structured_pred_mae = (
            np.mean(np.abs(f_struct_pred_om.reshape(-1, 1) - EY_do_A.reshape(-1, 1)))
        ).item()

        print("Data size: {}, Seed: {}, Algorithm: KPV_Nystrom".format(n_plus_m, seed))
        print("Structured function test set MSE: {}".format(structured_pred_mse))
        print("Structured function test set MAE: {}".format(structured_pred_mae))

        Result_Dict = {
            "Algorithm": "KPV_Nystrom",
            "Data_Size": n_plus_m,
            "Seed": seed,
            "Causal_MSE": structured_pred_mse,
        }

        df_results = pd.concat([df_results, pd.DataFrame([Result_Dict])], ignore_index=True)

        # # Treatment Bridge-based Estimation: KAP Nyström

        f_struct_pred_tm = model_DR.treatment_bridge_algo.predict(
            A_test_tensor_transformed,
            transformers[1],
        ).reshape(do_A_size, -1).detach().cpu().numpy()

        structured_pred_mse = (
            np.mean((f_struct_pred_tm.reshape(-1, 1) - EY_do_A.reshape(-1, 1)) ** 2)
        ).item()
        structured_pred_mae = (
            np.mean(np.abs(f_struct_pred_tm.reshape(-1, 1) - EY_do_A.reshape(-1, 1)))
        ).item()

        print("Data size: {}, Seed: {}, Algorithm: KAP_Nystrom".format(n_plus_m, seed))
        print("Structured function test set MSE: {}".format(structured_pred_mse))
        print("Structured function test set MAE: {}".format(structured_pred_mae))

        Result_Dict = {
            "Algorithm": "KAP_Nystrom",
            "Data_Size": n_plus_m,
            "Seed": seed,
            "Causal_MSE": structured_pred_mse,
        }

        df_results = pd.concat([df_results, pd.DataFrame([Result_Dict])], ignore_index=True)

        df_results.to_csv(csv_name_for_results, index=False)

        del model_DR, A, Y, Z, W, do_A, EY_do_A
        del A_tensor, Y_tensor, Z_tensor, W_tensor, A_test_tensor
        del A_tensor_transformed, Y_tensor_transformed, Z_tensor_transformed, W_tensor_transformed, A_test_tensor_transformed

        if torch.cuda.is_available():
            torch.cuda.empty_cache()