import torch
import yaml
import os
import numpy as np
import pandas as pd
import torch.nn as nn
from copy import deepcopy
from datetime import datetime

import sys
sys.path.append("../../src/other_methods")
sys.path.append("../..")

from PKDR.utils.data_class import PVTrainDataSet,PVTestDataSet
from PKDR.model.rkhs.Trainer import RKHS_Trainer

from src.simulation_utils.data_generating_processes import PCL_Synthetic_High_Dim
from src.torch_utils.scalers import TorchStandardScaler, TorchIdentityTransformer
from src.python_utils.visualization_utils import plot_observational_vs_causal_effect, plot_causal_effect_estimation
from src.torch_utils.densratio import HighDimCausalDensityRatioEstimator

cfg = "../../src/other_methods/PKDR/configs/rkhs.yaml"
cnf_cfg = "../../src/other_methods/PKDR/configs/cnf.yaml"

with open(cfg) as stream:
    try:
        cfg = yaml.safe_load(stream)
        # print(yaml.safe_load(stream))
    except yaml.YAMLError as exc:
        print(exc)

class obj(object):
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(k, (list, tuple)):
                setattr(self, k, [obj(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, obj(v) if isinstance(v, dict) else v)

if not os.path.exists("../Results"):
    os.mkdir("../Results")

if not os.path.exists("../Results/Synthetic_High_Dim_Experiment"):
    os.mkdir("../Results/Synthetic_High_Dim_Experiment")

data_size_list = [2000, 5000, 10000, 15000, 20000]
seed_list = np.arange(0, 3000, 100)

# Get current date and time in YYYY-MM-DD_HH-MM-SS format
current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
data_sizes_str = "_".join(map(str, data_size_list))

csv_name_for_results = f"../Results/Synthetic_High_Dim_Experiment/PKDR_SyntheticHighDim_Results_{data_sizes_str}_{current_date}.csv"

print(f"Running simulation for sizes: {data_size_list}")

print("Running the script PKDR_Synthetic_High_Dim to generate results and save at {}".format(csv_name_for_results))

df_results = pd.DataFrame(columns = ["Algorithm", "Data_Size", "Seed", "Causal_MSE"])

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
        print("Running PKDR with data size {} and seed {}".format(n_plus_m, seed))
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

        device = "cpu"
        Z_tensor = torch.tensor(Z, dtype=torch.float32).to(device)
        W_tensor = torch.tensor(W, dtype=torch.float32).to(device)
        A_tensor = torch.tensor(A, dtype=torch.float32).to(device)
        Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        do_A_tensor = torch.tensor(do_A, dtype=torch.float32).to(device)

        # transformers = [TorchStandardScaler(), TorchStandardScaler(), TorchStandardScaler(), TorchStandardScaler()]
        transformers = [TorchStandardScaler(), TorchIdentityTransformer(), TorchStandardScaler(), TorchStandardScaler(), TorchStandardScaler()]
        # transformers = [TorchIdentityTransformer(), TorchIdentityTransformer(), TorchIdentityTransformer(), TorchIdentityTransformer()]

        A_tensor_transformed = transformers[0].fit_transform(A_tensor).detach().cpu().numpy()
        Y_tensor_transformed = transformers[1].fit_transform(Y_tensor).detach().cpu().numpy()
        Z_tensor_transformed = transformers[2].fit_transform(Z_tensor).detach().cpu().numpy()
        W_tensor_transformed = transformers[3].fit_transform(W_tensor).detach().cpu().numpy()
        X_tensor_transformed = transformers[4].fit_transform(X_tensor).detach().cpu().numpy()
        do_A_tensor_transformed = transformers[0].transform(do_A_tensor).detach().cpu().numpy()

        #####################################################################################
        ############## The following part is not necessary step for PKDR algorithm
        ############## Nonetheless, we run this density ratio part to apply inlier filtering
        ############## so that for simulations it is going to use the same set of data with 
        ############## our proposed method.
        #####################################################################################

        # MAF Config for High Dim
        ratio_estimator = HighDimCausalDensityRatioEstimator(
            features_dim=1,  # A is 1D
            context_dim=110,   # W and X are 110-D
            hidden_features=(128,64,32), # Keep it small for N=500
            transforms=4,    # Shallow flow avoids overfitting
            flow_type="maf", 
            activation=nn.Tanh,
            lr=1e-4,
            n_epochs=100,    
            batch_size=512,  
            device=device,
            verbose = False,
        )

        ratio_estimator.fit(A_tensor_transformed, W_tensor_transformed, X_tensor_transformed)
        dens_ratio_tensor = ratio_estimator.predict_ratio(A_tensor_transformed, W_tensor_transformed, X_tensor_transformed).to("cpu")
        inlier_dens_ratio_range = (1e-2, 10.0)
        inlier_indices = ((dens_ratio_tensor > inlier_dens_ratio_range[0]) & (dens_ratio_tensor < inlier_dens_ratio_range[1])).view(-1)
        n_sample = inlier_indices.sum().item()
        print("Ratio of inlier data picked according to density ratio estimation is {}".format(n_sample / A_tensor_transformed.shape[0]))
        ##### Filter to get inlier data
        A_tensor, Y_tensor, Z_tensor, W_tensor, X_tensor = A_tensor[inlier_indices], Y_tensor[inlier_indices], Z_tensor[inlier_indices], W_tensor[inlier_indices], X_tensor[inlier_indices]

        transformers = [TorchStandardScaler(), TorchIdentityTransformer(), TorchStandardScaler(), TorchStandardScaler(), TorchStandardScaler()]

        A_tensor_transformed = transformers[0].fit_transform(A_tensor).detach().cpu().numpy()
        Y_tensor_transformed = transformers[1].fit_transform(Y_tensor).detach().cpu().numpy()
        Z_tensor_transformed = transformers[2].fit_transform(Z_tensor).detach().cpu().numpy()
        W_tensor_transformed = transformers[3].fit_transform(W_tensor).detach().cpu().numpy()
        X_tensor_transformed = transformers[4].fit_transform(X_tensor).detach().cpu().numpy()
        do_A_tensor_transformed = transformers[0].transform(do_A_tensor).detach().cpu().numpy()
        #####################################################################################
        #####################################################################################
        #####################################################################################
        #####################################################################################

        data_size = A_tensor_transformed.shape[0]
        A_transformed = np.array(A_tensor_transformed).reshape(data_size, -1)
        Z_transformed = np.array(Z_tensor_transformed).reshape(data_size, -1)
        W_transformed = np.array(W_tensor_transformed).reshape(data_size, -1)
        Y_transformed = np.array(Y_tensor_transformed).reshape(data_size, -1)
        X_transformed = np.array(X_tensor_transformed).reshape(data_size, -1)

        train_dataset = PVTrainDataSet( treatment = A_transformed,
                                        treatment_proxy = Z_transformed,
                                        outcome_proxy = W_transformed,
                                        outcome = Y_transformed,
                                        backdoor = X_transformed)

        rkhs_train = RKHS_Trainer(train_dataset, **cfg)

        # Train q and h
        rkhs_train.fit_q_cv(type = "cnf", cnf = cnf_cfg)
        rkhs_train.fit_h_cv()
        do_A_size = do_A.shape[0]

        ATE_dr_transformed = rkhs_train._drtest(do_A_tensor_transformed, train_dataset)

        f_struct_pred = transformers[1].inverse_transform(torch.tensor(np.array(ATE_dr_transformed)).view(-1, 1)).detach().cpu().numpy()

        structured_pred_mse = (np.mean((f_struct_pred.reshape(-1, 1) - EY_do_A.reshape(-1, 1)) ** 2))
        structured_pred_mae = (np.mean(np.abs(f_struct_pred.reshape(-1, 1) - EY_do_A.reshape(-1, 1))))
        print("Structured function test set MSE: {}".format(structured_pred_mse))
        print("Structured function test set MAE: {}".format(structured_pred_mae))

        Result_Dict = {
                        "Algorithm" : "PKDR",
                        "Data_Size" : n_plus_m,
                        "Seed" : seed,
                        "Causal_MSE" : structured_pred_mse,
                    }

        df_results = pd.concat([df_results, pd.DataFrame([Result_Dict])], ignore_index=True)

        df_results.to_csv(csv_name_for_results)