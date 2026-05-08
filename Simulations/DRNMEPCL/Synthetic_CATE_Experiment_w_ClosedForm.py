#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import numpy as np
import pandas as pd
from IPython.display import clear_output, display

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
from datetime import datetime

import os 
import sys
sys.path.append("../../src")

from simulation_utils.data_generating_processes import generate_synthetic_CATE_data
from torch_utils.scalers import TorchStandardScaler, TorchIdentityTransformer, TorchLogStandardScaler
from torch_utils.dataset_classes import HeterogeneousProxyDataset
from torch_utils.densratio import HeterogeneousCausalKDEDensRatioTorch
from torch_utils.networks import ConditionalMeanMLP, EnsembleConditionalMeanMLP
from torch_utils.torch_loss import get_loss_function
from python_utils.helpers import slice_tuple

# Outcome bridge imports
from simulation_utils.outcome_pcl_net_configs import OutcomePCLConfig
from simulation_utils.outcome_pcl_net_nn_structures import build_nets_for_outcome_pcl_net_synthetic_cate_experiment
from simulation_utils.outcome_pcl_net_optimizers import build_sgd_optimizers_for_outcome_pcl_net, build_adam_optimizers_for_outcome_pcl_net
from neural_causal_learning.proxy_outcome_neural_mean_embedding import (HeterogeneousOutcomeBridgePCLNET, 
                                                                              train_deep_feature_proxy_closed_form_ate_model)
from neural_causal_learning.proxy_outcome_neural_mean_embedding import create_third_stage_dataset_for_outcome_pcl_net_cate

# Treatment bridge imports
from simulation_utils.treatment_pcl_net_nn_structures import build_nets_for_treatment_pcl_net_synthetic_cate_experiment
from simulation_utils.treatment_pcl_net_optimizers import build_sgd_optimizers_for_treatment_pcl_net, build_adam_optimizers_for_treatment_pcl_net
from simulation_utils.treatment_pcl_net_configs import TreatmentPCLConfig
from neural_causal_learning.proxy_treatment_neural_mean_embedding import create_third_stage_dataset_for_treatment_pcl_net_cate, train_third_stage_treatment_pcl_net, train_third_stage_treatment_pcl_net_ensemble
from neural_causal_learning.proxy_treatment_neural_mean_embedding import HeterogeneousTreatmentBridgePCLNET, train_treatment_pcl_net_ate_model

from neural_causal_learning.doubly_robust_proxy_neural_mean_embedding import create_third_stage_dataset_for_dr_proxy_neural_mean_embedding_cate, create_third_stage_dataset_for_dr_proxy_neural_mean_embedding_cate_v2

from dataclasses import asdict

def get_meta_params(outcome_cfg, treatment_cfg, train_kwargs):
    """Flattens all configs into a single dictionary with prefixes."""
    meta = {}
    
    # Helper to handle types
    def process_val(v):
        if isinstance(v, (list, tuple)):
            return str(v)  # Convert [1, 2] or (1, 2) to "[1, 2]" or "(1, 2)"
        if isinstance(v, (int, float, str, bool)):
            return v
        return None # Skip dicts or other complex objects

    # Extract Outcome Bridge Params
    for k, v in asdict(outcome_cfg).items():
        val = process_val(v)
        if val is not None:
            meta[f"out_{k}"] = val
            
    # Extract Treatment Bridge Params
    for k, v in asdict(treatment_cfg).items():
        val = process_val(v)
        if val is not None:
            meta[f"treat_{k}"] = val
            
    # Extract Third Stage Params
    for k, v in train_kwargs.items():
        val = process_val(v)
        if val is not None:
            meta[f"third_{k}"] = val
    
    return meta

# # Create Dataset and Dataloader

# In[2]:
if not os.path.exists("../Results"):
    os.mkdir("../Results")

if not os.path.exists("../Results/CATE_Synthetic_Experiment"):
    os.mkdir("../Results/CATE_Synthetic_Experiment")

data_size_list = [2000, 5000, 10000, 15000, 20000]
seed_list = np.arange(0, 3000, 100)

# Get current date and time in YYYY-MM-DD_HH-MM-SS format
current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
data_sizes_str = "_".join(map(str, data_size_list))

csv_name_for_results = f"../Results/CATE_Synthetic_Experiment/DRPCLNET_SyntheticCATE_Results_with_ClosedForm_{data_sizes_str}_{current_date}.csv"

print(f"Running simulation for sizes: {data_size_list}")

print("Running the script Synthetic_CATE_Experiment_with_ClosedForm to generate results and save at {}".format(csv_name_for_results))

df_results = pd.DataFrame(columns = ["Algorithm", "Data_Size", "Seed", "Causal_MSE"])

sigma = 0.1
uniform_noise_upper_bound = 0.1,
uniform_noise_lower_bound = -0.1,

for n_plus_m in data_size_list:
    for seed in seed_list:
        np.random.seed(seed)
        torch.manual_seed(seed)

        n_sample = n_plus_m

        U, W, Z, V, A, Y, covariate_v_test, do_A, EY_do_A = generate_synthetic_CATE_data(   n_sample, 
                                                                                            sigma,
                                                                                            uniform_noise_upper_bound,
                                                                                            uniform_noise_lower_bound,
                                                                                            seed = seed)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Z_tensor = torch.tensor(Z, dtype=torch.float32)
        W_tensor = torch.tensor(W, dtype=torch.float32)
        A_tensor = torch.tensor(A, dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.float32)
        V_tensor = torch.tensor(V, dtype=torch.float32)

        do_A_tensor = torch.tensor(do_A, dtype = torch.float32)
        covariate_v_test_tensor = torch.tensor(covariate_v_test, dtype = torch.float32)
        EY_do_A_tensor = torch.tensor(EY_do_A, dtype = torch.float32)

        #### Perform Density Ratio Estimation to Generate Labels of the Second Stage Regression. You are welcome to use your favorite density ratio estimator.
        transformers = [TorchStandardScaler(), TorchStandardScaler(), TorchStandardScaler(), TorchStandardScaler(), TorchStandardScaler()]
        dens_ratio_transformer = TorchIdentityTransformer()

        A_tensor_transformed = transformers[0].fit_transform(A_tensor).to(device)
        V_tensor_transformed = transformers[4].fit_transform(V_tensor).to(device)
        W_tensor_transformed = transformers[3].fit_transform(W_tensor).to(device)

        ratio_estimator = HeterogeneousCausalKDEDensRatioTorch(device = device)
        ratio_estimator.fit(A_tensor_transformed, V_tensor_transformed, W_tensor_transformed)
        dens_ratio_tensor = ratio_estimator.predict_ratio(A_tensor_transformed, V_tensor_transformed, W_tensor_transformed).to("cpu")

        inlier_dens_ratio_range = (1e-2, 10.0)
        inlier_indices = ((dens_ratio_tensor > inlier_dens_ratio_range[0]) & (dens_ratio_tensor < inlier_dens_ratio_range[1])).view(-1)
        n_sample = inlier_indices.sum().item()
        print("Ratio of inlier data picked according to density ratio estimation is {}".format(n_sample / A_tensor_transformed.shape[0]))
        ##### Filter to get inlier data
        A_tensor, Y_tensor, Z_tensor, W_tensor, dens_ratio_tensor = A_tensor[inlier_indices], Y_tensor[inlier_indices], Z_tensor[inlier_indices], W_tensor[inlier_indices], dens_ratio_tensor[inlier_indices]

        #### Create Dataloaders with data normalization objects
        val_perc = 0.1
        stage1_perc = 0.75
        stage2_perc = 0.75
        if stage2_perc is None:
            stage2_perc = 1 - stage1_perc

        data_indices = np.random.permutation(n_sample)
        train_indices = data_indices[:int(n_sample * (1 - val_perc))]
        val_indices = data_indices[int(n_sample * (1 - val_perc)):]
        train_data_size = train_indices.shape[0]

        # Fit the transformers only in training data to avoid data leakage for validation set!
        transformers[0].fit(A_tensor[train_indices])
        transformers[1].fit(Y_tensor[train_indices])
        transformers[2].fit(Z_tensor[train_indices])
        transformers[3].fit(W_tensor[train_indices])
        transformers[4].fit(V_tensor[train_indices])
        dens_ratio_transformer.fit(dens_ratio_tensor[train_indices])

        if (stage1_perc > 0.) & (stage1_perc < 1.):
            stage1_data_size = int(train_data_size * stage1_perc)
            stage2_data_size = int(train_data_size * stage2_perc)
            stage1_idx, stage2_idx = train_indices[:stage1_data_size], train_indices[-stage2_data_size:]
        else:
            stage1_data_size, stage2_data_size = train_data_size, train_data_size
            stage1_idx, stage2_idx = train_indices, train_indices


        first_stage_train_dataset = HeterogeneousProxyDataset( slice_tuple((A_tensor, Y_tensor, Z_tensor, W_tensor, V_tensor), stage1_idx), 
                                                    dens_ratio = dens_ratio_tensor[stage1_idx],
                                                    dens_ratio_transformer = dens_ratio_transformer,
                                                    transformers = transformers,
                                                    device = "cpu"
                                                )

        second_stage_train_dataset = HeterogeneousProxyDataset(slice_tuple((A_tensor, Y_tensor, Z_tensor, W_tensor, V_tensor), stage2_idx), 
                                                    dens_ratio = dens_ratio_tensor[stage2_idx],
                                                    dens_ratio_transformer = dens_ratio_transformer,
                                                    transformers = transformers,
                                                    device = "cpu"
                                                )

        validation_dataset = HeterogeneousProxyDataset(slice_tuple((A_tensor, Y_tensor, Z_tensor, W_tensor, V_tensor), val_indices), 
                                            dens_ratio = dens_ratio_tensor[val_indices],
                                            dens_ratio_transformer = dens_ratio_transformer,
                                            transformers = transformers,
                                            device = "cpu"
                                                )

        batch_size=1024
        first_stage_train_dataloader = DataLoader(first_stage_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        second_stage_train_dataloader = DataLoader(second_stage_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        #### Setup the configurations for training the two networks and the loss functions and regularization parameters. 
        # SETUP CONFIGURATION for Outcome Model
        outcome_cfg = OutcomePCLConfig.synthetic_cate()
        outcome_cfg.second_stage_loss_name = "mse"
        outcome_cfg.device = device
        outcome_cfg.plot_loss = False
        # SETUP CONFIGURATION for Treatment Model
        treatment_cfg = TreatmentPCLConfig.synthetic_cate()
        treatment_cfg.second_stage_loss_name = "mse"
        treatment_cfg.device = device
        treatment_cfg.plot_loss = False
        # SETUP CONFIGURATION for Third Stage DR Model/Treatment Model
        INPUT_DIM = 2
        OUTPUT_DIM = 1 

        # # 1. Define hyperparameters
        train_kwargs = {
            'lr': 1e-4 if n_plus_m > 10000 else 1e-3,  # Use a smaller learning rate for larger datasets,
            'weight_decay': 1e-6,
            'n_epochs': 100,
            'loss_fn': nn.MSELoss(),
            'gap_penalty_weight': 0.0,
            'log_per_epoch': 10,
            'input_dim': INPUT_DIM,
            'output_dim': OUTPUT_DIM,
            'hidden_dims': [64, 128],
            'dropout_rate': 0.05
        }
        n_ensemble = 5
        # Capture metadata for this specific simulation run
        meta_params = get_meta_params(outcome_cfg, treatment_cfg, train_kwargs)

        # # Outcome model training

        # In[3]:

        (first_stage_featurizer, 
        treatment_featurizer, 
        covariate_featurizer,
        outcome_proxy_featurizer,) = build_nets_for_outcome_pcl_net_synthetic_cate_experiment(outcome_cfg.device)

        outcome_model = HeterogeneousOutcomeBridgePCLNET(   first_stage_featurizer,
                                                            treatment_featurizer, 
                                                            covariate_featurizer,
                                                            outcome_proxy_featurizer, 
                                                            None,
                                                            None,
                                                            device=device
                                                        )


        (optimizers_stage1, 
        optimizers_stage2, 
        schedulers_stage1, 
        schedulers_stage2) = build_adam_optimizers_for_outcome_pcl_net( outcome_model, 
                                                                        outcome_cfg.lr_first_stage,
                                                                        outcome_cfg.lr_second_stage_ax,
                                                                        outcome_cfg.lr_second_stage_w,
                                                                        outcome_cfg.weight_decay,
                                                                        gamma = outcome_cfg.scheduler_gamma)

        # 5. UNPACK REGULARIZERS
        reg1, reg2_inner, reg2_outer = outcome_cfg.regularizers

        # 6. BUILD LOSS FUNCTIONS
        loss_fn_1 = get_loss_function(outcome_cfg.first_stage_loss_name, outcome_cfg.first_stage_loss_kwargs)
        loss_fn_2 = get_loss_function(outcome_cfg.second_stage_loss_name, outcome_cfg.second_stage_loss_kwargs)

        outcome_model = train_deep_feature_proxy_closed_form_ate_model(
                                                                        pcl_model = outcome_model,
                                                                        first_stage_train_dataloader = first_stage_train_dataloader,
                                                                        second_stage_train_dataloader = second_stage_train_dataloader,
                                                                        stage1_optimizers = optimizers_stage1,
                                                                        stage2_optimizers = optimizers_stage2,
                                                                        stage1_schedulers = schedulers_stage1,
                                                                        stage2_schedulers = schedulers_stage2,

                                                                        n_epochs = outcome_cfg.n_epochs,
                                                                        stage1_iter = outcome_cfg.stage1_iter,
                                                                        stage2_iter = outcome_cfg.stage2_iter,

                                                                        first_stage_final_layer_regularizer = reg1,
                                                                        second_stage_final_layer_regularizer = reg2_outer,
                                                                        second_stage_first_final_layer_regularizer = reg2_inner,
                                                                        regularizer_annealing_method=outcome_cfg.reg_annealing_method,
                                                                        consider_prev_weight=outcome_cfg.consider_prev_weight,

                                                                        first_stage_loss_fn = loss_fn_1,
                                                                        second_stage_loss_fn = loss_fn_2,
                                                                        second_stage_head_lr = outcome_cfg.second_stage_head_lr,
                                                                        second_stage_head_steps = outcome_cfg.second_stage_head_steps,
                                                                        # Logging
                                                                        log_per_epoch=outcome_cfg.log_per_epoch,
                                                                        plot_loss=outcome_cfg.plot_loss,
                                                                        validation_dataloader=validation_dataloader,
                                                                        do_A = do_A_tensor,
                                                                        EY_do_A = EY_do_A_tensor
        )

        third_stage_dataset_outcome, third_stage_dataset_val_outcome = create_third_stage_dataset_for_outcome_pcl_net_cate(   outcome_model, 
                                                                                                            second_stage_train_dataloader,
                                                                                                            validation_dataloader,
                                                                                                            device = device)


        INPUT_DIM_OM = 8
        OUTPUT_DIM_OM = 16 
        train_kwargs_om = copy.deepcopy(train_kwargs)
        train_kwargs_om["input_dim"] = INPUT_DIM_OM
        train_kwargs_om["output_dim"] = OUTPUT_DIM_OM

        # 2. Train the Ensemble
        # Note: Pass 'ConditionalMeanEstimator' (the class), NOT 'ConditionalMeanEstimator(...)' (the object)
        third_stage_net_outcome = train_third_stage_treatment_pcl_net_ensemble(
            model_class=ConditionalMeanMLP,
            dataloader=DataLoader(third_stage_dataset_outcome, batch_size=batch_size, shuffle=True, num_workers=0),
            val_dataloader=DataLoader(third_stage_dataset_val_outcome, batch_size=batch_size, shuffle=True, num_workers=0),
            n_members=n_ensemble,             # 5-10 models reduces variance significantly
            device = device,
            **train_kwargs_om
        )

        f_struct_pred_om = outcome_model.pred_structural_function( do_A_tensor.to(device),
                                                                covariate_v_test_tensor.to(device),
                                                                third_stage_net_outcome,
                                                                second_stage_train_dataset.transformers[0].to(device),
                                                                second_stage_train_dataset.transformers[4].to(device),
                                                                second_stage_train_dataset.transformers[1].to(device)
                                                                ).detach().cpu().numpy()

        structured_pred_mse_om = (np.mean(((f_struct_pred_om).reshape(-1, 1) - EY_do_A.reshape(-1, 1)) ** 2))
        # print("Structured function test set MSE - Outcome Model: {}".format(structured_pred_mse_om))

        Result_Dict = {
                        "Algorithm" : "OutcomeBridgePCLNET",
                        "Data_Size" : n_plus_m,
                        "Seed" : seed,
                        "Causal_MSE" : structured_pred_mse_om,
                        **meta_params # Merges all hyperparameters into this row
                    }

        df_results = pd.concat([df_results, pd.DataFrame([Result_Dict])], ignore_index=True)

        df_results.to_csv(csv_name_for_results, index=False)

        # # Treatment model training

        # In[4]:

        (first_stage_featurizer, 
        second_stage_ax_featurizer, 
        treatment_proxy_featurizer) = build_nets_for_treatment_pcl_net_synthetic_cate_experiment(treatment_cfg.device)

        treatment_model = HeterogeneousTreatmentBridgePCLNET(first_stage_featurizer,
                                                            second_stage_ax_featurizer, 
                                                            treatment_proxy_featurizer,
                                                            dens_ratio_transformer = copy.deepcopy(dens_ratio_transformer),
                                                            device = treatment_cfg.device
                                                    )



        (stage1_optimizers, 
        stage2_optimizers, 
        stage1_schedulers, 
        stage2_schedulers) = build_adam_optimizers_for_treatment_pcl_net(  treatment_model,
                                                                            lr_first_stage=treatment_cfg.lr_first_stage,         # <--- Auto-handles 0.0 logic to disable the first stage
                                                                            lr_second_stage_ax=treatment_cfg.lr_second_stage_ax,
                                                                            lr_second_stage_z=treatment_cfg.lr_second_stage_z,
                                                                            weight_decay=treatment_cfg.weight_decay,
                                                                            # Note: If using Adam, momentum is usually betas=(mom, 0.999)
                                                                            # If using SGD, pass momentum directly. 
                                                                            # Adjust your build_optimizer function signature if needed.
                                                                            gamma=treatment_cfg.scheduler_gamma)

        # 5. UNPACK REGULARIZERS
        reg1, reg2_inner, reg2_outer = treatment_cfg.regularizers

        # 6. BUILD LOSS FUNCTIONS
        loss_fn_1 = get_loss_function(treatment_cfg.first_stage_loss_name, treatment_cfg.first_stage_loss_kwargs)
        loss_fn_2 = get_loss_function(treatment_cfg.second_stage_loss_name, treatment_cfg.second_stage_loss_kwargs)

        treatment_model = train_treatment_pcl_net_ate_model(
                                                            model=treatment_model,
                                                            first_stage_train_dataloader=first_stage_train_dataloader,
                                                            second_stage_train_dataloader=second_stage_train_dataloader,
                                                            stage1_optimizers=stage1_optimizers,
                                                            stage2_optimizers=stage2_optimizers,
                                                            stage1_schedulers=stage1_schedulers,
                                                            stage2_schedulers=stage2_schedulers,

                                                            n_epochs=treatment_cfg.n_epochs,
                                                            stage1_iter=treatment_cfg.stage1_iter,
                                                            stage2_iter=treatment_cfg.stage2_iter,

                                                            # Regularization
                                                            first_stage_final_layer_regularizer=reg1,
                                                            second_stage_first_final_layer_regularizer=reg2_inner,
                                                            second_stage_final_layer_regularizer=reg2_outer,
                                                            regularizer_annealing_method=treatment_cfg.reg_annealing_method,
                                                            consider_prev_weight=treatment_cfg.consider_prev_weight,

                                                            # Losses
                                                            # Note: Check if your train function expects 'first_stage_loss_fn' or just 'loss_fn'
                                                            # Based on your previous snippet, it seems to accept arguments via **kwargs or named args
                                                            first_stage_loss_fn=loss_fn_1,
                                                            second_stage_loss_fn=loss_fn_2,
                                                            second_stage_head_lr = treatment_cfg.second_stage_head_lr,
                                                            second_stage_head_steps = treatment_cfg.second_stage_head_steps,
                                                            negative_penalty=treatment_cfg.negative_penalty,

                                                            # Logging
                                                            log_per_epoch=treatment_cfg.log_per_epoch,
                                                            plot_loss=treatment_cfg.plot_loss,
                                                            validation_dataloader=validation_dataloader,
                                                        )

        treatment_featurizer = torch.nn.Identity()
        (third_stage_dataset_treatment, 
        third_stage_dataset_val_treatment) = create_third_stage_dataset_for_treatment_pcl_net_cate( treatment_model, 
                                                                                                    second_stage_train_dataloader,
                                                                                                    validation_dataloader,
                                                                                                    TorchStandardScaler(),
                                                                                                    TorchStandardScaler(),
                                                                                                    dens_ratio_pred_tolerance = 100.5,
                                                                                                    device = device)

        # 2. Train the Ensemble
        # Note: Pass 'ConditionalMeanEstimator' (the class), NOT 'ConditionalMeanEstimator(...)' (the object)
        third_stage_net_treatment = train_third_stage_treatment_pcl_net_ensemble(
            model_class=ConditionalMeanMLP,
            dataloader=DataLoader(third_stage_dataset_treatment, batch_size=batch_size, shuffle=True, num_workers=0),
            val_dataloader=DataLoader(third_stage_dataset_val_treatment, batch_size=batch_size, shuffle=True, num_workers=0),
            n_members=n_ensemble,             # 5-10 models reduces variance significantly
            device = device,
            **train_kwargs
        )

        do_A_transformed = second_stage_train_dataloader.dataset.transformers[0].to(device).transform(torch.tensor(do_A).to(device).to(torch.float32))
        covariate_V_transformed = second_stage_train_dataloader.dataset.transformers[4].to(device).transform(torch.tensor(covariate_v_test).to(device).to(torch.float32))  

        cate_input = third_stage_dataset_treatment.input_transformer.to(device).transform(torch.hstack([do_A_transformed, covariate_V_transformed]))
        f_struct_pred_tm = third_stage_dataset_treatment.outcome_transformer.to(device).inverse_transform(third_stage_net_treatment(cate_input)).detach().cpu().numpy()

        structured_pred_mse_tm = (np.mean(((f_struct_pred_tm).reshape(-1, 1) - EY_do_A.reshape(-1, 1)) ** 2))
        # print("Structured function test set MSE - Treatment Model: {}".format(structured_pred_mse_tm))
        Result_Dict = {
                        "Algorithm" : "TreatmentBridgePCLNET",
                        "Data_Size" : n_plus_m,
                        "Seed" : seed,
                        "Causal_MSE" : structured_pred_mse_tm,
                        **meta_params # Merges all hyperparameters into this row
                    }

        df_results = pd.concat([df_results, pd.DataFrame([Result_Dict])], ignore_index=True)

        df_results.to_csv(csv_name_for_results, index=False)

        # # Doubly Robust Training - Version 1

        # In[5]:


        (third_stage_dataset_dr, 
        third_stage_dataset_val_dr) = create_third_stage_dataset_for_dr_proxy_neural_mean_embedding_cate(    outcome_model,
                                                                                                            treatment_model, 
                                                                                                            second_stage_train_dataloader,
                                                                                                            validation_dataloader,
                                                                                                            TorchStandardScaler(),
                                                                                                            TorchStandardScaler(),
                                                                                                            dens_ratio_pred_tolerance = 100.5,
                                                                                                            device = device)


        # 2. Train the Ensemble
        # Note: Pass 'ConditionalMeanEstimator' (the class), NOT 'ConditionalMeanEstimator(...)' (the object)
        third_stage_net_dr = train_third_stage_treatment_pcl_net_ensemble(
            model_class=ConditionalMeanMLP,
            dataloader=DataLoader(third_stage_dataset_dr, batch_size=batch_size, shuffle=True, num_workers=0),
            val_dataloader=DataLoader(third_stage_dataset_val_dr, batch_size=batch_size, shuffle=True, num_workers=0),
            n_members=n_ensemble,             # 5-10 models reduces variance significantly
            device = device,
            **train_kwargs
        )

        do_A_transformed = second_stage_train_dataloader.dataset.transformers[0].to(device).transform(torch.tensor(do_A).to(device).to(torch.float32))
        covariate_V_transformed = second_stage_train_dataloader.dataset.transformers[4].to(device).transform(torch.tensor(covariate_v_test).to(device).to(torch.float32))  

        cate_input = third_stage_dataset_dr.input_transformer.to(device).transform(torch.hstack([do_A_transformed, covariate_V_transformed]))
        f_struct_pred_slack = third_stage_dataset_dr.outcome_transformer.to(device).inverse_transform(third_stage_net_dr(cate_input)).detach().cpu().numpy()
        f_struct_pred_dr = f_struct_pred_om + f_struct_pred_slack

        structured_pred_mse_dr = (np.mean(((f_struct_pred_dr).reshape(-1, 1) - EY_do_A.reshape(-1, 1)) ** 2))
        Result_Dict = {
                        "Algorithm" : "DRPCLNET_Version1",
                        "Data_Size" : n_plus_m,
                        "Seed" : seed,
                        "Causal_MSE" : structured_pred_mse_dr,
                        **meta_params # Merges all hyperparameters into this row
                    }

        df_results = pd.concat([df_results, pd.DataFrame([Result_Dict])], ignore_index=True)

        df_results.to_csv(csv_name_for_results, index=False)

        print("Structured function test set MSE - Outcome Model: {}".format(structured_pred_mse_om))
        print("Structured function test set MSE - Treatment Model: {}".format(structured_pred_mse_tm))
        print("Structured function test set MSE - Doubly Robust Model: {}".format(structured_pred_mse_dr))

        # # Doubly Robust Training - Version 2

        # In[6]:


        (third_stage_dataset_dr, 
        third_stage_dataset_val_dr) = create_third_stage_dataset_for_dr_proxy_neural_mean_embedding_cate_v2( outcome_model,
                                                                                                            treatment_model, 
                                                                                                            second_stage_train_dataloader,
                                                                                                            validation_dataloader,
                                                                                                            TorchStandardScaler(),
                                                                                                            TorchStandardScaler(),
                                                                                                            dens_ratio_pred_tolerance = 100.5,
                                                                                                            device = device)

        # 2. Train the Ensemble
        # Note: Pass 'ConditionalMeanEstimator' (the class), NOT 'ConditionalMeanEstimator(...)' (the object)
        third_stage_net_dr = train_third_stage_treatment_pcl_net_ensemble(
            model_class=ConditionalMeanMLP,
            dataloader=DataLoader(third_stage_dataset_dr, batch_size=batch_size, shuffle=True, num_workers=0),
            val_dataloader=DataLoader(third_stage_dataset_val_dr, batch_size=batch_size, shuffle=True, num_workers=0),
            n_members=n_ensemble,             # 5-10 models reduces variance significantly
            device = device,
            **train_kwargs
        )

        do_A_transformed = second_stage_train_dataloader.dataset.transformers[0].to(device).transform(torch.tensor(do_A).to(device).to(torch.float32))
        covariate_V_transformed = second_stage_train_dataloader.dataset.transformers[4].to(device).transform(torch.tensor(covariate_v_test).to(device).to(torch.float32))  

        cate_input = third_stage_dataset_dr.input_transformer.to(device).transform(torch.hstack([do_A_transformed, covariate_V_transformed]))
        f_struct_pred_slack = third_stage_dataset_dr.outcome_transformer.to(device).inverse_transform(third_stage_net_dr(cate_input)).detach().cpu().numpy()
        f_struct_pred_dr = f_struct_pred_om + f_struct_pred_tm - f_struct_pred_slack

        structured_pred_mse_dr = (np.mean(((f_struct_pred_dr).reshape(-1, 1) - EY_do_A.reshape(-1, 1)) ** 2))
        Result_Dict = {
                        "Algorithm" : "DRPCLNET_Version2",
                        "Data_Size" : n_plus_m,
                        "Seed" : seed,
                        "Causal_MSE" : structured_pred_mse_dr,
                        **meta_params # Merges all hyperparameters into this row
                    }

        df_results = pd.concat([df_results, pd.DataFrame([Result_Dict])], ignore_index=True)

        df_results.to_csv(csv_name_for_results, index=False)

        print("Structured function test set MSE - Outcome Model: {}".format(structured_pred_mse_om))
        print("Structured function test set MSE - Treatment Model: {}".format(structured_pred_mse_tm))
        print("Structured function test set MSE - Doubly Robust Model: {}".format(structured_pred_mse_dr))


        del outcome_model, treatment_model, third_stage_net_outcome, third_stage_net_treatment, third_stage_net_dr
        # In[ ]:
