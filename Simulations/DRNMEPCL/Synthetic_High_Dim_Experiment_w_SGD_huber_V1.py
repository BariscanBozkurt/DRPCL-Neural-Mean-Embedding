import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime
import os
import sys
import copy
import matplotlib.pyplot as plt

# Ensure src is in the path
sys.path.append("../../src")

# Utilities
from simulation_utils.data_generating_processes import PCL_Synthetic_High_DimNew as PCL_Synthetic_High_Dim
from torch_utils.scalers import TorchStandardScaler, TorchIdentityTransformer, TorchLogStandardScaler
from torch_utils.dataset_classes import ProxyDataset
from torch_utils.densratio import HighDimCausalDensityRatioEstimator, CausalRuLSIFTorch, CausalKDEDensRatioTorch
from torch_utils.networks import ConditionalMeanMLP
from torch_utils.torch_loss import get_loss_function
from python_utils.helpers import slice_tuple
from python_utils.visualization_utils import plot_regression_dashboard, plot_observational_vs_causal_effect, plot_causal_effect_estimation

# Treatment Bridge Network Imports
from simulation_utils.treatment_pcl_net_configs import TreatmentPCLConfig
from simulation_utils.treatment_pcl_net_nn_structures import build_nets_for_treatment_pcl_net_synthetic_high_dim_new_version_experiment as build_nets_for_treatment_pcl_net_synthetic_high_dim_experiment
from simulation_utils.treatment_pcl_net_optimizers import build_sgd_optimizers_for_treatment_pcl_net, build_adam_optimizers_for_treatment_pcl_net
from neural_causal_learning.proxy_treatment_neural_mean_embedding import create_third_stage_dataset_for_treatment_pcl_net_ate, train_third_stage_treatment_pcl_net, train_third_stage_treatment_pcl_net_ensemble
from neural_causal_learning.proxy_treatment_neural_mean_embedding_w_SGD import TreatmentBridgePCLNET, train_treatment_pcl_net_ate_model

# Outcome Bridge Network Imports
from simulation_utils.outcome_pcl_net_configs import OutcomePCLConfig
from simulation_utils.outcome_pcl_net_nn_structures import build_nets_for_outcome_pcl_net_synthetic_high_dim_new_version_experiment as build_nets_for_outcome_pcl_net_synthetic_high_dim_experiment
from simulation_utils.outcome_pcl_net_optimizers import build_sgd_optimizers_for_outcome_pcl_net, build_adam_optimizers_for_outcome_pcl_net
from neural_causal_learning.proxy_outcome_neural_mean_embedding_w_SGD import (OutcomeBridgePCLNET, 
                                                                              train_deep_feature_proxy_closed_form_ate_model)

# Doubly Robust Network Imports
from neural_causal_learning.doubly_robust_proxy_neural_mean_embedding import create_third_stage_dataset_for_dr_proxy_neural_mean_embedding_ate, create_third_stage_dataset_for_dr_proxy_neural_mean_embedding_ate_v2
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

def run_size_experiment(n_sample, seed_list, output_file):
    plot_things = False
    # Fixed Simulation Parameters
    dim_z = 10
    dim_w = 10
    dim_x = 100
    type_ = "quadratic"
    do_A_range = (0.0, 1.0)
    do_A_size = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    df_results = pd.DataFrame(columns=["Algorithm", "Data_Size", "Seed", "Causal_MSE"])
    n_plus_m = n_sample
    for seed in seed_list:
        np.random.seed(seed)
        torch.manual_seed(seed)


        data_generator = PCL_Synthetic_High_Dim(seed, n_sample, dim_z = dim_z, dim_w = dim_w, dim_x = dim_x, type_ = type_)

        A, Z, W, Y, X = data_generator.generatate_high()

        do_A, EY_do_A = PCL_Synthetic_High_Dim.generate_test_effect(  do_A_range[0],
                                                                    do_A_range[1],
                                                                    do_A_size, 
                                                                    type_,
                                                                    dim_z,
                                                                    dim_w,
                                                                    dim_x)

        # plot_observational_vs_causal_effect(A, Y, do_A, EY_do_A)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Z_tensor = torch.tensor(Z, dtype=torch.float32)
        W_tensor = torch.tensor(W, dtype=torch.float32)
        A_tensor = torch.tensor(A, dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.float32)
        X_tensor = torch.tensor(X, dtype=torch.float32)

        do_A_tensor = torch.tensor(do_A, dtype = torch.float32)
        EY_do_A_tensor = torch.tensor(EY_do_A, dtype = torch.float32)
        #### Perform Density Ratio Estimation to Generate Labels of the Second Stage Regression. You are welcome to use your favorite density ratio estimator.
        transformers = [TorchStandardScaler(), TorchStandardScaler(), TorchStandardScaler(), TorchStandardScaler(), TorchStandardScaler()]
        dens_ratio_transformer = TorchIdentityTransformer()

        A_tensor_transformed = transformers[0].fit_transform(A_tensor).to(device)
        # Y_tensor_transformed = transformers[1].fit_transform(Y_tensor)
        # Z_tensor_transformed = transformers[2].fit_transform(Z_tensor)
        W_tensor_transformed = transformers[3].fit_transform(W_tensor).to(device)
        X_tensor_transformed = transformers[4].fit_transform(X_tensor).to(device)

        # MAF Config for High Dim
        ratio_estimator = HighDimCausalDensityRatioEstimator(
            features_dim=1,  # A is 1D
            context_dim=110,   # W and X are 110-D
            hidden_features=(128,64,32), # Keep it small for N=500
            transforms=4,    # Shallow flow avoids overfitting
            flow_type="real_nvp", 
            activation=nn.Tanh,
            lr=1e-4,
            n_epochs=100,    
            batch_size=1024,  
            device=device,
            verbose = False,
        )

        if n_sample > 25000:
            dens_ratio_index = np.random.choice(A_tensor_transformed.shape[0], 20000, replace=False) # Subsample for faster density ratio estimation
            ratio_estimator.fit(A_tensor_transformed[dens_ratio_index], W_tensor_transformed[dens_ratio_index], X_tensor_transformed[dens_ratio_index])
        else:
            ratio_estimator.fit(A_tensor_transformed, W_tensor_transformed, X_tensor_transformed)
                    
        # ratio_estimator.fit(A_tensor_transformed, W_tensor_transformed, X_tensor_transformed)
        dens_ratio_tensor = ratio_estimator.predict_ratio(A_tensor_transformed, W_tensor_transformed, X_tensor_transformed).to("cpu")
        inlier_dens_ratio_range = (1e-2, 5.0)
        inlier_indices = ((dens_ratio_tensor > inlier_dens_ratio_range[0]) & (dens_ratio_tensor < inlier_dens_ratio_range[1])).view(-1)
        n_sample = inlier_indices.sum().item()
        print("Ratio of inlier data picked according to density ratio estimation is {}".format(n_sample / A_tensor_transformed.shape[0]))
        ##### Filter to get inlier data
        A_tensor, Y_tensor, Z_tensor, W_tensor, X_tensor, dens_ratio_tensor = A_tensor[inlier_indices], Y_tensor[inlier_indices], Z_tensor[inlier_indices], W_tensor[inlier_indices], X_tensor[inlier_indices], dens_ratio_tensor[inlier_indices]
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
        transformers[4].fit(X_tensor[train_indices])
        dens_ratio_transformer.fit(dens_ratio_tensor[train_indices])

        if (stage1_perc > 0.) & (stage1_perc < 1.):
            stage1_data_size = int(train_data_size * stage1_perc)
            stage2_data_size = int(train_data_size * stage2_perc)
            stage1_idx, stage2_idx = train_indices[:stage1_data_size], train_indices[-stage2_data_size:]
        else:
            stage1_data_size, stage2_data_size = train_data_size, train_data_size
            stage1_idx, stage2_idx = train_indices, train_indices
                    

        first_stage_train_dataset = ProxyDataset( slice_tuple((A_tensor, Y_tensor, Z_tensor, W_tensor, X_tensor), stage1_idx), 
                                                    dens_ratio = dens_ratio_tensor[stage1_idx],
                                                    dens_ratio_transformer = dens_ratio_transformer,
                                                    transformers = copy.deepcopy(transformers),
                                                    device = "cpu"
                                                )

        second_stage_train_dataset = ProxyDataset(slice_tuple((A_tensor, Y_tensor, Z_tensor, W_tensor, X_tensor), stage2_idx), 
                                                    dens_ratio = dens_ratio_tensor[stage2_idx],
                                                    dens_ratio_transformer = dens_ratio_transformer,
                                                    transformers = copy.deepcopy(transformers),
                                                    device = "cpu"
                                                )

        validation_dataset = ProxyDataset(slice_tuple((A_tensor, Y_tensor, Z_tensor, W_tensor, X_tensor), val_indices), 
                                            dens_ratio = dens_ratio_tensor[val_indices],
                                            dens_ratio_transformer = dens_ratio_transformer,
                                            transformers = copy.deepcopy(transformers),
                                            device = "cpu"
                                                )

        batch_size=2048 if n_plus_m > 25000 else 1024
        first_stage_train_dataloader = DataLoader(first_stage_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        second_stage_train_dataloader = DataLoader(second_stage_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        #### Setup the configurations for training the two networks and the loss functions and regularization parameters. 
        # SETUP CONFIGURATION for Outcome Model
        outcome_cfg = OutcomePCLConfig.high_dim()
        outcome_cfg.device = device
        outcome_cfg.plot_loss = plot_things
        outcome_cfg.n_epochs = 100
        outcome_cfg.reg_second_stage = (50., 500.)
        outcome_cfg.reg_first_stage = (5e-3, 1e-2)
        outcome_cfg.reg_second_stage_first = (5e-3, 1e-2)
        outcome_cfg.reg_annealing_method = "linear"
        outcome_cfg.lr_first_stage_base = 5e-5
        outcome_cfg.lr_second_stage_ax = 1e-4
        outcome_cfg.lr_second_stage_w = 1e-4
        outcome_cfg.second_stage_loss_name = "huber"
        outcome_cfg.second_stage_loss_kwargs = {"delta": 1.0}
        outcome_cfg.second_stage_head_lr = 1e-2 if n_sample <= 10000 else 1e-3
        outcome_cfg.second_stage_head_steps = 5 if n_sample <= 10000 else 10
        outcome_cfg.weight_decay = 1e-5
        outcome_cfg.scheduler_gamma = 0.99
        if n_sample <= 25000:
            outcome_cfg.reg_first_stage = (5e-3, 10)
            outcome_cfg.reg_second_stage_first = (5e-3, 10)
            outcome_cfg.reg_second_stage = (10., 250.) if n_sample <= 10000 else (50., 500.)
            outcome_cfg.second_stage_head_lr = 1e-2
            outcome_cfg.lr_first_stage_base = 1e-4 if n_sample <= 10000 else 5e-5
            outcome_cfg.lr_second_stage_ax = 1e-4 if n_sample <= 10000 else 5e-5
            outcome_cfg.lr_second_stage_w = 1e-4 if n_sample <= 10000 else 5e-5
            outcome_cfg.scheduler_gamma = 0.99
        ## SETUP CONFIGURATION for Treatment Model
        treatment_cfg = TreatmentPCLConfig.high_dim()
        treatment_cfg.device = device
        treatment_cfg.plot_loss = plot_things

        treatment_cfg.lr_first_stage_base = 5e-5
        treatment_cfg.lr_second_stage_ax = 1e-4
        treatment_cfg.lr_second_stage_z = 1e-4
        treatment_cfg.reg_first_stage = (1e-3, 1e-3)
        treatment_cfg.reg_second_stage_first = (1e-3, 1e-3)
        treatment_cfg.reg_second_stage = (50, 500.)
        treatment_cfg.n_epochs = 100
        treatment_cfg.second_stage_loss_name = "huber"
        treatment_cfg.second_stage_loss_kwargs = {"delta": 1.0}
        treatment_cfg.second_stage_head_lr = 1e-2 if n_sample <= 10000 else 1e-3
        treatment_cfg.second_stage_head_steps = 5 if n_sample <= 10000 else 10
        treatment_cfg.weight_decay = 1e-8
        treatment_cfg.negative_penalty = 0.
        treatment_cfg.scheduler_gamma = 0.99
        if n_sample <= 10000:
            treatment_cfg.reg_second_stage = (10, 150.) if n_sample <= 10000 else (50., 500.)
            treatment_cfg.negative_penalty = 10.
            treatment_cfg.lr_first_stage_base = 1e-4 if n_sample <= 10000 else 5e-5
            treatment_cfg.lr_second_stage_ax = 1e-4 if n_sample <= 10000 else 5e-5
            treatment_cfg.lr_second_stage_z = 1e-4 if n_sample <= 10000 else 5e-5
            treatment_cfg.scheduler_gamma = 0.99
        # SETUP CONFIGURATION for Third Stage DR Model/Treatment Model
        INPUT_DIM = 1
        OUTPUT_DIM = 1 
        third_stage_lr = 1e-4 if n_plus_m < 25000 else 5e-5
        # # 1. Define hyperparameters
        train_kwargs = {
            'lr': third_stage_lr,  # Lower LR for more stable convergence
            'weight_decay': 1e-6 if n_plus_m < 25000 else 1e-7,      # Increased to prevent fitting noise
            'n_epochs': 100,           # More epochs since LR is lower
            'loss_fn': nn.MSELoss(), # Robust to bridge outliers
            'gap_penalty_weight': 0.0, # Small penalty to encourage smoothness
            'log_per_epoch': 20,
            'input_dim': INPUT_DIM,
            'output_dim': OUTPUT_DIM,
            'hidden_dims': [128, 128], # Wider but more regularized
            'dropout_rate': 0.1  
        }
        n_ensemble = 5
        # Capture metadata for this specific simulation run
        meta_params = get_meta_params(outcome_cfg, treatment_cfg, train_kwargs)

        # # Training Outcome Bridge Network

        # In[3]:

        (first_stage_featurizer, 
        treatment_featurizer, 
        backdoor_featurizer,
        outcome_proxy_featurizer,) = build_nets_for_outcome_pcl_net_synthetic_high_dim_experiment(outcome_cfg.device)

        outcome_model = OutcomeBridgePCLNET(   first_stage_featurizer,
                                            treatment_featurizer, 
                                            outcome_proxy_featurizer, 
                                            backdoor_featurizer,
                                            None,
                                            None,
                                            device = device
                                        )


        (optimizers_stage1, 
        optimizers_stage2, 
        schedulers_stage1, 
        schedulers_stage2) = build_adam_optimizers_for_outcome_pcl_net( outcome_model, 
                                                                        outcome_cfg.lr_first_stage,
                                                                        outcome_cfg.lr_second_stage_ax,
                                                                        outcome_cfg.lr_second_stage_w,
                                                                        outcome_cfg.weight_decay,
                                                                        gamma = outcome_cfg.scheduler_gamma
                                                                    )

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

        f_struct_pred_om = outcome_model.pred_structural_function(do_A_tensor,
                                                                second_stage_train_dataset.transformers[0],
                                                                second_stage_train_dataset.transformers[1]
                                                                ).detach().cpu().numpy()

        structured_pred_mse_om = (np.mean(((f_struct_pred_om).reshape(-1, 1) - EY_do_A.reshape(-1, 1)) ** 2))

            
        Result_Dict = {
                        "Algorithm" : "OutcomeBridgePCLNET",
                        "Data_Size" : n_plus_m,
                        "Seed" : seed,
                        "Causal_MSE" : structured_pred_mse_om,
                        **meta_params # Merges all hyperparameters into this row
                    }

        df_results = pd.concat([df_results, pd.DataFrame([Result_Dict])], ignore_index=True)

        if plot_things:
            plot_causal_effect_estimation(do_A, EY_do_A, f_struct_pred_om)
            plt.show()
            
            outcome_model.eval()
            Y_pred = outcome_model(first_stage_train_dataset.transformers[0].to(device).transform(A_tensor.to(device)).to(device), 
                                   first_stage_train_dataset.transformers[2].to(device).transform(Z_tensor.to(device)).to(device),
                                  first_stage_train_dataset.transformers[4].to(device).transform(X_tensor.to(device)).to(device),)
            Y_pred = first_stage_train_dataset.transformers[1].to(device).inverse_transform(Y_pred).detach().cpu().numpy()
            Y_gt = Y_tensor.detach().cpu().numpy()
            
            plot_regression_dashboard(Y_gt, Y_pred,)
            plt.show()
        
        # # Training Treatment Bridge Network

        # In[4]:

        (first_stage_featurizer, 
        second_stage_ax_featurizer, 
        treatment_proxy_featurizer) = build_nets_for_treatment_pcl_net_synthetic_high_dim_experiment(treatment_cfg.device)

        treatment_model = TreatmentBridgePCLNET(first_stage_featurizer,
                                                second_stage_ax_featurizer, 
                                                treatment_proxy_featurizer,
                                                dens_ratio_transformer = copy.deepcopy(dens_ratio_transformer),
                                                device = treatment_cfg.device
                                        )


        # SGD optimizers also work fine. Its code is commented out above
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
        third_stage_dataset, third_stage_dataset_val = create_third_stage_dataset_for_treatment_pcl_net_ate(  treatment_model, 
                                                                                                            second_stage_train_dataloader,
                                                                                                            validation_dataloader,
                                                                                                            TorchStandardScaler(),
                                                                                                            TorchStandardScaler(),
                                                                                                            treatment_featurizer = treatment_featurizer,
                                                                                                            dens_ratio_pred_tolerance = 100.5,
                                                                                                            device = device)

        # 2. Train the Ensemble
        # Note: Pass 'ConditionalMeanEstimator' (the class), NOT 'ConditionalMeanEstimator(...)' (the object)
        third_stage_net = train_third_stage_treatment_pcl_net_ensemble(
            model_class=ConditionalMeanMLP,
            dataloader=DataLoader(third_stage_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
            val_dataloader=DataLoader(third_stage_dataset_val, batch_size=batch_size, shuffle=True, num_workers=0),
            n_members=n_ensemble,             # 5-10 models reduces variance significantly
            device = device,
            **train_kwargs
        )

        do_A_transformed = second_stage_train_dataloader.dataset.transformers[0].to(device).transform(torch.tensor(do_A).to(device).to(torch.float32))
        phi_do_A = treatment_featurizer(do_A_transformed)
        phi_do_A_transformed = third_stage_dataset.input_transformer.to(device).transform(phi_do_A)
        f_struct_pred_tm = third_stage_dataset.outcome_transformer.to(device).inverse_transform(third_stage_net(phi_do_A_transformed)).detach().cpu().numpy()

        structured_pred_mse_tm = (np.mean(((f_struct_pred_tm).reshape(-1, 1) - EY_do_A.reshape(-1, 1)) ** 2))
        Result_Dict = {
                        "Algorithm" : "TreatmentBridgePCLNET",
                        "Data_Size" : n_plus_m,
                        "Seed" : seed,
                        "Causal_MSE" : structured_pred_mse_tm,
                        **meta_params # Merges all hyperparameters into this row
                    }

        df_results = pd.concat([df_results, pd.DataFrame([Result_Dict])], ignore_index=True)

        if plot_things:
            plot_causal_effect_estimation(do_A, EY_do_A, f_struct_pred_tm)
            plt.show()
            
            A_normalized = second_stage_train_dataset.A
            W_normalized = second_stage_train_dataset.W
            X_normalized = second_stage_train_dataset.X
            
            nn_dens_ratio_preds = treatment_model.predict_dens_ratio(A_normalized.to(device), W_normalized.to(device), X_normalized.to(device))
            nn_dens_ratio_preds_np = nn_dens_ratio_preds.detach().cpu().numpy().reshape(-1)
            gt_dens_ratio_normalized = second_stage_train_dataset.dens_ratio
            gt_dens_ratio = dens_ratio_transformer.inverse_transform(gt_dens_ratio_normalized)
            gt_dens_ratio_np = gt_dens_ratio.detach().cpu().numpy().reshape(-1)
            
            
            plot_regression_dashboard(gt_dens_ratio_np, 
                                      nn_dens_ratio_preds_np,
                                      log_scale = False)
            plt.show()
        # # Doubly Robust Training (Version 1) - First Generate the Dataset and then Train a NN to Precict $\mathbb{E}[(Y - h(A, X, W)) \varphi(A, X, Z) \mid A = a]$

        # In[5]:


        treatment_featurizer = torch.nn.Identity()

        (third_stage_dataset_dr,
        third_stage_dataset_dr_val) = create_third_stage_dataset_for_dr_proxy_neural_mean_embedding_ate(
                                                                                                        outcome_model,
                                                                                                        treatment_model, 
                                                                                                        second_stage_train_dataloader,
                                                                                                        validation_dataloader,
                                                                                                        outcome_transformer = TorchStandardScaler(),
                                                                                                        input_transformer = TorchStandardScaler(),
                                                                                                        input_type = "features",
                                                                                                        treatment_featurizer = treatment_featurizer,
                                                                                                        dens_ratio_pred_tolerance = 100.5,
                                                                                                        device = device
                                                                                                    )

        # 2. Train the Ensemble
        # Note: Pass 'ConditionalMeanEstimator' (the class), NOT 'ConditionalMeanEstimator(...)' (the object)
        third_stage_net_dr = train_third_stage_treatment_pcl_net_ensemble(
            model_class=ConditionalMeanMLP,
            dataloader=DataLoader(third_stage_dataset_dr, batch_size=batch_size, shuffle=True, num_workers=0),
            val_dataloader=DataLoader(third_stage_dataset_dr_val, batch_size=batch_size, shuffle=True, num_workers=0),
            n_members=n_ensemble,             # 5-10 models reduces variance significantly
            device = device,
            **train_kwargs
        )


        # In[6]:


        do_A_transformed = second_stage_train_dataloader.dataset.transformers[0].to(device).transform(torch.tensor(do_A).to(device).to(torch.float32))
        do_A_transformed = treatment_featurizer(do_A_transformed)
        do_A_transformed = third_stage_dataset_dr.input_transformer.to(device).transform(do_A_transformed)
        f_struct_pred_slack = third_stage_dataset_dr.outcome_transformer.to(device).inverse_transform(third_stage_net_dr(do_A_transformed)).detach().cpu().numpy()

        f_struct_pred_dr = f_struct_pred_om + f_struct_pred_slack

        if plot_things:
            plot_causal_effect_estimation(do_A, EY_do_A, f_struct_pred_dr)
            plt.show()
            
        structured_pred_mse_dr = (np.mean(((f_struct_pred_dr).reshape(-1, 1) - EY_do_A.reshape(-1, 1)) ** 2))

        Result_Dict = {
                        "Algorithm" : "DRPCLNET_Version1",
                        "Data_Size" : n_plus_m,
                        "Seed" : seed,
                        "Causal_MSE" : structured_pred_mse_dr,
                        **meta_params # Merges all hyperparameters into this row
                    }

        df_results = pd.concat([df_results, pd.DataFrame([Result_Dict])], ignore_index=True)

        print("Structured function test set MSE - DR: {}".format(structured_pred_mse_dr))
        print("Structured function test set MSE - Treatment Model: {}".format(structured_pred_mse_tm))
        print("Structured function test set MSE - Outcome Model: {}".format(structured_pred_mse_om))


        # # Doubly Robust Training (Version 2) - First Generate the Dataset and then Train a NN to Precict $\mathbb{E}[h(A, X, W) \varphi(A, X, Z) \mid A = a]$

        # In[7]:


        treatment_featurizer = torch.nn.Identity()
        # treatment_featurizer = outcome_model.treatment_featurizer
        (third_stage_dataset_dr,
        third_stage_dataset_dr_val) = create_third_stage_dataset_for_dr_proxy_neural_mean_embedding_ate_v2(
                                                                                                        outcome_model,
                                                                                                        treatment_model, 
                                                                                                        second_stage_train_dataloader,
                                                                                                        validation_dataloader,
                                                                                                        outcome_transformer = TorchStandardScaler(),
                                                                                                        input_transformer = TorchStandardScaler(),
                                                                                                        input_type = "features",
                                                                                                        treatment_featurizer = treatment_featurizer,
                                                                                                        dens_ratio_pred_tolerance = 100.5,
                                                                                                        device = device
                                                                                                    )

        # 2. Train the Ensemble
        # Note: Pass 'ConditionalMeanEstimator' (the class), NOT 'ConditionalMeanEstimator(...)' (the object)
        third_stage_net_dr = train_third_stage_treatment_pcl_net_ensemble(
            model_class=ConditionalMeanMLP,
            dataloader=DataLoader(third_stage_dataset_dr, batch_size=batch_size, shuffle=True, num_workers=0),
            val_dataloader=DataLoader(third_stage_dataset_dr_val, batch_size=batch_size, shuffle=True, num_workers=0),
            n_members=n_ensemble,             # 5-10 models reduces variance significantly
            device = device,
            **train_kwargs
        )

        # In[8]:

        do_A_transformed = second_stage_train_dataloader.dataset.transformers[0].to(device).transform(torch.tensor(do_A).to(device).to(torch.float32))
        do_A_transformed = treatment_featurizer(do_A_transformed)
        do_A_transformed = third_stage_dataset_dr.input_transformer.to(device).transform(do_A_transformed)
        f_struct_pred_slack = third_stage_dataset_dr.outcome_transformer.to(device).inverse_transform(third_stage_net_dr(do_A_transformed)).detach().cpu().numpy()

        f_struct_pred_dr = f_struct_pred_om + f_struct_pred_tm - f_struct_pred_slack
        
        if plot_things:
            plot_causal_effect_estimation(do_A, EY_do_A, f_struct_pred_dr)
            plt.show()
            
        structured_pred_mse_dr = (np.mean(((f_struct_pred_dr).reshape(-1, 1) - EY_do_A.reshape(-1, 1)) ** 2))

        Result_Dict = {
                        "Algorithm" : "DRPCLNET_Version2",
                        "Data_Size" : n_plus_m,
                        "Seed" : seed,
                        "Causal_MSE" : structured_pred_mse_dr,
                        **meta_params # Merges all hyperparameters into this row
                    }

        df_results = pd.concat([df_results, pd.DataFrame([Result_Dict])], ignore_index=True)

        print("Structured function test set MSE - DR: {}".format(structured_pred_mse_dr))
        print("Structured function test set MSE - Treatment Model: {}".format(structured_pred_mse_tm))
        print("Structured function test set MSE - Outcome Model: {}".format(structured_pred_mse_om))

        del outcome_model, treatment_model, third_stage_net, third_stage_net_dr

        # Generate Data
        df_results.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, required=True)
    args = parser.parse_args()

    seeds = np.arange(0, 3000, 100)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("../Results/Synthetic_High_Dim_Experiment_NewVersion", exist_ok=True)
    out_path = f"../Results/Synthetic_High_Dim_Experiment_NewVersion/DRPCLNET_SyntheticHighDim_NewVersion_huber_{args.size}_{timestamp}.csv"

    run_size_experiment(args.size, seeds, out_path)