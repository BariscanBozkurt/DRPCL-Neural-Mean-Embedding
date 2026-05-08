#!/usr/bin/env python
# coding: utf-8

import os
import gc
import sys
import copy
import random
import argparse
from datetime import datetime

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.append("../../src")

# Utilities
from simulation_utils.data_generating_processes import SyntheticLowDimATTBenchmark
from torch_utils.scalers import TorchStandardScaler, TorchIdentityTransformer
from torch_utils.dataset_classes import ProxyDataset
from torch_utils.densratio import CausalKDEDensRatioTorch, AnchoredATTDensRatio
from torch_utils.networks import ConditionalMeanMLP
from torch_utils.torch_loss import get_loss_function
from python_utils.helpers import slice_tuple

# Treatment Bridge
from simulation_utils.treatment_pcl_net_configs import TreatmentPCLConfig
from simulation_utils.treatment_pcl_net_nn_structures import (
    build_nets_for_treatment_pcl_net_synthetic_low_dim_experiment,
)
from simulation_utils.treatment_pcl_net_optimizers import (
    build_adam_optimizers_for_treatment_pcl_net,
)
from neural_causal_learning.proxy_treatment_neural_mean_embedding import (
    create_third_stage_dataset_for_treatment_pcl_net_ate,
    train_third_stage_treatment_pcl_net_ensemble,
)
from neural_causal_learning.proxy_treatment_neural_mean_embedding_w_SGD import (
    TreatmentBridgePCLNET,
    train_treatment_pcl_net_ate_model,
)

# Outcome Bridge
from simulation_utils.outcome_pcl_net_configs import OutcomePCLConfig
from simulation_utils.outcome_pcl_net_nn_structures import (
    build_nets_for_outcome_pcl_net_synthetic_low_dim_experiment,
)
from simulation_utils.outcome_pcl_net_optimizers import (
    build_adam_optimizers_for_outcome_pcl_net,
)
from neural_causal_learning.proxy_outcome_neural_mean_embedding_w_SGD import (
    OutcomeBridgePCLNET,
    train_deep_feature_proxy_closed_form_ate_model,
)
from neural_causal_learning.proxy_outcome_neural_mean_embedding import (
    create_third_stage_dataset_for_outcome_pcl_net_att,
)

# Doubly robust
from neural_causal_learning.doubly_robust_proxy_neural_mean_embedding import (
    create_third_stage_dataset_for_dr_proxy_neural_mean_embedding_ate,
    create_third_stage_dataset_for_dr_proxy_neural_mean_embedding_ate_v2,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_mse(pred: np.ndarray, truth: np.ndarray) -> float:
    pred = np.asarray(pred).reshape(-1, 1)
    truth = np.asarray(truth).reshape(-1, 1)
    return float(np.mean((pred - truth) ** 2))


def safe_feature_dim(t: torch.Tensor) -> int:
    return 1 if t.ndim == 1 else int(t.shape[1])


def build_third_stage_kwargs(n_sample: int, input_dim: int, output_dim: int):
    lr = 5e-4 if n_sample > 5000 else 1e-3
    lr = lr if n_sample <= 15000 else 1e-4
    return {
        "lr": lr,
        "weight_decay": 1e-6,
        "n_epochs": 100,
        "loss_fn": nn.MSELoss(),
        "gap_penalty_weight": 0.0,
        "log_per_epoch": 10,
        "input_dim": int(input_dim),
        "output_dim": int(output_dim),
        "hidden_dims": [32, 64],
        "dropout_rate": 0.01,
    }


def predict_scalar_stage3_curve(
    do_A_tensor: torch.Tensor,
    second_stage_dataset,
    third_stage_dataset,
    third_stage_net: nn.Module,
    device: torch.device,
):
    """
    Used for TreatmentNet and DR stage-3 regressions:
    learned function is scalar-valued in a only.
    """
    do_A_transformed = second_stage_dataset.transformers[0].to(device).transform(
        do_A_tensor.to(device).to(torch.float32)
    )
    phi_do_A = do_A_transformed  # identity featurizer
    phi_do_A_transformed = third_stage_dataset.input_transformer.to(device).transform(phi_do_A)
    preds = third_stage_dataset.outcome_transformer.to(device).inverse_transform(
        third_stage_net(phi_do_A_transformed)
    )
    return preds.detach().cpu().numpy()


def append_rows(df_results: pd.DataFrame, rows: list, csv_path: str) -> pd.DataFrame:
    if len(rows) == 0:
        return df_results
    df_results = pd.concat([df_results, pd.DataFrame(rows)], ignore_index=True)
    df_results.to_csv(csv_path, index=False)
    return df_results


# ---------------------------------------------------------------------
# Single seed run
# ---------------------------------------------------------------------
def run_single_experiment(
    n_sample: int,
    seed: int,
    args,
    device: torch.device,
):
    seed_everything(seed)

    benchmark = SyntheticLowDimATTBenchmark()
    data = benchmark.generate_dataset(
        n=n_sample,
        seed=seed,
        anchor_a=args.anchor_a,
        do_A_range=(args.do_a_min, args.do_a_max),
        do_A_size=args.do_a_size,
        include_ate_curve=True,
    )

    A = data["A"]
    Y = data["Y"]
    Z = data["Z"]
    W = data["W"]
    A_anchor = data["A_anchor"]
    do_A = data["do_A"]
    EY_do_A = data["EY_att"]   # <-- ground-truth ATT curve

    A_tensor = torch.tensor(A, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)
    Z_tensor = torch.tensor(Z, dtype=torch.float32)
    W_tensor = torch.tensor(W, dtype=torch.float32)
    A_anchor_tensor = torch.tensor(A_anchor, dtype=torch.float32)
    do_A_tensor = torch.tensor(do_A, dtype=torch.float32)
    EY_do_A_tensor = torch.tensor(EY_do_A, dtype=torch.float32)

    # ------------------------------------------------------------------
    # 1) Anchored ATT density-ratio estimation
    # ------------------------------------------------------------------
    transformers = [
        TorchStandardScaler(),  # A
        TorchStandardScaler(),  # Y
        TorchStandardScaler(),  # Z
        TorchStandardScaler(),  # W
    ]
    dens_ratio_transformer = TorchIdentityTransformer()

    A_tensor_transformed = transformers[0].fit_transform(A_tensor).to(device)
    W_tensor_transformed = transformers[3].fit_transform(W_tensor).to(device)
    A_anchor_transformed = transformers[0].transform(A_anchor_tensor).to(device)

    base_ratio_estimator = CausalKDEDensRatioTorch(device=device)
    att_ratio_estimator = AnchoredATTDensRatio(
        base_ratio_estimator,
        eps=1e-6,
        clip_min=1e-4,
        clip_max=100.0,
    )

    att_ratio_estimator.fit(A_tensor_transformed, W_tensor_transformed)
    dens_ratio_att_tensor, _, _ = att_ratio_estimator.predict_ratio(
        A_tensor_transformed,
        W_tensor_transformed,
        A_anchor_transformed,
    )
    dens_ratio_tensor = dens_ratio_att_tensor.detach().cpu()

    inlier_indices = (
        (dens_ratio_tensor > args.inlier_min) &
        (dens_ratio_tensor < args.inlier_max)
    ).view(-1)

    effective_n = int(inlier_indices.sum().item())
    inlier_ratio = effective_n / float(A_tensor.shape[0])

    print(f"[N={n_sample}, seed={seed}] Inlier ratio: {inlier_ratio:.4f}")

    if effective_n < max(args.batch_size // 4, 128):
        print(f"[N={n_sample}, seed={seed}] Too few inliers after filtering. Skipping.")
        return []

    # Filter after density-ratio estimation
    A_tensor = A_tensor[inlier_indices]
    Y_tensor = Y_tensor[inlier_indices]
    Z_tensor = Z_tensor[inlier_indices]
    W_tensor = W_tensor[inlier_indices]
    dens_ratio_tensor = dens_ratio_tensor[inlier_indices]

    # ------------------------------------------------------------------
    # 2) Train / validation split
    # ------------------------------------------------------------------
    data_indices = np.random.permutation(effective_n)
    train_indices = data_indices[: int(effective_n * (1.0 - args.val_perc))]
    val_indices = data_indices[int(effective_n * (1.0 - args.val_perc)) :]
    train_data_size = len(train_indices)

    # Refit transformers only on the training split
    transformers[0].fit(A_tensor[train_indices])
    transformers[1].fit(Y_tensor[train_indices])
    transformers[2].fit(Z_tensor[train_indices])
    transformers[3].fit(W_tensor[train_indices])
    dens_ratio_transformer.fit(dens_ratio_tensor[train_indices])

    if 0.0 < args.stage1_perc < 1.0:
        stage1_data_size = int(train_data_size * args.stage1_perc)
        stage2_data_size = int(train_data_size * args.stage2_perc)
        stage1_idx = train_indices[:stage1_data_size]
        stage2_idx = train_indices[-stage2_data_size:]
    else:
        stage1_idx, stage2_idx = train_indices, train_indices

    first_stage_train_dataset = ProxyDataset(
        slice_tuple((A_tensor, Y_tensor, Z_tensor, W_tensor), stage1_idx),
        dens_ratio=dens_ratio_tensor[stage1_idx],
        dens_ratio_transformer=dens_ratio_transformer,
        transformers=copy.deepcopy(transformers),
        device="cpu",
    )

    second_stage_train_dataset = ProxyDataset(
        slice_tuple((A_tensor, Y_tensor, Z_tensor, W_tensor), stage2_idx),
        dens_ratio=dens_ratio_tensor[stage2_idx],
        dens_ratio_transformer=dens_ratio_transformer,
        transformers=copy.deepcopy(transformers),
        device="cpu",
    )

    validation_dataset = ProxyDataset(
        slice_tuple((A_tensor, Y_tensor, Z_tensor, W_tensor), val_indices),
        dens_ratio=dens_ratio_tensor[val_indices],
        dens_ratio_transformer=dens_ratio_transformer,
        transformers=copy.deepcopy(transformers),
        device="cpu",
    )

    first_stage_train_dataloader = DataLoader(
        first_stage_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    second_stage_train_dataloader = DataLoader(
        second_stage_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # ------------------------------------------------------------------
    # 3) Configurations (same as notebook)
    # ------------------------------------------------------------------
    outcome_cfg = OutcomePCLConfig.low_dim()
    outcome_cfg.device = device
    outcome_cfg.plot_loss = False
    outcome_cfg.n_epochs = 100
    outcome_cfg.lr_first_stage_base = 1e-3
    outcome_cfg.reg_second_stage = (1e-3, 10.0)
    outcome_cfg.second_stage_head_lr = 1.0
    outcome_cfg.second_stage_head_steps = 10
    outcome_cfg.second_stage_loss_name = "log_cosh"
    outcome_cfg.second_stage_loss_kwargs = {}

    treatment_cfg = TreatmentPCLConfig.low_dim()
    treatment_cfg.device = device
    treatment_cfg.plot_loss = False
    treatment_cfg.lr_first_stage_base = 5e-4
    treatment_cfg.reg_first_stage = (1e-5, 1e-3)
    treatment_cfg.reg_second_stage_first = (1e-5, 1e-4)
    treatment_cfg.reg_second_stage = (1e-5, 1e-1)
    if n_sample > 15000:
        treatment_cfg.reg_first_stage = (1e-3, 1e-1)
        treatment_cfg.reg_second_stage_first = (1e-3, 1e-2)
        treatment_cfg.reg_second_stage = (1e-3, 10.0)
    treatment_cfg.n_epochs = 100
    treatment_cfg.second_stage_loss_name = "log_cosh"
    treatment_cfg.second_stage_loss_kwargs = {}
    treatment_cfg.second_stage_head_lr = 1.0
    treatment_cfg.second_stage_head_steps = 15
    treatment_cfg.negative_penalty = 10.0

    common_row = {
        "Experiment": "SyntheticLowDim_ATT",
        "Data_Size": int(n_sample),
        "Effective_Data_Size": int(effective_n),
        "Seed": int(seed),
        "Anchor_A": float(args.anchor_a),
        "Inlier_Ratio": float(inlier_ratio),
    }

    results = []

    # ------------------------------------------------------------------
    # 4) Outcome bridge
    # ------------------------------------------------------------------
    (
        out_first_stage_featurizer,
        out_treatment_featurizer,
        out_outcome_proxy_featurizer,
    ) = build_nets_for_outcome_pcl_net_synthetic_low_dim_experiment(outcome_cfg.device)

    outcome_model = OutcomeBridgePCLNET(
        out_first_stage_featurizer,
        out_treatment_featurizer,
        out_outcome_proxy_featurizer,
        None,
        None,
        device=outcome_cfg.device,
    )

    (
        optimizers_stage1,
        optimizers_stage2,
        schedulers_stage1,
        schedulers_stage2,
    ) = build_adam_optimizers_for_outcome_pcl_net(
        outcome_model,
        outcome_cfg.lr_first_stage,
        outcome_cfg.lr_second_stage_ax,
        outcome_cfg.lr_second_stage_w,
        outcome_cfg.weight_decay,
        gamma=outcome_cfg.scheduler_gamma,
    )

    reg1, reg2_inner, reg2_outer = outcome_cfg.regularizers
    loss_fn_1 = get_loss_function(outcome_cfg.first_stage_loss_name, outcome_cfg.first_stage_loss_kwargs)
    loss_fn_2 = get_loss_function(outcome_cfg.second_stage_loss_name, outcome_cfg.second_stage_loss_kwargs)

    outcome_model = train_deep_feature_proxy_closed_form_ate_model(
        pcl_model=outcome_model,
        first_stage_train_dataloader=first_stage_train_dataloader,
        second_stage_train_dataloader=second_stage_train_dataloader,
        stage1_optimizers=optimizers_stage1,
        stage2_optimizers=optimizers_stage2,
        stage1_schedulers=schedulers_stage1,
        stage2_schedulers=schedulers_stage2,
        n_epochs=outcome_cfg.n_epochs,
        stage1_iter=outcome_cfg.stage1_iter,
        stage2_iter=outcome_cfg.stage2_iter,
        first_stage_final_layer_regularizer=reg1,
        second_stage_final_layer_regularizer=reg2_outer,
        second_stage_first_final_layer_regularizer=reg2_inner,
        regularizer_annealing_method=outcome_cfg.reg_annealing_method,
        consider_prev_weight=outcome_cfg.consider_prev_weight,
        first_stage_loss_fn=loss_fn_1,
        second_stage_loss_fn=loss_fn_2,
        second_stage_head_lr=outcome_cfg.second_stage_head_lr,
        second_stage_head_steps=outcome_cfg.second_stage_head_steps,
        log_per_epoch=outcome_cfg.log_per_epoch,
        plot_loss=outcome_cfg.plot_loss,
        validation_dataloader=validation_dataloader,
    )

    third_stage_dataset_outcome, third_stage_dataset_val_outcome = create_third_stage_dataset_for_outcome_pcl_net_att(
        outcome_model,
        second_stage_train_dataloader,
        validation_dataloader,
        device=device,
    )

    train_kwargs_outcome = build_third_stage_kwargs(
        n_sample=n_sample,
        input_dim=safe_feature_dim(third_stage_dataset_outcome.tensors[0]),
        output_dim=safe_feature_dim(third_stage_dataset_outcome.tensors[1]),
    )

    third_stage_net_outcome = train_third_stage_treatment_pcl_net_ensemble(
        model_class=ConditionalMeanMLP,
        dataloader=DataLoader(
            third_stage_dataset_outcome,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        ),
        val_dataloader=DataLoader(
            third_stage_dataset_val_outcome,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        ),
        n_members=args.n_ensemble,
        device=device,
        **train_kwargs_outcome,
    )

    f_struct_pred_om = outcome_model.pred_conditional_structural_function(
        do_A_tensor,
        A_anchor_tensor,
        third_stage_net_outcome,
        second_stage_train_dataset.transformers[0],
        second_stage_train_dataset.transformers[1],
    ).detach().cpu().numpy()

    structured_pred_mse_om = compute_mse(f_struct_pred_om, EY_do_A)
    print(f"[N={n_sample}, seed={seed}] OutcomeBridge ATT MSE: {structured_pred_mse_om:.6f}")

    results.append({
        **common_row,
        "Algorithm": "OutcomeBridgePCLNET",
        "Causal_MSE": structured_pred_mse_om,
    })

    # ------------------------------------------------------------------
    # 5) Treatment bridge
    # ------------------------------------------------------------------
    (
        treat_first_stage_featurizer,
        second_stage_ax_featurizer,
        treatment_proxy_featurizer,
    ) = build_nets_for_treatment_pcl_net_synthetic_low_dim_experiment(treatment_cfg.device)

    treatment_model = TreatmentBridgePCLNET(
        treat_first_stage_featurizer,
        second_stage_ax_featurizer,
        treatment_proxy_featurizer,
        dens_ratio_transformer=copy.deepcopy(dens_ratio_transformer),
        device=treatment_cfg.device,
    )

    (
        stage1_optimizers,
        stage2_optimizers,
        stage1_schedulers,
        stage2_schedulers,
    ) = build_adam_optimizers_for_treatment_pcl_net(
        treatment_model,
        lr_first_stage=treatment_cfg.lr_first_stage,
        lr_second_stage_ax=treatment_cfg.lr_second_stage_ax,
        lr_second_stage_z=treatment_cfg.lr_second_stage_z,
        weight_decay=treatment_cfg.weight_decay,
        gamma=treatment_cfg.scheduler_gamma,
    )

    reg1, reg2_inner, reg2_outer = treatment_cfg.regularizers
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
        first_stage_final_layer_regularizer=reg1,
        second_stage_first_final_layer_regularizer=reg2_inner,
        second_stage_final_layer_regularizer=reg2_outer,
        regularizer_annealing_method=treatment_cfg.reg_annealing_method,
        consider_prev_weight=treatment_cfg.consider_prev_weight,
        first_stage_loss_fn=loss_fn_1,
        second_stage_loss_fn=loss_fn_2,
        second_stage_head_lr=treatment_cfg.second_stage_head_lr,
        second_stage_head_steps=treatment_cfg.second_stage_head_steps,
        negative_penalty=treatment_cfg.negative_penalty,
        log_per_epoch=treatment_cfg.log_per_epoch,
        plot_loss=treatment_cfg.plot_loss,
        validation_dataloader=validation_dataloader,
    )

    identity_treatment_featurizer = torch.nn.Identity()
    third_stage_dataset_treat, third_stage_dataset_val_treat = create_third_stage_dataset_for_treatment_pcl_net_ate(
        treatment_model,
        second_stage_train_dataloader,
        validation_dataloader,
        outcome_transformer=TorchStandardScaler(),
        input_transformer=TorchStandardScaler(),
        treatment_featurizer=identity_treatment_featurizer,
        dens_ratio_pred_tolerance=args.dens_ratio_pred_tolerance,
        device=device,
    )

    train_kwargs_scalar = build_third_stage_kwargs(
        n_sample=n_sample,
        input_dim=safe_feature_dim(third_stage_dataset_treat.tensors[0]),
        output_dim=safe_feature_dim(third_stage_dataset_treat.tensors[1]),
    )

    third_stage_net_treat = train_third_stage_treatment_pcl_net_ensemble(
        model_class=ConditionalMeanMLP,
        dataloader=DataLoader(
            third_stage_dataset_treat,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        ),
        val_dataloader=DataLoader(
            third_stage_dataset_val_treat,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        ),
        n_members=args.n_ensemble,
        device=device,
        **train_kwargs_scalar,
    )

    f_struct_pred_tm = predict_scalar_stage3_curve(
        do_A_tensor,
        second_stage_train_dataset,
        third_stage_dataset_treat,
        third_stage_net_treat,
        device,
    )

    structured_pred_mse_tm = compute_mse(f_struct_pred_tm, EY_do_A)
    print(f"[N={n_sample}, seed={seed}] TreatmentBridge ATT MSE: {structured_pred_mse_tm:.6f}")

    results.append({
        **common_row,
        "Algorithm": "TreatmentBridgePCLNET",
        "Causal_MSE": structured_pred_mse_tm,
    })

    # ------------------------------------------------------------------
    # 6) Doubly robust version 1
    # ------------------------------------------------------------------
    third_stage_dataset_dr1, third_stage_dataset_val_dr1 = create_third_stage_dataset_for_dr_proxy_neural_mean_embedding_ate(
        outcome_model,
        treatment_model,
        second_stage_train_dataloader,
        validation_dataloader,
        outcome_transformer=TorchStandardScaler(),
        input_transformer=TorchStandardScaler(),
        input_type="features",
        treatment_featurizer=identity_treatment_featurizer,
        dens_ratio_pred_tolerance=args.dens_ratio_pred_tolerance,
        device=device,
    )

    train_kwargs_dr1 = build_third_stage_kwargs(
        n_sample=n_sample,
        input_dim=safe_feature_dim(third_stage_dataset_dr1.tensors[0]),
        output_dim=safe_feature_dim(third_stage_dataset_dr1.tensors[1]),
    )

    third_stage_net_dr1 = train_third_stage_treatment_pcl_net_ensemble(
        model_class=ConditionalMeanMLP,
        dataloader=DataLoader(
            third_stage_dataset_dr1,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        ),
        val_dataloader=DataLoader(
            third_stage_dataset_val_dr1,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        ),
        n_members=args.n_ensemble,
        device=device,
        **train_kwargs_dr1,
    )

    f_struct_pred_slack_v1 = predict_scalar_stage3_curve(
        do_A_tensor,
        second_stage_train_dataset,
        third_stage_dataset_dr1,
        third_stage_net_dr1,
        device,
    )
    f_struct_pred_dr_v1 = f_struct_pred_om + f_struct_pred_slack_v1
    structured_pred_mse_dr_v1 = compute_mse(f_struct_pred_dr_v1, EY_do_A)

    print(f"[N={n_sample}, seed={seed}] DRPCLNET_V1 ATT MSE: {structured_pred_mse_dr_v1:.6f}")

    results.append({
        **common_row,
        "Algorithm": "DRPCLNET_Version1",
        "Causal_MSE": structured_pred_mse_dr_v1,
    })

    # ------------------------------------------------------------------
    # 7) Doubly robust version 2
    # ------------------------------------------------------------------
    third_stage_dataset_dr2, third_stage_dataset_val_dr2 = create_third_stage_dataset_for_dr_proxy_neural_mean_embedding_ate_v2(
        outcome_model,
        treatment_model,
        second_stage_train_dataloader,
        validation_dataloader,
        outcome_transformer=TorchStandardScaler(),
        input_transformer=TorchStandardScaler(),
        input_type="features",
        treatment_featurizer=identity_treatment_featurizer,
        dens_ratio_pred_tolerance=args.dens_ratio_pred_tolerance,
        device=device,
    )

    train_kwargs_dr2 = build_third_stage_kwargs(
        n_sample=n_sample,
        input_dim=safe_feature_dim(third_stage_dataset_dr2.tensors[0]),
        output_dim=safe_feature_dim(third_stage_dataset_dr2.tensors[1]),
    )

    third_stage_net_dr2 = train_third_stage_treatment_pcl_net_ensemble(
        model_class=ConditionalMeanMLP,
        dataloader=DataLoader(
            third_stage_dataset_dr2,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        ),
        val_dataloader=DataLoader(
            third_stage_dataset_val_dr2,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        ),
        n_members=args.n_ensemble,
        device=device,
        **train_kwargs_dr2,
    )

    f_struct_pred_slack_v2 = predict_scalar_stage3_curve(
        do_A_tensor,
        second_stage_train_dataset,
        third_stage_dataset_dr2,
        third_stage_net_dr2,
        device,
    )
    f_struct_pred_dr_v2 = f_struct_pred_om + f_struct_pred_tm - f_struct_pred_slack_v2
    structured_pred_mse_dr_v2 = compute_mse(f_struct_pred_dr_v2, EY_do_A)

    print(f"[N={n_sample}, seed={seed}] DRPCLNET_V2 ATT MSE: {structured_pred_mse_dr_v2:.6f}")

    results.append({
        **common_row,
        "Algorithm": "DRPCLNET_Version2",
        "Causal_MSE": structured_pred_mse_dr_v2,
    })

    return results


# ---------------------------------------------------------------------
# Loop over seeds for one N
# ---------------------------------------------------------------------
def run_size_experiment(n_sample: int, seed_list, output_file: str, args):
    torch.set_num_threads(max(1, int(args.torch_num_threads)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_results = pd.DataFrame(
        columns=[
            "Experiment",
            "Algorithm",
            "Data_Size",
            "Effective_Data_Size",
            "Seed",
            "Anchor_A",
            "Inlier_Ratio",
            "Causal_MSE",
        ]
    )

    for seed in seed_list:
        try:
            rows = run_single_experiment(
                n_sample=n_sample,
                seed=int(seed),
                args=args,
                device=device,
            )
            df_results = append_rows(df_results, rows, output_file)
        except Exception as e:
            print(f"[ERROR] N={n_sample}, seed={seed}: {repr(e)}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df_results.to_csv(output_file, index=False)
    print(f"Finished N={n_sample}. Saved to: {output_file}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--output_dir", type=str, default="../Results/Synthetic_Low_Dim_ATT_Experiment")
    parser.add_argument("--run_tag", type=str, default=None)

    parser.add_argument("--anchor_a", type=float, default=-1.0)
    parser.add_argument("--do_a_min", type=float, default=-1.0)
    parser.add_argument("--do_a_max", type=float, default=2.0)
    parser.add_argument("--do_a_size", type=int, default=100)

    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--seed_stop", type=int, default=3000)
    parser.add_argument("--seed_step", type=int, default=100)

    parser.add_argument("--inlier_min", type=float, default=1e-3)
    parser.add_argument("--inlier_max", type=float, default=100.0)
    parser.add_argument("--dens_ratio_pred_tolerance", type=float, default=100.5)

    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--val_perc", type=float, default=0.1)
    parser.add_argument("--stage1_perc", type=float, default=0.75)
    parser.add_argument("--stage2_perc", type=float, default=0.75)

    parser.add_argument("--n_ensemble", type=int, default=5)
    parser.add_argument("--torch_num_threads", type=int, default=1)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    seed_list = np.arange(args.seed_start, args.seed_stop, args.seed_step)

    if args.run_tag is None:
        run_tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        run_tag = args.run_tag

    anchor_tag = str(args.anchor_a).replace("-", "m").replace(".", "p")

    output_file = os.path.join(
        args.output_dir,
        f"DRPCLNET_SyntheticLowDim_ATT_anchor_{anchor_tag}_N{args.size}_{run_tag}.csv",
    )

    print(f"Running only size N={args.size}")
    print(f"Anchor a' = {args.anchor_a}")
    print(f"Output file: {output_file}")

    run_size_experiment(args.size, seed_list, output_file, args)