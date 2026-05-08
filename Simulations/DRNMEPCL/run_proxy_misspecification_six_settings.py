#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import copy
import argparse
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.append("../../src")

from torch_utils.scalers import TorchStandardScaler, TorchIdentityTransformer
from torch_utils.dataset_classes import ProxyDataset
from torch_utils.densratio import CausalKDEDensRatioTorch
from torch_utils.networks import ConditionalMeanMLP
from torch_utils.torch_loss import get_loss_function
from torch_utils.kernel_utils import RBF
from python_utils.helpers import slice_tuple

from simulation_utils.treatment_pcl_net_configs import TreatmentPCLConfig
from simulation_utils.treatment_pcl_net_optimizers import build_adam_optimizers_for_treatment_pcl_net
from neural_causal_learning.proxy_treatment_neural_mean_embedding import (
    create_third_stage_dataset_for_treatment_pcl_net_ate,
    train_third_stage_treatment_pcl_net_ensemble,
)
from neural_causal_learning.proxy_treatment_neural_mean_embedding_w_SGD import (
    TreatmentBridgePCLNET,
    train_treatment_pcl_net_ate_model,
)

from simulation_utils.outcome_pcl_net_configs import OutcomePCLConfig
from simulation_utils.outcome_pcl_net_optimizers import build_adam_optimizers_for_outcome_pcl_net
from neural_causal_learning.proxy_outcome_neural_mean_embedding_w_SGD import (
    OutcomeBridgePCLNET,
    train_deep_feature_proxy_closed_form_ate_model,
)

from neural_causal_learning.doubly_robust_proxy_neural_mean_embedding import (
    create_third_stage_dataset_for_dr_proxy_neural_mean_embedding_ate,
    create_third_stage_dataset_for_dr_proxy_neural_mean_embedding_ate_v2,
)

from causal_learning.doubly_robust_kernel_proxy_causal_learning import DoublyRobustKernelProxyATE


# ---------------------------------------------------------------------
# Data-generating process
# ---------------------------------------------------------------------

def Lambda(x: np.ndarray) -> np.ndarray:
    return 0.8 / (1.0 + np.exp(-x)) + 0.1


def _beta_params_for_setting(setting: int) -> Tuple[float, float]:
    if setting in {1, 2}:
        return 5, 4
    if setting in {3, 4}:
        return 8, 4
    if setting in {5, 6}:
        return 3, 5
    raise ValueError("setting must be one of {1, 2, 3, 4, 5, 6}.")


def monte_carlo_dose_response(
    do_A: np.ndarray,
    setting: int,
    mc_size: int = 100_000,
    seed: Optional[int] = None,
    normal_std: float = 0.1,
    setting6_w_noise_range: Tuple[float, float] = (-100.0, 100.0),
) -> np.ndarray:
    do_A = np.asarray(do_A, dtype=float).reshape(-1)

    rng = np.random.default_rng(seed)
    beta_a, beta_b = _beta_params_for_setting(setting)
    U_mc = rng.beta(beta_a, beta_b, size=mc_size)

    if setting in {1, 2, 3, 4}:
        return np.mean(2.0 * U_mc - 1.0) + np.cos(1.5 * do_A)

    if setting == 5:
        W_mc = -(U_mc ** 2) + rng.uniform(0.0, 1.0, size=mc_size)
        return np.asarray([
            np.mean(3.0 * W_mc - 0.1 * a - np.cos(0.5 * a + 5.0 * U_mc))
            for a in do_A
        ])

    if setting == 6:
        W1_mc = rng.normal(loc=-1.0, scale=normal_std, size=mc_size)
        W2_mc = rng.normal(loc=1.0, scale=normal_std, size=mc_size)
        W_noise_mc = rng.uniform(
            setting6_w_noise_range[0],
            setting6_w_noise_range[1],
            size=mc_size,
        )
        W_mc = Lambda((1.0 - U_mc) * W1_mc + U_mc * W2_mc + W_noise_mc)
        return np.asarray([
            np.mean(3.0 * W_mc - 2.0 * a - np.cos(10.0 * a + 5.0 * U_mc))
            for a in do_A
        ])

    raise ValueError("setting must be one of {1, 2, 3, 4, 5, 6}.")


def exact_dose_response(do_A: np.ndarray, setting: int) -> np.ndarray:
    do_A = np.asarray(do_A, dtype=float).reshape(-1)
    beta_a, beta_b = _beta_params_for_setting(setting)

    if setting in {1, 2, 3, 4}:
        return 2.0 * beta_a / (beta_a + beta_b) - 1.0 + np.cos(1.5 * do_A)

    return np.full_like(do_A, np.nan, dtype=float)


def generate_proxy_misspecification_data(
    setting: int,
    seed: int,
    data_size: int = 1000,
    do_A_size: int = 100,
    do_A_range: Optional[Tuple[float, float]] = None,
    mc_size: int = 100_000,
    normal_std: float = 0.1,
    return_2d: bool = False,
    setting6_w_noise_range: Tuple[float, float] = (-100.0, 100.0),
) -> Dict[str, Any]:
    if setting not in {1, 2, 3, 4, 5, 6}:
        raise ValueError("setting must be one of {1, 2, 3, 4, 5, 6}.")

    rng = np.random.default_rng(seed)
    beta_a, beta_b = _beta_params_for_setting(setting)

    U = rng.beta(beta_a, beta_b, size=data_size)

    if setting == 1:
        W = Lambda(U) + rng.uniform(0.0, 1.0, size=data_size)

        Z1 = rng.normal(loc=-1.0, scale=normal_std, size=data_size)
        Z2 = rng.normal(loc=1.0, scale=normal_std, size=data_size)
        Z = (1.0 - U) * Z1 + U * Z2 + rng.uniform(0.0, 100.0, size=data_size)

        A = 0.1 * U + 0.1 * Z + rng.uniform(0.0, 1.0, size=data_size)
        Y = (2.0 * U - 1.0) + np.cos(1.5 * A)

    elif setting == 2:
        Z = Lambda(U) + rng.uniform(0.0, 1.0, size=data_size)

        W1 = rng.normal(loc=-1.0, scale=normal_std, size=data_size)
        W2 = rng.normal(loc=1.0, scale=normal_std, size=data_size)
        W = (1.0 - U) * W1 + U * W2 + rng.uniform(0.0, 100.0, size=data_size)

        A = 0.1 * U + 0.1 * Z + rng.uniform(0.0, 1.0, size=data_size)
        Y = (2.0 * U - 1.0) + np.cos(1.5 * A)

    elif setting == 3:
        W = U + rng.uniform(0.0, 1.0, size=data_size)

        Z1 = rng.normal(loc=-1.0, scale=normal_std, size=data_size)
        Z2 = rng.normal(loc=1.0, scale=normal_std, size=data_size)
        Z = Lambda((1.0 - U) * Z1 + U * Z2) + rng.uniform(0.0, 100.0, size=data_size)

        A = 0.1 * U + 0.1 * Z + rng.uniform(0.0, 1.0, size=data_size)
        Y = (2.0 * U - 1.0) + np.cos(1.5 * A)

    elif setting == 4:
        Z = U + rng.uniform(0.0, 1.0, size=data_size)

        W1 = rng.normal(loc=-1.0, scale=normal_std, size=data_size)
        W2 = rng.normal(loc=1.0, scale=normal_std, size=data_size)
        W = Lambda((1.0 - U) * W1 + U * W2) + rng.uniform(0.0, 100.0, size=data_size)

        A = 0.1 * U + 0.1 * Z + rng.uniform(0.0, 1.0, size=data_size)
        Y = (2.0 * U - 1.0) + np.cos(1.5 * A)

    elif setting == 5:
        W = -(U ** 2) + rng.uniform(0.0, 1.0, size=data_size)

        Z1 = rng.normal(loc=-1.0, scale=normal_std, size=data_size)
        Z2 = rng.normal(loc=1.0, scale=normal_std, size=data_size)
        Z = Lambda((1.0 - U) * Z1 + U * Z2) + rng.uniform(0.0, 100.0, size=data_size)

        A = 0.25 * np.sqrt(np.abs(U)) - 0.2 * Z + rng.uniform(0.0, 1.0, size=data_size)
        Y = 3.0 * W - 0.1 * A - np.cos(0.5 * A + 5.0 * U)

    else:
        Z = -(U ** 2) + rng.uniform(0.0, 1.0, size=data_size)

        W1 = rng.normal(loc=-1.0, scale=normal_std, size=data_size)
        W2 = rng.normal(loc=1.0, scale=normal_std, size=data_size)
        W_noise = rng.uniform(
            setting6_w_noise_range[0],
            setting6_w_noise_range[1],
            size=data_size,
        )
        W = Lambda((1.0 - U) * W1 + U * W2 + W_noise)

        A = 0.25 * np.sqrt(np.abs(U)) - 0.2 * Z + rng.uniform(0.0, 1.0, size=data_size)
        Y = 3.0 * W - 2.0 * A - np.cos(10.0 * A + 5.0 * U)

    if do_A_range is None:
        do_A_range = (float(A.min()), float(A.max()))

    do_A = np.linspace(do_A_range[0], do_A_range[1], do_A_size)

    EY_do_A = monte_carlo_dose_response(
        do_A=do_A,
        setting=setting,
        mc_size=mc_size,
        seed=seed + 10_000,
        normal_std=normal_std,
        setting6_w_noise_range=setting6_w_noise_range,
    )

    EY_do_A_exact = exact_dose_response(do_A=do_A, setting=setting)

    if return_2d:
        U = U.reshape(-1, 1)
        A = A.reshape(-1, 1)
        Z = Z.reshape(-1, 1)
        W = W.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        do_A = do_A.reshape(-1, 1)
        EY_do_A = EY_do_A.reshape(-1, 1)
        EY_do_A_exact = EY_do_A_exact.reshape(-1, 1)

    return {
        "setting": setting,
        "U": U,
        "A": A,
        "Z": Z,
        "W": W,
        "Y": Y,
        "do_A": do_A,
        "EY_do_A": EY_do_A,
        "EY_do_A_exact": EY_do_A_exact,
        "beta_a": beta_a,
        "beta_b": beta_b,
    }


def generate_data(
    seed_: int,
    setting: int = 1,
    data_size: int = 1000,
    do_A_size: int = 100,
    mc_size: int = 100_000,
):
    data = generate_proxy_misspecification_data(
        setting=setting,
        seed=seed_,
        data_size=data_size,
        do_A_size=do_A_size,
        mc_size=mc_size,
        return_2d=True,
    )
    return (
        data["U"],
        data["A"],
        data["Z"],
        data["W"],
        data["Y"],
        data["do_A"],
        data["EY_do_A"],
    )


# ---------------------------------------------------------------------
# Neural network builders
# ---------------------------------------------------------------------

class SimpleFeaturizer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_width: int = 64,
        dropout_rate: float = 0.0,
        final_activation: str = "tanh",
    ):
        super().__init__()
        self.output_dim = output_dim
        self.final_activation = final_activation

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_width),
            nn.LayerNorm(hidden_width),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_width, hidden_width),
            nn.LayerNorm(hidden_width),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_width, output_dim),
            nn.LayerNorm(output_dim),
        )

        self.tanh = nn.Tanh()
        self.gelu = nn.GELU()

    def _small_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)

        if self.final_activation == "tanh":
            return self.tanh(out)
        if self.final_activation == "gelu":
            return self.gelu(out)
        if self.final_activation == "linear":
            return out

        raise ValueError(f"Unknown final_activation={self.final_activation}")


def build_nets_for_outcome_pcl_net_proxy_misspecification_experiment(
    device: str = "cuda",
    a_dim: int = 1,
    x_dim: int = 0,
    z_dim: int = 1,
    w_dim: int = 1,
    first_stage_output_dim: int = 32,
    w_output_dim: int = 8,
    ax_output_dim: int = 8,
    hidden_width: int = 64,
    dropout_rate: float = 0.0,
    final_activation: str = "gelu",
):
    first_stage_featurizer = SimpleFeaturizer(
        input_dim=a_dim + x_dim + z_dim,
        output_dim=first_stage_output_dim,
        hidden_width=hidden_width,
        dropout_rate=dropout_rate,
        final_activation=final_activation,
    ).to(device)

    treatment_featurizer = SimpleFeaturizer(
        input_dim=a_dim + x_dim,
        output_dim=ax_output_dim,
        hidden_width=hidden_width,
        dropout_rate=dropout_rate,
        final_activation=final_activation,
    ).to(device)

    outcome_proxy_featurizer = SimpleFeaturizer(
        input_dim=w_dim,
        output_dim=w_output_dim,
        hidden_width=hidden_width,
        dropout_rate=dropout_rate,
        final_activation=final_activation,
    ).to(device)

    return first_stage_featurizer, treatment_featurizer, outcome_proxy_featurizer


def build_nets_for_treatment_pcl_net_proxy_misspecification_experiment(
    device: str = "cuda",
    a_dim: int = 1,
    x_dim: int = 0,
    w_dim: int = 1,
    z_dim: int = 1,
    first_stage_output_dim: int = 128,
    z_output_dim: int = 8,
    ax_output_dim: int = 32,
    hidden_width: int = 64,
    dropout_rate: float = 0.0,
    final_activation: str = "gelu",
):
    first_stage_featurizer = SimpleFeaturizer(
        input_dim=a_dim + x_dim + w_dim,
        output_dim=first_stage_output_dim,
        hidden_width=hidden_width,
        dropout_rate=dropout_rate,
        final_activation=final_activation,
    ).to(device)

    second_stage_ax_featurizer = SimpleFeaturizer(
        input_dim=a_dim + x_dim,
        output_dim=ax_output_dim,
        hidden_width=hidden_width,
        dropout_rate=dropout_rate,
        final_activation=final_activation,
    ).to(device)

    treatment_proxy_featurizer = SimpleFeaturizer(
        input_dim=z_dim,
        output_dim=z_output_dim,
        hidden_width=hidden_width,
        dropout_rate=dropout_rate,
        final_activation=final_activation,
    ).to(device)

    treatment_proxy_featurizer._small_init()

    return first_stage_featurizer, second_stage_ax_featurizer, treatment_proxy_featurizer


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def serialize_array_for_csv(x) -> str:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return json.dumps(np.asarray(x, dtype=float).reshape(-1).tolist())


def mse(pred, truth) -> float:
    pred = np.asarray(pred).reshape(-1, 1)
    truth = np.asarray(truth).reshape(-1, 1)
    return float(np.mean((pred - truth) ** 2))


def parse_seeds(seed_string: str) -> List[int]:
    if seed_string.strip().lower() == "default":
        return list(range(0, 1000, 100))
    return [int(s.strip()) for s in seed_string.split(",") if s.strip()]


def add_result(
    rows,
    output_csv,
    algorithm,
    setting,
    seed,
    data_size,
    pred,
    do_A,
    EY_do_A,
    extra=None,
    error=None,
):
    if pred is None:
        causal_mse = np.nan
        pred_serialized = "[]"
    else:
        causal_mse = mse(pred, EY_do_A)
        pred_serialized = serialize_array_for_csv(pred)

    row = {
        "Algorithm": algorithm,
        "Setting": setting,
        "Scenario": f"setting_{setting}",
        "Data_Size": data_size,
        "Seed": seed,
        "Causal_MSE": causal_mse,
        "Estimated_Response": pred_serialized,
        "Treatment_Grid": serialize_array_for_csv(do_A),
        "Ground_Truth_Response": serialize_array_for_csv(EY_do_A),
        "Error": "" if error is None else str(error),
    }

    if extra is not None:
        row.update(extra)

    rows.append(row)
    pd.DataFrame(rows).to_csv(output_csv, index=False)


# ---------------------------------------------------------------------
# Main simulation for one setting and one seed
# ---------------------------------------------------------------------

def run_one_seed(setting, seed, data_size, output_csv, rows, device):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    U, A, Z, W, Y, do_A, EY_do_A = generate_data(
        seed_=seed,
        setting=setting,
        data_size=data_size,
        do_A_size=100,
        mc_size=100_000,
    )

    Z_np, W_np, A_np, Y_np = Z.copy(), W.copy(), A.copy(), Y.copy()

    Z_tensor = torch.tensor(Z_np, dtype=torch.float32)
    W_tensor = torch.tensor(W_np, dtype=torch.float32)
    A_tensor = torch.tensor(A_np, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_np, dtype=torch.float32)

    do_A_tensor = torch.tensor(do_A, dtype=torch.float32)
    EY_do_A_tensor = torch.tensor(EY_do_A, dtype=torch.float32)

    transformers = [
        TorchStandardScaler(),
        TorchStandardScaler(),
        TorchStandardScaler(),
        TorchStandardScaler(),
    ]
    dens_ratio_transformer = TorchIdentityTransformer()

    A_tensor_transformed = transformers[0].fit_transform(A_tensor).to(device)
    W_tensor_transformed = transformers[3].fit_transform(W_tensor).to(device)

    ratio_estimator = CausalKDEDensRatioTorch(device=device)
    ratio_estimator.fit(A_tensor_transformed, W_tensor_transformed)
    dens_ratio_tensor = ratio_estimator.predict_ratio(
        A_tensor_transformed,
        W_tensor_transformed,
    ).to("cpu")

    inlier_dens_ratio_range = (1e-3, 100.0)
    inlier_indices = (
        (dens_ratio_tensor > inlier_dens_ratio_range[0])
        & (dens_ratio_tensor < inlier_dens_ratio_range[1])
    ).view(-1)

    inlier_ratio = float(inlier_indices.sum().item() / A_tensor_transformed.shape[0])

    A_tensor = A_tensor[inlier_indices]
    Y_tensor = Y_tensor[inlier_indices]
    Z_tensor = Z_tensor[inlier_indices]
    W_tensor = W_tensor[inlier_indices]
    dens_ratio_tensor = dens_ratio_tensor[inlier_indices]

    n_sample = inlier_indices.sum().item()

    val_perc = 0.1
    stage1_perc = 0.75
    stage2_perc = 0.75

    data_indices = np.random.permutation(n_sample)
    train_indices = data_indices[: int(n_sample * (1 - val_perc))]
    val_indices = data_indices[int(n_sample * (1 - val_perc)) :]
    train_data_size = train_indices.shape[0]

    transformers[0].fit(A_tensor[train_indices])
    transformers[1].fit(Y_tensor[train_indices])
    transformers[2].fit(Z_tensor[train_indices])
    transformers[3].fit(W_tensor[train_indices])
    dens_ratio_transformer.fit(dens_ratio_tensor[train_indices])

    stage1_data_size = int(train_data_size * stage1_perc)
    stage2_data_size = int(train_data_size * stage2_perc)
    stage1_idx = train_indices[:stage1_data_size]
    stage2_idx = train_indices[-stage2_data_size:]

    first_stage_train_dataset = ProxyDataset(
        slice_tuple((A_tensor, Y_tensor, Z_tensor, W_tensor), stage1_idx),
        dens_ratio=dens_ratio_tensor[stage1_idx],
        dens_ratio_transformer=dens_ratio_transformer,
        transformers=transformers,
        device="cpu",
    )

    second_stage_train_dataset = ProxyDataset(
        slice_tuple((A_tensor, Y_tensor, Z_tensor, W_tensor), stage2_idx),
        dens_ratio=dens_ratio_tensor[stage2_idx],
        dens_ratio_transformer=dens_ratio_transformer,
        transformers=transformers,
        device="cpu",
    )

    validation_dataset = ProxyDataset(
        slice_tuple((A_tensor, Y_tensor, Z_tensor, W_tensor), val_indices),
        dens_ratio=dens_ratio_tensor[val_indices],
        dens_ratio_transformer=dens_ratio_transformer,
        transformers=transformers,
        device="cpu",
    )

    batch_size = 1024
    first_stage_train_dataloader = DataLoader(first_stage_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    second_stage_train_dataloader = DataLoader(second_stage_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    extra = {"Inlier_Ratio": inlier_ratio}

    outcome_model = None
    treatment_model = None
    third_stage_net = None
    third_stage_net_dr = None

    # --------------------------------------------------------------
    # OutcomeNet
    # --------------------------------------------------------------
    try:
        outcome_cfg = OutcomePCLConfig.low_dim()
        outcome_cfg.device = device
        outcome_cfg.plot_loss = False
        outcome_cfg.n_epochs = 100
        outcome_cfg.lr_first_stage_base = 1e-3
        outcome_cfg.reg_first_stage = (1e-2, 1e-1)
        outcome_cfg.reg_second_stage_first = (1e-2, 1e-1)
        outcome_cfg.reg_second_stage = (1.0, 10.0)
        outcome_cfg.second_stage_head_lr = 1.0
        outcome_cfg.second_stage_head_steps = 10
        outcome_cfg.second_stage_loss_name = "log_cosh"
        outcome_cfg.second_stage_loss_kwargs = {}

        first_stage_featurizer, treatment_featurizer, outcome_proxy_featurizer = (
            build_nets_for_outcome_pcl_net_proxy_misspecification_experiment(outcome_cfg.device)
        )

        outcome_model = OutcomeBridgePCLNET(
            first_stage_featurizer,
            treatment_featurizer,
            outcome_proxy_featurizer,
            None,
            None,
            device=outcome_cfg.device,
        )

        optimizers_stage1, optimizers_stage2, schedulers_stage1, schedulers_stage2 = (
            build_adam_optimizers_for_outcome_pcl_net(
                outcome_model,
                outcome_cfg.lr_first_stage,
                outcome_cfg.lr_second_stage_ax,
                outcome_cfg.lr_second_stage_w,
                outcome_cfg.weight_decay,
                gamma=outcome_cfg.scheduler_gamma,
            )
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
            plot_loss=False,
            validation_dataloader=validation_dataloader,
            do_A=do_A_tensor,
            EY_do_A=EY_do_A_tensor,
        )

        f_struct_pred_om = outcome_model.pred_structural_function(
            do_A_tensor,
            second_stage_train_dataset.transformers[0],
            second_stage_train_dataset.transformers[1],
        ).detach().cpu().numpy()

        add_result(rows, output_csv, "OutcomeNet", setting, seed, data_size, f_struct_pred_om, do_A, EY_do_A, extra)

    except Exception as e:
        add_result(rows, output_csv, "OutcomeNet", setting, seed, data_size, None, do_A, EY_do_A, extra, error=e)
        f_struct_pred_om = None

    # --------------------------------------------------------------
    # TreatmentNet
    # --------------------------------------------------------------
    try:
        treatment_cfg = TreatmentPCLConfig.low_dim()
        treatment_cfg.device = device
        treatment_cfg.plot_loss = False
        treatment_cfg.lr_first_stage_base = 5e-4
        treatment_cfg.reg_first_stage = (1e-2, 1e-1)
        treatment_cfg.reg_second_stage_first = (1e-2, 1e-1)
        treatment_cfg.reg_second_stage = (1.0, 10.0)
        treatment_cfg.n_epochs = 100
        treatment_cfg.second_stage_loss_name = "log_cosh"
        treatment_cfg.second_stage_loss_kwargs = {}
        treatment_cfg.second_stage_head_lr = 1.0
        treatment_cfg.second_stage_head_steps = 15
        treatment_cfg.negative_penalty = 10.0

        first_stage_featurizer, second_stage_ax_featurizer, treatment_proxy_featurizer = (
            build_nets_for_treatment_pcl_net_proxy_misspecification_experiment(treatment_cfg.device)
        )

        treatment_model = TreatmentBridgePCLNET(
            first_stage_featurizer,
            second_stage_ax_featurizer,
            treatment_proxy_featurizer,
            dens_ratio_transformer=copy.deepcopy(dens_ratio_transformer),
            device=treatment_cfg.device,
        )

        stage1_optimizers, stage2_optimizers, stage1_schedulers, stage2_schedulers = (
            build_adam_optimizers_for_treatment_pcl_net(
                treatment_model,
                lr_first_stage=treatment_cfg.lr_first_stage,
                lr_second_stage_ax=treatment_cfg.lr_second_stage_ax,
                lr_second_stage_z=treatment_cfg.lr_second_stage_z,
                weight_decay=treatment_cfg.weight_decay,
                gamma=treatment_cfg.scheduler_gamma,
            )
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
            negative_penalty=treatment_cfg.negative_penalty,
            log_per_epoch=treatment_cfg.log_per_epoch,
            plot_loss=False,
            validation_dataloader=validation_dataloader,
        )

        treatment_featurizer = torch.nn.Identity()

        third_stage_dataset, third_stage_dataset_val = create_third_stage_dataset_for_treatment_pcl_net_ate(
            treatment_model,
            second_stage_train_dataloader,
            validation_dataloader,
            TorchStandardScaler(),
            TorchStandardScaler(),
            treatment_featurizer=treatment_featurizer,
            dens_ratio_pred_tolerance=100.5,
            device=device,
        )

        train_kwargs_treatment = {
            "lr": 1e-2,
            "weight_decay": 1e-6,
            "n_epochs": 100,
            "loss_fn": nn.MSELoss(),
            "gap_penalty_weight": 0.0,
            "log_per_epoch": 10,
            "input_dim": 1,
            "output_dim": 1,
            "hidden_dims": [32, 64],
            "dropout_rate": 0.01,
        }

        third_stage_net = train_third_stage_treatment_pcl_net_ensemble(
            model_class=ConditionalMeanMLP,
            dataloader=DataLoader(third_stage_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
            val_dataloader=DataLoader(third_stage_dataset_val, batch_size=batch_size, shuffle=True, num_workers=0),
            n_members=5,
            device=device,
            **train_kwargs_treatment,
        )

        do_A_transformed = second_stage_train_dataloader.dataset.transformers[0].to(device).transform(
            torch.tensor(do_A, dtype=torch.float32, device=device)
        )
        phi_do_A = treatment_featurizer(do_A_transformed)
        phi_do_A_transformed = third_stage_dataset.input_transformer.to(device).transform(phi_do_A)

        f_struct_pred_tm = third_stage_dataset.outcome_transformer.to(device).inverse_transform(
            third_stage_net(phi_do_A_transformed)
        ).detach().cpu().numpy()

        add_result(rows, output_csv, "TreatmentNet", setting, seed, data_size, f_struct_pred_tm, do_A, EY_do_A, extra)

    except Exception as e:
        add_result(rows, output_csv, "TreatmentNet", setting, seed, data_size, None, do_A, EY_do_A, extra, error=e)
        f_struct_pred_tm = None

    # --------------------------------------------------------------
    # DRNet V1
    # --------------------------------------------------------------
    try:
        if outcome_model is None or treatment_model is None or f_struct_pred_om is None:
            raise RuntimeError("OutcomeNet or TreatmentNet failed; cannot run DRNet V1.")

        treatment_featurizer = torch.nn.Identity()

        third_stage_dataset_dr, third_stage_dataset_dr_val = create_third_stage_dataset_for_dr_proxy_neural_mean_embedding_ate(
            outcome_model,
            treatment_model,
            second_stage_train_dataloader,
            validation_dataloader,
            outcome_transformer=TorchStandardScaler(),
            input_transformer=TorchStandardScaler(),
            input_type="features",
            treatment_featurizer=treatment_featurizer,
            dens_ratio_pred_tolerance=100.5,
            device=device,
        )

        train_kwargs_dr = {
            "lr": 5e-3,
            "weight_decay": 1e-6,
            "n_epochs": 100,
            "loss_fn": nn.MSELoss(),
            "gap_penalty_weight": 0.0,
            "log_per_epoch": 10,
            "input_dim": 1,
            "output_dim": 1,
            "hidden_dims": [32, 64],
            "dropout_rate": 0.01,
        }

        third_stage_net_dr = train_third_stage_treatment_pcl_net_ensemble(
            model_class=ConditionalMeanMLP,
            dataloader=DataLoader(third_stage_dataset_dr, batch_size=batch_size, shuffle=True, num_workers=0),
            val_dataloader=DataLoader(third_stage_dataset_dr_val, batch_size=batch_size, shuffle=True, num_workers=0),
            n_members=5,
            device=device,
            **train_kwargs_dr,
        )

        do_A_transformed = second_stage_train_dataloader.dataset.transformers[0].to(device).transform(
            torch.tensor(do_A, dtype=torch.float32, device=device)
        )
        do_A_transformed = treatment_featurizer(do_A_transformed)
        do_A_transformed = third_stage_dataset_dr.input_transformer.to(device).transform(do_A_transformed)

        f_struct_pred_slack = third_stage_dataset_dr.outcome_transformer.to(device).inverse_transform(
            third_stage_net_dr(do_A_transformed)
        ).detach().cpu().numpy()

        f_struct_pred_dr_v1 = f_struct_pred_om + f_struct_pred_slack

        add_result(rows, output_csv, "DRNet_V1", setting, seed, data_size, f_struct_pred_dr_v1, do_A, EY_do_A, extra)

    except Exception as e:
        add_result(rows, output_csv, "DRNet_V1", setting, seed, data_size, None, do_A, EY_do_A, extra, error=e)

    # --------------------------------------------------------------
    # DRNet V2
    # --------------------------------------------------------------
    try:
        if outcome_model is None or treatment_model is None or f_struct_pred_om is None or f_struct_pred_tm is None:
            raise RuntimeError("OutcomeNet or TreatmentNet failed; cannot run DRNet V2.")

        treatment_featurizer = torch.nn.Identity()

        third_stage_dataset_dr, third_stage_dataset_dr_val = create_third_stage_dataset_for_dr_proxy_neural_mean_embedding_ate_v2(
            outcome_model,
            treatment_model,
            second_stage_train_dataloader,
            validation_dataloader,
            outcome_transformer=TorchStandardScaler(),
            input_transformer=TorchStandardScaler(),
            input_type="features",
            treatment_featurizer=treatment_featurizer,
            dens_ratio_pred_tolerance=100.5,
            device=device,
        )

        train_kwargs_dr = {
            "lr": 5e-3,
            "weight_decay": 1e-6,
            "n_epochs": 100,
            "loss_fn": nn.MSELoss(),
            "gap_penalty_weight": 0.0,
            "log_per_epoch": 10,
            "input_dim": 1,
            "output_dim": 1,
            "hidden_dims": [32, 64],
            "dropout_rate": 0.01,
        }

        third_stage_net_dr = train_third_stage_treatment_pcl_net_ensemble(
            model_class=ConditionalMeanMLP,
            dataloader=DataLoader(third_stage_dataset_dr, batch_size=batch_size, shuffle=True, num_workers=0),
            val_dataloader=DataLoader(third_stage_dataset_dr_val, batch_size=batch_size, shuffle=True, num_workers=0),
            n_members=5,
            device=device,
            **train_kwargs_dr,
        )

        do_A_transformed = second_stage_train_dataloader.dataset.transformers[0].to(device).transform(
            torch.tensor(do_A, dtype=torch.float32, device=device)
        )
        do_A_transformed = treatment_featurizer(do_A_transformed)
        do_A_transformed = third_stage_dataset_dr.input_transformer.to(device).transform(do_A_transformed)

        f_struct_pred_slack = third_stage_dataset_dr.outcome_transformer.to(device).inverse_transform(
            third_stage_net_dr(do_A_transformed)
        ).detach().cpu().numpy()

        f_struct_pred_dr_v2 = f_struct_pred_om + f_struct_pred_tm - f_struct_pred_slack

        add_result(rows, output_csv, "DRNet_V2", setting, seed, data_size, f_struct_pred_dr_v2, do_A, EY_do_A, extra)

    except Exception as e:
        add_result(rows, output_csv, "DRNet_V2", setting, seed, data_size, None, do_A, EY_do_A, extra, error=e)

    # --------------------------------------------------------------
    # Kernel baselines: DRKPV, KPV, Kernel Alternative Proxy
    # --------------------------------------------------------------
    try:
        Z_tensor_k = torch.tensor(Z_np, dtype=torch.float32, device=device)
        W_tensor_k = torch.tensor(W_np, dtype=torch.float32, device=device)
        A_tensor_k = torch.tensor(A_np, dtype=torch.float32, device=device)
        Y_tensor_k = torch.tensor(Y_np, dtype=torch.float32, device=device)
        A_test_tensor_k = torch.tensor(do_A, dtype=torch.float32, device=device)

        transformers_k = [
            TorchStandardScaler(),
            TorchStandardScaler(),
            TorchStandardScaler(),
            TorchStandardScaler(),
        ]

        A_tensor_transformed_k = transformers_k[0].fit_transform(A_tensor_k)
        Y_tensor_transformed_k = transformers_k[1].fit_transform(Y_tensor_k)
        Z_tensor_transformed_k = transformers_k[2].fit_transform(Z_tensor_k)
        W_tensor_transformed_k = transformers_k[3].fit_transform(W_tensor_k)
        A_test_tensor_transformed_k = transformers_k[0].transform(A_test_tensor_k)

        treatment_bridge_algo_param_dict_default = {
            "kernel_A": RBF(use_length_scale_heuristic=True),
            "kernel_W": RBF(use_length_scale_heuristic=True),
            "kernel_Z": RBF(use_length_scale_heuristic=True),
            "lambda_": 1e-3,
            "eta": 1e-3,
            "lambda2_": 1e-3,
            "optimize_lambda_parameters": True,
            "optimize_eta_parameter": True,
            "lambda_optimization_range": (1e-5, 1.0),
            "eta_optimization_range": (5e-4, 1e-2),
            "stage1_perc": 0.5,
            "regularization_grid_points": 25,
            "make_psd_eps": 5e-6,
            "label_variance_in_lambda_opt": 0.0,
            "label_variance_in_eta_opt": 3.0,
        }

        outcome_bridge_kpv_algo_param_dict_default = {
            "algorithm_name": "Kernel_Proxy_Variable",
            "kernel_A": RBF(use_length_scale_heuristic=True),
            "kernel_W": RBF(use_length_scale_heuristic=True),
            "kernel_Z": RBF(use_length_scale_heuristic=True),
            "lambda1_": 0.1,
            "lambda2_": 0.1,
            "optimize_lambda1_parameter": True,
            "optimize_lambda2_parameter": True,
            "lambda1_optimization_range": (1e-5, 1.0),
            "lambda2_optimization_range": (1e-5, 1.0),
            "stage1_perc": 0.5,
            "regularization_grid_points": 25,
            "make_psd_eps": 5e-6,
        }

        model_DR = DoublyRobustKernelProxyATE(
            treatment_bridge_params=treatment_bridge_algo_param_dict_default,
            outcome_bridge_params=outcome_bridge_kpv_algo_param_dict_default,
            lambda_DR_optimization_range=(1e-5, 1.0),
            regularization_grid_points=25,
        )

        model_DR.fit(
            (A_tensor_transformed_k, Z_tensor_transformed_k, W_tensor_transformed_k),
            Y_tensor_transformed_k,
        )

        do_A_size = do_A.reshape(-1).shape[0]

        f_struct_pred_DR = model_DR.predict(
            A_test_tensor_transformed_k,
            transformers_k[1],
        ).reshape(do_A_size, -1).detach().cpu().numpy()

        f_struct_pred_KPV = model_DR.outcome_bridge_algo.predict(
            A_test_tensor_transformed_k,
            transformers_k[1],
        ).reshape(do_A_size, -1).detach().cpu().numpy()

        f_struct_pred_KAP = model_DR.treatment_bridge_algo.predict(
            A_test_tensor_transformed_k,
            transformers_k[1],
        ).reshape(do_A_size, -1).detach().cpu().numpy()

        add_result(rows, output_csv, "DRKPV", setting, seed, data_size, f_struct_pred_DR, do_A, EY_do_A, extra)
        add_result(rows, output_csv, "KPV", setting, seed, data_size, f_struct_pred_KPV, do_A, EY_do_A, extra)
        add_result(rows, output_csv, "KernelAlternativeProxy", setting, seed, data_size, f_struct_pred_KAP, do_A, EY_do_A, extra)

    except Exception as e:
        add_result(rows, output_csv, "DRKPV", setting, seed, data_size, None, do_A, EY_do_A, extra, error=e)
        add_result(rows, output_csv, "KPV", setting, seed, data_size, None, do_A, EY_do_A, extra, error=e)
        add_result(rows, output_csv, "KernelAlternativeProxy", setting, seed, data_size, None, do_A, EY_do_A, extra, error=e)

    del outcome_model, treatment_model, third_stage_net, third_stage_net_dr

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, choices=[1, 2, 3, 4, 5, 6], default=None)
    parser.add_argument("--data_size", type=int, default=2000)
    parser.add_argument("--seeds", type=str, default="default")
    parser.add_argument("--outdir", type=str, default="../Results/Proxy_Misspecification_6Settings")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu")
    seeds = parse_seeds(args.seeds)

    settings = [args.scenario] if args.scenario is not None else [1, 2, 3, 4, 5, 6]

    if args.run_name is None:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        run_name = args.run_name

    for setting in settings:
        output_csv = os.path.join(
            args.outdir,
            f"proxy_misspecification_setting_{setting}_n{args.data_size}_{run_name}.csv",
        )

        print(f"Writing results to: {output_csv}")
        print(f"Setting: {setting}; seeds: {seeds}; device: {device}")

        rows = []

        for seed in seeds:
            print(f"\n=== Setting {setting}, seed {seed} ===")
            run_one_seed(
                setting=setting,
                seed=seed,
                data_size=args.data_size,
                output_csv=output_csv,
                rows=rows,
                device=device,
            )


if __name__ == "__main__":
    main()