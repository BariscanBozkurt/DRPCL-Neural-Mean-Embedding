import numpy as np
import torch
import torch.nn as nn   
from typing import Tuple
from torch.nn.utils import spectral_norm
import torch.nn.utils.parametrizations as parametrizations

class GaussianFourierProjection(nn.Module):
    """
    Projects 1D input to high-dimensional Fourier features.
    x -> [sin(2*pi*x*B), cos(2*pi*x*B)]
    This guarantees inputs are mapped to a high-rank manifold.
    """
    def __init__(self, input_dim=1, embedding_size=64, scale=1.0):
        super().__init__()
        # B is a fixed, non-learnable matrix sampled from Normal(0, scale)
        self.B = nn.Parameter(torch.randn(input_dim, embedding_size) * scale, requires_grad=False)
        self.embedding_size = embedding_size

    def forward(self, x):
        # x: (N, 1) -> (N, embedding_size)
        x_proj = 2 * np.pi * x @ self.B 
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1) / np.sqrt(self.embedding_size)
    

def build_nets_for_treatment_pcl_net_synthetic_low_dim_experiment(device: str = "cuda") -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module]:
    """
    Builds efficient featurizer networks for the low-dimensional synthetic benchmark.
    Significantly reduces final output dimensions (from 64/128 to 16/32) to speed up 
    closed-form matrix operations (reducing the final layer size from 8192 to 512).
    """
    FIRST_STAGE_OUTPUT_DIM = 128
    Z_OUTPUT_DIM = 16
    SECOND_STAGE_AX_OUTPUT_DIM = 32
    INTERNAL_WIDTH = 512 # Keep internal layers reasonably wide
    DROPOUT_RATE = 0.05


    class FeaturizerBase(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, final_activation: str = "linear"):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, INTERNAL_WIDTH),
                nn.LayerNorm(INTERNAL_WIDTH),
                nn.GELU(), 
                # nn.BatchNorm1d(INTERNAL_WIDTH),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(INTERNAL_WIDTH, INTERNAL_WIDTH * 2),
                nn.LayerNorm(INTERNAL_WIDTH * 2),
                nn.GELU(),
                # nn.BatchNorm1d(INTERNAL_WIDTH // 2),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(INTERNAL_WIDTH * 2, output_dim),
                nn.LayerNorm(output_dim),
            )
            self.output_dim = output_dim
            self.final_activation = final_activation
            self.final_bn = nn.BatchNorm1d(output_dim)
            self.tanh = nn.Tanh()
            self.gelu = nn.GELU()
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.net(x)
            if self.final_activation == "tanh":
                return self.tanh(out) 
            elif self.final_activation == "gelu":
                return self.gelu(out)
            return out
    
    # --- Treatment Featurizers (Input: 1D) ---
    class FirstStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 3):
            super().__init__(input_dim=input_dim, output_dim=FIRST_STAGE_OUTPUT_DIM, final_activation="gelu")

    # --- Outcome Proxy Featurizer (Input: 2D) ---
    class SecondStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 1):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_AX_OUTPUT_DIM, final_activation="gelu")

    # --- Treatment Proxy Featurizer (Input: 2D) ---
    class TreatmentProxyFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 2):
            super().__init__(input_dim=input_dim, output_dim=Z_OUTPUT_DIM, final_activation="gelu")

    def _small_init(m):
        if isinstance(m, nn.Linear):
            # We use a very small scale (0.01) to squash the initial output variance
            nn.init.kaiming_normal_(m.weight, a=0.01) # Use GELU/ReLU compatible init
            m.weight.data *= 0.05 
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 0.01) # Scale down LayerNorm gains too

    # Instantiate the models
    first_stage_featurizer = FirstStageFeaturizer().to(device)
    second_stage_ax_featurizer = SecondStageFeaturizer().to(device)
    treatment_proxy_featurizer = TreatmentProxyFeaturizer().to(device)
    treatment_proxy_featurizer.apply(_small_init)
    return first_stage_featurizer, second_stage_ax_featurizer, treatment_proxy_featurizer


def build_nets_for_treatment_pcl_net_synthetic_high_dim_new_version_experimentV1(device: str = "cuda") -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module]:
    FIRST_STAGE_OUTPUT_DIM = 128
    Z_OUTPUT_DIM = 8
    SECOND_STAGE_AX_OUTPUT_DIM = 16
    INTERNAL_WIDTH = 512 # Keep internal layers reasonably wide
    DROPOUT_RATE = 0.1


    class FeaturizerBase(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, DROPOUT_RATE: float = 0.0):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, INTERNAL_WIDTH),
                nn.LayerNorm(INTERNAL_WIDTH),
                nn.GELU(),           
                # nn.BatchNorm1d(INTERNAL_WIDTH),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(INTERNAL_WIDTH, INTERNAL_WIDTH * 2),
                nn.LayerNorm(INTERNAL_WIDTH * 2),
                nn.GELU(),
                # nn.BatchNorm1d(INTERNAL_WIDTH * 2),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(INTERNAL_WIDTH * 2, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(), 
                # nn.BatchNorm1d(output_dim)
            )
            self.output_dim = output_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)
    
    # --- Treatment Featurizers (Input: 1D) ---
    class FirstStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 111):
            super().__init__(input_dim=input_dim, output_dim=FIRST_STAGE_OUTPUT_DIM)

    # --- Outcome Proxy Featurizer (Input: 2D) ---
    class SecondStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 101):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_AX_OUTPUT_DIM)

    # --- Treatment Proxy Featurizer (Input: 2D) ---
    class TreatmentProxyFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 10, DROPOUT_RATE: float = DROPOUT_RATE):
            super().__init__(input_dim=input_dim, output_dim=Z_OUTPUT_DIM)


    # Instantiate the models
    first_stage_featurizer = FirstStageFeaturizer().to(device)
    second_stage_ax_featurizer = SecondStageFeaturizer().to(device)
    treatment_proxy_featurizer = TreatmentProxyFeaturizer().to(device)

    return first_stage_featurizer, second_stage_ax_featurizer, treatment_proxy_featurizer
    
def build_nets_for_treatment_pcl_net_synthetic_high_dim_new_version_experiment(device: str = "cuda") -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module]:
    """
    Builds efficient featurizer networks for the low-dimensional synthetic benchmark.
    Significantly reduces final output dimensions (from 64/128 to 16/32) to speed up 
    closed-form matrix operations (reducing the final layer size from 8192 to 512).
    """
    FIRST_STAGE_OUTPUT_DIM = 128
    Z_OUTPUT_DIM = 8
    SECOND_STAGE_AX_OUTPUT_DIM = 16
    INTERNAL_WIDTH = 512 # Keep internal layers reasonably wide
    DROPOUT_RATE = 0.1


    class FeaturizerBase(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, DROPOUT_RATE: float = 0.0):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, INTERNAL_WIDTH),
                nn.LayerNorm(INTERNAL_WIDTH),
                nn.GELU(),           
                nn.BatchNorm1d(INTERNAL_WIDTH),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(INTERNAL_WIDTH, INTERNAL_WIDTH * 2),
                nn.LayerNorm(INTERNAL_WIDTH * 2),
                nn.GELU(),
                nn.BatchNorm1d(INTERNAL_WIDTH * 2),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(INTERNAL_WIDTH * 2, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(), 
                nn.BatchNorm1d(output_dim)
            )
            self.output_dim = output_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)
    
    # --- Treatment Featurizers (Input: 1D) ---
    class FirstStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 111):
            super().__init__(input_dim=input_dim, output_dim=FIRST_STAGE_OUTPUT_DIM)

    # --- Outcome Proxy Featurizer (Input: 2D) ---
    class SecondStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 101):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_AX_OUTPUT_DIM)

    # --- Treatment Proxy Featurizer (Input: 2D) ---
    class TreatmentProxyFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 10, DROPOUT_RATE: float = DROPOUT_RATE):
            super().__init__(input_dim=input_dim, output_dim=Z_OUTPUT_DIM)

    def _small_init(m):
        if isinstance(m, nn.Linear):
            # We use a very small scale (0.01) to squash the initial output variance
            nn.init.kaiming_normal_(m.weight, a=0.05) # Use GELU/ReLU compatible init
            m.weight.data *= 0.05 
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 0.05) # Scale down LayerNorm gains too

    # Instantiate the models
    first_stage_featurizer = FirstStageFeaturizer().to(device)
    second_stage_ax_featurizer = SecondStageFeaturizer().to(device)
    treatment_proxy_featurizer = TreatmentProxyFeaturizer().to(device)
    treatment_proxy_featurizer.apply(_small_init)

    return first_stage_featurizer, second_stage_ax_featurizer, treatment_proxy_featurizer
    # FIRST_STAGE_OUTPUT_DIM = 128
    # Z_OUTPUT_DIM = 8
    # SECOND_STAGE_AX_OUTPUT_DIM = 16
    # INTERNAL_WIDTH = 512 # Keep internal layers reasonably wide
    # DROPOUT_RATE = 0.1


    # class FeaturizerBase(nn.Module):
    #     def __init__(self, input_dim: int, output_dim: int, DROPOUT_RATE: float = 0.0):
    #         super().__init__()
    #         self.net = nn.Sequential(
    #             nn.Linear(input_dim, INTERNAL_WIDTH),
    #             nn.LayerNorm(INTERNAL_WIDTH),
    #             nn.GELU(),           
    #             # nn.BatchNorm1d(INTERNAL_WIDTH),
    #             nn.Dropout(DROPOUT_RATE),
    #             nn.Linear(INTERNAL_WIDTH, INTERNAL_WIDTH * 2),
    #             nn.LayerNorm(INTERNAL_WIDTH * 2),
    #             nn.GELU(),
    #             # nn.BatchNorm1d(INTERNAL_WIDTH * 2),
    #             nn.Dropout(DROPOUT_RATE),
    #             nn.Linear(INTERNAL_WIDTH * 2, output_dim),
    #             nn.LayerNorm(output_dim),
    #             nn.GELU(), 
    #             # nn.BatchNorm1d(output_dim)
    #         )
    #         self.output_dim = output_dim

    #     def forward(self, x: torch.Tensor) -> torch.Tensor:
    #         return self.net(x)
    
    # # --- Treatment Featurizers (Input: 1D) ---
    # class FirstStageFeaturizer(FeaturizerBase):
    #     def __init__(self, input_dim: int = 111):
    #         super().__init__(input_dim=input_dim, output_dim=FIRST_STAGE_OUTPUT_DIM)

    # # --- Outcome Proxy Featurizer (Input: 2D) ---
    # class SecondStageFeaturizer(FeaturizerBase):
    #     def __init__(self, input_dim: int = 101):
    #         super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_AX_OUTPUT_DIM)

    # # --- Treatment Proxy Featurizer (Input: 2D) ---
    # class TreatmentProxyFeaturizer(FeaturizerBase):
    #     def __init__(self, input_dim: int = 10, DROPOUT_RATE: float = DROPOUT_RATE):
    #         super().__init__(input_dim=input_dim, output_dim=Z_OUTPUT_DIM)


    # # Instantiate the models
    # first_stage_featurizer = FirstStageFeaturizer().to(device)
    # second_stage_ax_featurizer = SecondStageFeaturizer().to(device)
    # treatment_proxy_featurizer = TreatmentProxyFeaturizer().to(device)

    # return first_stage_featurizer, second_stage_ax_featurizer, treatment_proxy_featurizer


def build_nets_for_treatment_pcl_net_synthetic_high_dim_experiment(device: str = "cuda") -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module]:
    """
    Builds efficient featurizer networks for the low-dimensional synthetic benchmark.
    Significantly reduces final output dimensions (from 64/128 to 16/32) to speed up 
    closed-form matrix operations (reducing the final layer size from 8192 to 512).
    """
    FIRST_STAGE_OUTPUT_DIM = 128
    Z_OUTPUT_DIM = 8
    SECOND_STAGE_AX_OUTPUT_DIM = 16
    INTERNAL_WIDTH = 512 # Keep internal layers reasonably wide
    DROPOUT_RATE = 0.2


    class FeaturizerBase(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, DROPOUT_RATE: float = 0.0):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, INTERNAL_WIDTH),
                nn.LayerNorm(INTERNAL_WIDTH),
                nn.GELU(),           
                nn.BatchNorm1d(INTERNAL_WIDTH),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(INTERNAL_WIDTH, INTERNAL_WIDTH * 2),
                nn.LayerNorm(INTERNAL_WIDTH * 2),
                nn.GELU(),
                nn.BatchNorm1d(INTERNAL_WIDTH * 2),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(INTERNAL_WIDTH * 2, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(), 
                nn.BatchNorm1d(output_dim)
            )
            self.output_dim = output_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)
    
    # --- Treatment Featurizers (Input: 1D) ---
    class FirstStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 111):
            super().__init__(input_dim=input_dim, output_dim=FIRST_STAGE_OUTPUT_DIM)

    # --- Outcome Proxy Featurizer (Input: 2D) ---
    class SecondStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 101):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_AX_OUTPUT_DIM)

    # --- Treatment Proxy Featurizer (Input: 2D) ---
    class TreatmentProxyFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 10, DROPOUT_RATE: float = DROPOUT_RATE):
            super().__init__(input_dim=input_dim, output_dim=Z_OUTPUT_DIM)

    def _small_init(m):
        if isinstance(m, nn.Linear):
            # We use a very small scale (0.01) to squash the initial output variance
            nn.init.kaiming_normal_(m.weight, a=0.05) # Use GELU/ReLU compatible init
            m.weight.data *= 0.05 
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 0.05) # Scale down LayerNorm gains too

    # Instantiate the models
    first_stage_featurizer = FirstStageFeaturizer().to(device)
    second_stage_ax_featurizer = SecondStageFeaturizer().to(device)
    treatment_proxy_featurizer = TreatmentProxyFeaturizer().to(device)
    treatment_proxy_featurizer.apply(_small_init)

    return first_stage_featurizer, second_stage_ax_featurizer, treatment_proxy_featurizer


def build_nets_for_treatment_pcl_net_abortion_and_crime_experiment(device: str = "cuda") -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module]:
    """
    Builds efficient featurizer networks for the low-dimensional synthetic benchmark.
    Significantly reduces final output dimensions (from 64/128 to 16/32) to speed up 
    closed-form matrix operations (reducing the final layer size from 8192 to 512).
    """
    FIRST_STAGE_OUTPUT_DIM = 16
    Z_OUTPUT_DIM = 8
    SECOND_STAGE_AX_OUTPUT_DIM = 8
    INTERNAL_WIDTH = 32
    DROPOUT_RATE = 0.01

    class GaussianNoise(nn.Module):
        def __init__(self, std=0.05):
            super().__init__()
            self.std = std
        def forward(self, x):
            if self.training and self.std > 0:
                return x + torch.randn_like(x) * self.std
            return x


    class FeaturizerBase(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, final_activation: str = "linear"):
            super().__init__()
            self.net = nn.Sequential(
                # GaussianNoise(std=0.01), 
                nn.Linear(input_dim, INTERNAL_WIDTH),
                nn.LayerNorm(INTERNAL_WIDTH),
                nn.GELU(), 
                # nn.BatchNorm1d(INTERNAL_WIDTH),
                nn.Dropout(DROPOUT_RATE),
                # nn.Linear(INTERNAL_WIDTH, INTERNAL_WIDTH // 2),
                # nn.LayerNorm(INTERNAL_WIDTH // 2),
                # nn.GELU(),
                # nn.BatchNorm1d(INTERNAL_WIDTH // 2),
                # nn.Dropout(DROPOUT_RATE),
                nn.Linear(INTERNAL_WIDTH, output_dim),
                # nn.LayerNorm(output_dim),
            )
            self.output_dim = output_dim
            self.final_activation = final_activation
            self.tanh = nn.Tanh()
            self.gelu = nn.GELU()
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.net(x)
            if self.final_activation == "tanh":
                return self.tanh(out) 
            elif self.final_activation == "gelu":
                return self.gelu(out)
            return out
    
    # --- Treatment Featurizers (Input: 1D) ---
    class FirstStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 4, final_activation="gelu"):
            super().__init__(input_dim=input_dim, output_dim=FIRST_STAGE_OUTPUT_DIM)

    # --- Outcome Proxy Featurizer (Input: 2D) ---
    class SecondStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 1, final_activation="gelu"):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_AX_OUTPUT_DIM)

    # --- Treatment Proxy Featurizer (Input: 2D) ---
    class TreatmentProxyFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 1, final_activation="gelu"):
            super().__init__(input_dim=input_dim, output_dim=Z_OUTPUT_DIM)


    # Instantiate the models
    first_stage_featurizer = FirstStageFeaturizer().to(device)
    second_stage_ax_featurizer = SecondStageFeaturizer().to(device)
    treatment_proxy_featurizer = TreatmentProxyFeaturizer().to(device)

    return first_stage_featurizer, second_stage_ax_featurizer, treatment_proxy_featurizer


def build_nets_for_treatment_pcl_net_dsprite_experiment( device: str = "cuda") -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module]:
    """
    Builds efficient featurizer networks for the low-dimensional synthetic benchmark.
    Significantly reduces final output dimensions (from 64/128 to 16/32) to speed up 
    closed-form matrix operations (reducing the final layer size from 8192 to 512).
    """
    DIM_IM = 4096
    FIRST_STAGE_OUTPUT_DIM = 128
    Z_OUTPUT_DIM = 8
    SECOND_STAGE_AX_OUTPUT_DIM = 32
    INTERNAL_WIDTH = 1024 # Keep internal layers reasonably wide
    DROPOUT_RATE = 0.05


    class FeaturizerBase(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, final_activation: str = "linear"):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, INTERNAL_WIDTH),
                nn.LayerNorm(INTERNAL_WIDTH),
                nn.ReLU(), 
                nn.BatchNorm1d(INTERNAL_WIDTH),
                nn.Dropout(DROPOUT_RATE),

                nn.Linear(INTERNAL_WIDTH, INTERNAL_WIDTH // 2),
                nn.LayerNorm(INTERNAL_WIDTH // 2),
                nn.ReLU(),
                nn.BatchNorm1d(INTERNAL_WIDTH // 2),
                nn.Dropout(DROPOUT_RATE),

                nn.Linear(INTERNAL_WIDTH // 2, INTERNAL_WIDTH // 4),
                nn.LayerNorm(INTERNAL_WIDTH // 4),
                nn.ReLU(),
                nn.BatchNorm1d(INTERNAL_WIDTH // 4),
                nn.Dropout(DROPOUT_RATE),
                
                nn.Linear(INTERNAL_WIDTH // 4, output_dim),
                nn.LayerNorm(output_dim),
            )
            self.output_dim = output_dim
            self.final_activation = final_activation
            self.tanh = nn.Tanh()
            self.gelu = nn.ReLU()
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.net(x)
            if self.final_activation == "tanh":
                return self.tanh(out) 
            elif self.final_activation == "gelu":
                return self.gelu(out)
            return out
    
    # --- Treatment Featurizers (Input: 1D) ---
    class FirstStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 2 * DIM_IM):
            super().__init__(input_dim=input_dim, output_dim=FIRST_STAGE_OUTPUT_DIM, final_activation="gelu")

    # --- Outcome Proxy Featurizer (Input: 2D) ---
    class SecondStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = DIM_IM):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_AX_OUTPUT_DIM, final_activation="gelu")

    # --- Treatment Proxy Featurizer (Input: 3D) ---
    class TreatmentProxyFeaturizer(nn.Module):
        def __init__(self, input_dim: int = 3, output_dim: int = Z_OUTPUT_DIM, final_activation: str = "gelu"):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 8),
                nn.LayerNorm(8),
                nn.ReLU(), 
                nn.BatchNorm1d(8),
                nn.Dropout(DROPOUT_RATE),

                nn.Linear(8, 4),
                nn.LayerNorm(4),
                nn.ReLU(),
                nn.BatchNorm1d(4),
                nn.Dropout(DROPOUT_RATE),

                nn.Linear(4, output_dim),
                nn.LayerNorm(output_dim),
            )
            self.output_dim = output_dim
            self.final_activation = final_activation
            self.tanh = nn.Tanh()
            self.gelu = nn.ReLU()
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.net(x)
            if self.final_activation == "tanh":
                return self.tanh(out) 
            elif self.final_activation == "gelu":
                return self.gelu(out)
            return out

    def _small_init(m):
        if isinstance(m, nn.Linear):
            # We use a very small scale (0.01) to squash the initial output variance
            nn.init.kaiming_normal_(m.weight, a=0.05) # Use GELU/ReLU compatible init
            m.weight.data *= 0.05 
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 0.05) # Scale down LayerNorm gains too

    # Instantiate the models
    first_stage_featurizer = FirstStageFeaturizer().to(device)
    second_stage_ax_featurizer = SecondStageFeaturizer().to(device)
    treatment_proxy_featurizer = TreatmentProxyFeaturizer().to(device)
    # treatment_proxy_featurizer.apply(_small_init)

    return first_stage_featurizer, second_stage_ax_featurizer, treatment_proxy_featurizer


def build_nets_for_treatment_pcl_net_dsprite_experiment_with_compressed_images(latent_dim: int = 8, device: str = "cuda") -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module]:
    """
    Builds efficient featurizer networks for the low-dimensional synthetic benchmark.
    Significantly reduces final output dimensions (from 64/128 to 16/32) to speed up 
    closed-form matrix operations (reducing the final layer size from 8192 to 512).
    """
    DIM_IM = latent_dim
    FIRST_STAGE_OUTPUT_DIM = 32
    Z_OUTPUT_DIM = 16
    SECOND_STAGE_AX_OUTPUT_DIM = 16
    INTERNAL_WIDTH = 256 # Keep internal layers reasonably wide
    DROPOUT_RATE = 0.01


    class FeaturizerBase(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, final_activation: str = "linear"):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, INTERNAL_WIDTH),
                nn.LayerNorm(INTERNAL_WIDTH),
                nn.GELU(), 
                nn.BatchNorm1d(INTERNAL_WIDTH),
                nn.Dropout(DROPOUT_RATE),

                nn.Linear(INTERNAL_WIDTH, INTERNAL_WIDTH // 2),
                nn.LayerNorm(INTERNAL_WIDTH // 2),
                nn.GELU(),
                nn.BatchNorm1d(INTERNAL_WIDTH // 2),
                nn.Dropout(DROPOUT_RATE),

                # nn.Linear(INTERNAL_WIDTH // 2, INTERNAL_WIDTH // 4),
                # nn.LayerNorm(INTERNAL_WIDTH // 4),
                # nn.GELU(),
                # nn.BatchNorm1d(INTERNAL_WIDTH // 4),
                # nn.Dropout(DROPOUT_RATE),
                
                nn.Linear(INTERNAL_WIDTH // 2, output_dim),
                nn.LayerNorm(output_dim),
            )
            self.output_dim = output_dim
            self.final_activation = final_activation
            self.tanh = nn.Tanh()
            self.gelu = nn.GELU()
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.net(x)
            if self.final_activation == "tanh":
                return self.tanh(out) 
            elif self.final_activation == "gelu":
                return self.gelu(out)
            return out
    
    # --- Treatment Featurizers (Input: 1D) ---
    class FirstStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 2 * DIM_IM):
            super().__init__(input_dim=input_dim, output_dim=FIRST_STAGE_OUTPUT_DIM, final_activation="gelu")

    # --- Outcome Proxy Featurizer (Input: 2D) ---
    class SecondStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = DIM_IM):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_AX_OUTPUT_DIM, final_activation="gelu")

    # --- Treatment Proxy Featurizer (Input: 2D) ---
    class TreatmentProxyFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 3):
            super().__init__(input_dim=input_dim, output_dim=Z_OUTPUT_DIM, final_activation="gelu")


    # Instantiate the models
    first_stage_featurizer = FirstStageFeaturizer().to(device)
    second_stage_ax_featurizer = SecondStageFeaturizer().to(device)
    treatment_proxy_featurizer = TreatmentProxyFeaturizer().to(device)

    return first_stage_featurizer, second_stage_ax_featurizer, treatment_proxy_featurizer


def build_nets_for_treatment_pcl_net_demand_experiment(device: str = "cuda") -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module]:
    """
    Builds efficient featurizer networks for the low-dimensional synthetic benchmark.
    Significantly reduces final output dimensions (from 64/128 to 16/32) to speed up 
    closed-form matrix operations (reducing the final layer size from 8192 to 512).
    """
    FIRST_STAGE_OUTPUT_DIM = 16
    Z_OUTPUT_DIM = 4
    SECOND_STAGE_AX_OUTPUT_DIM = 4
    INTERNAL_WIDTH = 32 # Keep internal layers reasonably wide
    DROPOUT_RATE = 0.01


    class FeaturizerBase(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, final_activation: str = "linear"):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, INTERNAL_WIDTH),
                nn.LayerNorm(INTERNAL_WIDTH),
                nn.GELU(), 
                nn.BatchNorm1d(INTERNAL_WIDTH),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(INTERNAL_WIDTH, INTERNAL_WIDTH // 2),
                nn.LayerNorm(INTERNAL_WIDTH // 2),
                nn.GELU(),
                nn.BatchNorm1d(INTERNAL_WIDTH // 2),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(INTERNAL_WIDTH // 2, output_dim),
                nn.LayerNorm(output_dim),
            )
            self.output_dim = output_dim
            self.final_activation = final_activation
            self.tanh = nn.Tanh()
            self.relu = nn.GELU()
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.net(x)
            if self.final_activation == "tanh":
                return self.tanh(out) 
            elif self.final_activation == "relu":
                return self.relu(out)
            return out
    
    # --- Treatment Featurizers (Input: 1D) ---
    class FirstStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 2):
            super().__init__(input_dim=input_dim, output_dim=FIRST_STAGE_OUTPUT_DIM, final_activation="relu")

    # --- Outcome Proxy Featurizer (Input: 2D) ---
    class SecondStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 1):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_AX_OUTPUT_DIM, final_activation="relu")

    # --- Treatment Proxy Featurizer (Input: 2D) ---
    class TreatmentProxyFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 2):
            super().__init__(input_dim=input_dim, output_dim=Z_OUTPUT_DIM, final_activation="relu")


    # Instantiate the models
    first_stage_featurizer = FirstStageFeaturizer().to(device)
    second_stage_ax_featurizer = SecondStageFeaturizer().to(device)
    treatment_proxy_featurizer = TreatmentProxyFeaturizer().to(device)

    return first_stage_featurizer, second_stage_ax_featurizer, treatment_proxy_featurizer


def build_nets_for_treatment_pcl_net_synthetic_cate_experiment(device: str = "cuda") -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module]:
    """
    Builds efficient featurizer networks for the low-dimensional synthetic benchmark.
    Significantly reduces final output dimensions (from 64/128 to 16/32) to speed up 
    closed-form matrix operations (reducing the final layer size from 8192 to 512).
    """
    FIRST_STAGE_OUTPUT_DIM = 64
    Z_OUTPUT_DIM = 16
    SECOND_STAGE_AX_OUTPUT_DIM = 4
    INTERNAL_WIDTH = 256 # Keep internal layers reasonably wide
    DROPOUT_RATE = 0.1


    class FeaturizerBase(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, DROPOUT_RATE: float = 0.0):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, INTERNAL_WIDTH),
                nn.LayerNorm(INTERNAL_WIDTH),
                nn.GELU(),           
                nn.BatchNorm1d(INTERNAL_WIDTH),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(INTERNAL_WIDTH, INTERNAL_WIDTH * 2),
                nn.LayerNorm(INTERNAL_WIDTH * 2),
                nn.GELU(),
                nn.BatchNorm1d(INTERNAL_WIDTH * 2),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(INTERNAL_WIDTH * 2, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(), 
                # nn.BatchNorm1d(output_dim)
            )
            self.output_dim = output_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)
    
    # --- Treatment Featurizers (Input: 1D) ---
    class FirstStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 5):
            super().__init__(input_dim=input_dim, output_dim=FIRST_STAGE_OUTPUT_DIM)

    # --- Outcome Proxy Featurizer (Input: 2D) ---
    class SecondStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 2):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_AX_OUTPUT_DIM)

    # --- Treatment Proxy Featurizer (Input: 2D) ---
    class TreatmentProxyFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 3, DROPOUT_RATE: float = DROPOUT_RATE):
            super().__init__(input_dim=input_dim, output_dim=Z_OUTPUT_DIM)

    def _small_init(m):
        if isinstance(m, nn.Linear):
            # We use a very small scale (0.01) to squash the initial output variance
            nn.init.kaiming_normal_(m.weight, a=0.05) # Use GELU/ReLU compatible init
            m.weight.data *= 0.05 
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 0.05) # Scale down LayerNorm gains too
    # Instantiate the models
    first_stage_featurizer = FirstStageFeaturizer().to(device)
    second_stage_ax_featurizer = SecondStageFeaturizer().to(device)
    treatment_proxy_featurizer = TreatmentProxyFeaturizer().to(device)
    treatment_proxy_featurizer.apply(_small_init)
    
    return first_stage_featurizer, second_stage_ax_featurizer, treatment_proxy_featurizer