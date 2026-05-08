import torch
import torch.nn as nn   
from typing import Tuple

def build_nets_for_outcome_pcl_net_synthetic_low_dim_experiment(device: str = "cuda") -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module]:
    """
    Builds efficient featurizer networks for the low-dimensional synthetic benchmark.
    Significantly reduces final output dimensions (from 64/128 to 16/32) to speed up 
    closed-form matrix operations (reducing the final layer size from 8192 to 512).
    """
    FIRST_STAGE_OUTPUT_DIM = 128
    W_OUTPUT_DIM = 16
    SECOND_STAGE_A_OUTPUT_DIM = 8
    INTERNAL_WIDTH = 128 # Keep internal layers reasonably wide
    DROPOUT_RATE = 0.05
    class FeaturizerBase(nn.Module):
        """Standard featurizer structure with GELU activation."""
        def __init__(self, input_dim: int, output_dim: int, final_activation: str = "relu"):
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
            )
            # self.outbatchnorm = nn.BatchNorm1d(output_dim)
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
        def __init__(self, input_dim: int = 3, final_activation = "gelu"):
            super().__init__(input_dim=input_dim, output_dim=FIRST_STAGE_OUTPUT_DIM, final_activation=final_activation)

    # --- Outcome Proxy Featurizer (Input: 2D) ---
    class OutcomeProxyFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 2, final_activation = "gelu"):
            super().__init__(input_dim=input_dim, output_dim=W_OUTPUT_DIM, final_activation=final_activation)

    # --- Treatment Proxy Featurizer (Input: 2D) ---
    class TreatmentFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 1, final_activation = "gelu"):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_A_OUTPUT_DIM, final_activation=final_activation)

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
    outcome_proxy_featurizer = OutcomeProxyFeaturizer().to(device)
    treatment_featurizer = TreatmentFeaturizer().to(device)
    outcome_proxy_featurizer.apply(_small_init)
    return first_stage_featurizer, treatment_featurizer, outcome_proxy_featurizer

def build_nets_for_outcome_pcl_net_synthetic_high_dim_new_version_experimentV1(device: str = "cuda") -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module]:

    class GaussianNoise(nn.Module):
        def __init__(self, std=0.05):
            super().__init__()
            self.std = std
        def forward(self, x):
            if self.training and self.std > 0:
                return x + torch.randn_like(x) * self.std
            return x

    # --- DIMENSIONS ---
    # Reduced slightly to force compression
    FIRST_STAGE_OUTPUT_DIM = 256 
    W_OUTPUT_DIM = 8
    SECOND_STAGE_A_OUTPUT_DIM = 8
    SECOND_STAGE_X_OUTPUT_DIM = 32 

    INTERNAL_WIDTH = 256 
    
    DROPOUT_RATE = 0.05 

    class FeaturizerBase(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, final_activation: str = "linear"):
            super().__init__()
            self.net = nn.Sequential(
                # Stronger Input Noise for small N
                # GaussianNoise(std=0.0001), 
                
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

                # nn.Linear(INTERNAL_WIDTH // 2, INTERNAL_WIDTH // 2), 
                # nn.LayerNorm(INTERNAL_WIDTH // 2),
                # nn.ReLU(),
                # nn.Dropout(DROPOUT_RATE),
                
                nn.Linear(INTERNAL_WIDTH // 2, output_dim),
                nn.LayerNorm(output_dim),
                # Activation logic handled below
            )
            self.outbatchnorm = nn.BatchNorm1d(output_dim)
            self.final_activation = final_activation
            self.tanh = nn.Tanh()
            self.gelu = nn.GELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.net(x)
            if self.final_activation == "tanh":
                return (self.tanh(out) )
            elif self.final_activation == "gelu":
                return (self.gelu(out))
            return (out)
    
    class OutcomeProxyFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 10):
            super().__init__(input_dim=input_dim, output_dim=W_OUTPUT_DIM, final_activation="linear")

    # Keep others flexible but regularized
    class FirstStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 111):
            super().__init__(input_dim=input_dim, output_dim=FIRST_STAGE_OUTPUT_DIM, final_activation="gelu")

    class TreatmentFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 1):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_A_OUTPUT_DIM, final_activation="gelu")

    class BackdoorFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 100):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_X_OUTPUT_DIM, final_activation="gelu")
            
    # Instantiate the models
    first_stage_featurizer = FirstStageFeaturizer().to(device)
    treatment_featurizer = TreatmentFeaturizer().to(device)
    backdoor_featurizer = BackdoorFeaturizer().to(device)
    outcome_proxy_featurizer = OutcomeProxyFeaturizer().to(device)

    return first_stage_featurizer, treatment_featurizer, backdoor_featurizer, outcome_proxy_featurizer


def build_nets_for_outcome_pcl_net_synthetic_high_dim_new_version_experiment(device: str = "cuda") -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module]:
    
    class GaussianNoise(nn.Module):
        def __init__(self, std=0.05):
            super().__init__()
            self.std = std
        def forward(self, x):
            if self.training and self.std > 0:
                return x + torch.randn_like(x) * self.std
            return x

    # --- DIMENSIONS ---
    # Reduced slightly to force compression
    FIRST_STAGE_OUTPUT_DIM = 256 
    W_OUTPUT_DIM = 8
    SECOND_STAGE_A_OUTPUT_DIM = 8
    SECOND_STAGE_X_OUTPUT_DIM = 32 

    INTERNAL_WIDTH = 256 
    
    DROPOUT_RATE = 0.05

    class FeaturizerBase(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, final_activation: str = "linear"):
            super().__init__()
            self.net = nn.Sequential(
                # Stronger Input Noise for small N
                # GaussianNoise(std=0.0001), 
                
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

                # nn.Linear(INTERNAL_WIDTH * 2, INTERNAL_WIDTH * 2), 
                # nn.LayerNorm(INTERNAL_WIDTH * 2),
                # nn.ReLU(),
                # nn.Dropout(DROPOUT_RATE),
                
                nn.Linear(INTERNAL_WIDTH * 2, output_dim),
                nn.LayerNorm(output_dim),
                # Activation logic handled below
            )
            self.outbatchnorm = nn.BatchNorm1d(output_dim)
            self.final_activation = final_activation
            self.tanh = nn.Tanh()
            self.gelu = nn.GELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.net(x)
            if self.final_activation == "tanh":
                return (self.tanh(out) )
            elif self.final_activation == "gelu":
                # return self.outbatchnorm(self.gelu(out))
                return self.gelu(out)
            return (out)
    
    class OutcomeProxyFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 10):
            super().__init__(input_dim=input_dim, output_dim=W_OUTPUT_DIM, final_activation="linear")

    # Keep others flexible but regularized
    class FirstStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 111):
            super().__init__(input_dim=input_dim, output_dim=FIRST_STAGE_OUTPUT_DIM, final_activation="gelu")

    class TreatmentFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 1):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_A_OUTPUT_DIM, final_activation="gelu")

    class BackdoorFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 100):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_X_OUTPUT_DIM, final_activation="gelu")
            
    # 2. TARGETED SMALL INITIALIZATION
    # This ensures phi_W(w) starts very close to zero for stability
    def _small_init(m):
        if isinstance(m, nn.Linear):
            # We use a very small scale (0.01) to squash the initial output variance
            nn.init.kaiming_normal_(m.weight, a=0.05) # Use GELU/ReLU compatible init
            m.weight.data *= 0.1 
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 0.05) # Scale down LayerNorm gains too

    # Instantiate the models
    first_stage_featurizer = FirstStageFeaturizer().to(device)
    treatment_featurizer = TreatmentFeaturizer().to(device)
    backdoor_featurizer = BackdoorFeaturizer().to(device)
    outcome_proxy_featurizer = OutcomeProxyFeaturizer().to(device)
    # outcome_proxy_featurizer.apply(_small_init)

    return first_stage_featurizer, treatment_featurizer, backdoor_featurizer, outcome_proxy_featurizer


def build_nets_for_outcome_pcl_net_synthetic_high_dim_experiment(device: str = "cuda") -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module]:
    
    class GaussianNoise(nn.Module):
        def __init__(self, std=0.05):
            super().__init__()
            self.std = std
        def forward(self, x):
            if self.training and self.std > 0:
                return x + torch.randn_like(x) * self.std
            return x

    # --- DIMENSIONS ---
    # Reduced slightly to force compression
    FIRST_STAGE_OUTPUT_DIM = 256 
    W_OUTPUT_DIM = 8
    SECOND_STAGE_A_OUTPUT_DIM = 8
    SECOND_STAGE_X_OUTPUT_DIM = 32 

    INTERNAL_WIDTH = 256 
    
    DROPOUT_RATE = 0.1 

    class FeaturizerBase(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, final_activation: str = "linear"):
            super().__init__()
            self.net = nn.Sequential(
                # Stronger Input Noise for small N
                # GaussianNoise(std=0.0001), 
                
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

                # nn.Linear(INTERNAL_WIDTH * 2, INTERNAL_WIDTH * 2), 
                # nn.LayerNorm(INTERNAL_WIDTH * 2),
                # nn.ReLU(),
                # nn.Dropout(DROPOUT_RATE),
                
                nn.Linear(INTERNAL_WIDTH * 2, output_dim),
                nn.LayerNorm(output_dim),
                # Activation logic handled below
            )
            self.outbatchnorm = nn.BatchNorm1d(output_dim)
            self.final_activation = final_activation
            self.tanh = nn.Tanh()
            self.gelu = nn.GELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.net(x)
            if self.final_activation == "tanh":
                return (self.tanh(out) )
            elif self.final_activation == "gelu":
                return self.outbatchnorm(self.gelu(out))
            return (out)
    
    class OutcomeProxyFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 10):
            super().__init__(input_dim=input_dim, output_dim=W_OUTPUT_DIM, final_activation="linear")

    # Keep others flexible but regularized
    class FirstStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 111):
            super().__init__(input_dim=input_dim, output_dim=FIRST_STAGE_OUTPUT_DIM, final_activation="gelu")

    class TreatmentFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 1):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_A_OUTPUT_DIM, final_activation="gelu")

    class BackdoorFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 100):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_X_OUTPUT_DIM, final_activation="gelu")
            
    # 2. TARGETED SMALL INITIALIZATION
    # This ensures phi_W(w) starts very close to zero for stability
    def _small_init(m):
        if isinstance(m, nn.Linear):
            # We use a very small scale (0.01) to squash the initial output variance
            nn.init.kaiming_normal_(m.weight, a=0.05) # Use GELU/ReLU compatible init
            m.weight.data *= 0.1 
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 0.05) # Scale down LayerNorm gains too

    # Instantiate the models
    first_stage_featurizer = FirstStageFeaturizer().to(device)
    treatment_featurizer = TreatmentFeaturizer().to(device)
    backdoor_featurizer = BackdoorFeaturizer().to(device)
    outcome_proxy_featurizer = OutcomeProxyFeaturizer().to(device)
    # outcome_proxy_featurizer.apply(_small_init)

    return first_stage_featurizer, treatment_featurizer, backdoor_featurizer, outcome_proxy_featurizer


def build_nets_for_outcome_pcl_net_abortion_and_crime_experiment(device: str = "cuda") -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module]:
    
    class GaussianNoise(nn.Module):
        def __init__(self, std=0.05):
            super().__init__()
            self.std = std
        def forward(self, x):
            if self.training and self.std > 0:
                return x + torch.randn_like(x) * self.std
            return x

    FIRST_STAGE_OUTPUT_DIM = 64 
    W_OUTPUT_DIM = 16
    SECOND_STAGE_A_OUTPUT_DIM = 8

    INTERNAL_WIDTH = 256 
    
    DROPOUT_RATE = 0.01 

    class FeaturizerBase(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, final_activation: str = "linear"):
            super().__init__()
            self.net = nn.Sequential(
                # Stronger Input Noise for small N
                GaussianNoise(std=0.01), 
                
                nn.Linear(input_dim, INTERNAL_WIDTH),
                nn.LayerNorm(INTERNAL_WIDTH),
                nn.GELU(), 
                # nn.BatchNorm1d(INTERNAL_WIDTH),
                nn.Dropout(DROPOUT_RATE),
                
                nn.Linear(INTERNAL_WIDTH, INTERNAL_WIDTH // 2), 
                nn.LayerNorm(INTERNAL_WIDTH // 2),
                nn.GELU(),
                # nn.BatchNorm1d(INTERNAL_WIDTH // 2),
                nn.Dropout(DROPOUT_RATE),
                
                nn.Linear(INTERNAL_WIDTH // 2, output_dim),
                # Activation logic handled below
            )
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
    
    class OutcomeProxyFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 3):
            super().__init__(input_dim=input_dim, output_dim=W_OUTPUT_DIM, final_activation="gelu")

    # Keep others flexible but regularized
    class FirstStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 2):
            super().__init__(input_dim=input_dim, output_dim=FIRST_STAGE_OUTPUT_DIM, final_activation="gelu")

    class TreatmentFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 1):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_A_OUTPUT_DIM, final_activation="gelu")

            
    # Instantiate the models
    first_stage_featurizer = FirstStageFeaturizer().to(device)
    treatment_featurizer = TreatmentFeaturizer().to(device)
    outcome_proxy_featurizer = OutcomeProxyFeaturizer().to(device)

    return first_stage_featurizer, treatment_featurizer, outcome_proxy_featurizer


def build_nets_for_outcome_pcl_net_dsprite_experiment(device: str = "cuda") -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module, nn.Module, nn.Module]:
    """
    Builds the DFPCL featurizer networks based on Table 3 for the dSprite high-dimensional 
    image experiment. These networks include Spectral Normalization (SN) and 
    Batch Normalization (BN) for stability with high-dimensional input.
    
    Inputs: A (4096D), Z (3D), W (4096D).
    
    Returns: (phi_A1, psi_A2, psi_W, phi_Z)
    """

    # --- INPUT DIMENSIONS ---
    DIM_IMAGE = 4096 
    DIM_Z = 3 
    
    # --- Output/Internal Dimensions ---    
    FIRST_STAGE_OUTPUT_DIM = 128
    SECOND_STAGE_A_OUTPUT_DIM = 16
    OUT_DIM_W = 16
    
    # 1. Stage 1 Treatment Feature (phi_A1)
    class FirstStageFeaturizer(nn.Module):
        def __init__(self, input_dim: int = DIM_IMAGE + DIM_Z, output_dim: int = FIRST_STAGE_OUTPUT_DIM):
            super().__init__()
            self.net = nn.Sequential(
                # FC(4096, 1024), SN, ReLU
                nn.Linear(input_dim, 1024),
                nn.LayerNorm(1024),
                nn.ReLU(),
                # FC(1024, 512), SN, ReLU, BN
                nn.Linear(1024, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                # nn.BatchNorm1d(512), 
                # FC(512, 128), SN, ReLU
                nn.Linear(512, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                # FC(128, 32), SN, ReLU
                nn.Linear(128, output_dim), # Output dim is 32
                nn.LayerNorm(output_dim),
                nn.ReLU(),
            )
            self.output_dim = output_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    # 3. Outcome Proxy Feature (psi_W)
    class OutcomeProxyFeaturizer(nn.Module):
        def __init__(self, input_dim: int = DIM_IMAGE, output_dim: int = OUT_DIM_W):
            super().__init__()
            self.net = nn.Sequential(
                # FC(4096, 1024), SN, ReLU
                nn.Linear(input_dim, 1024),
                nn.LayerNorm(1024),
                nn.ReLU(),
                # FC(1024, 512), SN, ReLU, BN
                nn.Linear(1024, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                # FC(512, 128), SN, ReLU
                nn.Linear(512, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                # FC(128, 32), SN, ReLU
                nn.Linear(128, output_dim), # Output dim is 32
                nn.LayerNorm(output_dim),
                nn.ReLU(),
            )
            self.output_dim = output_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class TreatmentFeaturizer(nn.Module):
        def __init__(self, input_dim: int = DIM_IMAGE, output_dim: int = SECOND_STAGE_A_OUTPUT_DIM):
            super().__init__()
            self.net = nn.Sequential(
                # FC(4096, 1024), SN, ReLU
                nn.Linear(input_dim, 1024),
                nn.LayerNorm(1024),
                nn.ReLU(),
                # FC(1024, 512), SN, ReLU, BN
                nn.Linear(1024, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.BatchNorm1d(512), 
                # FC(512, 128), SN, ReLU
                nn.Linear(512, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                # FC(128, 32), SN, ReLU
                nn.Linear(128, output_dim), # Output dim is 32
                nn.LayerNorm(output_dim),
                nn.ReLU(),
            )
            self.output_dim = output_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    # 2. TARGETED SMALL INITIALIZATION
    # This ensures phi_W(w) starts very close to zero for stability
    def _small_init(m):
        if isinstance(m, nn.Linear):
            # We use a very small scale (0.01) to squash the initial output variance
            nn.init.kaiming_normal_(m.weight, a=0.2) # Use GELU/ReLU compatible init
            m.weight.data *= 0.2 
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 0.2) # Scale down LayerNorm gains too
            

    # Instantiate the models
    first_stage_featurizer = FirstStageFeaturizer().to(device)
    outcome_proxy_featurizer = OutcomeProxyFeaturizer().to(device)
    treatment_featurizer = TreatmentFeaturizer().to(device)
    # outcome_proxy_featurizer.apply(_small_init)
    return first_stage_featurizer, treatment_featurizer, outcome_proxy_featurizer


def build_nets_for_outcome_pcl_net_demand_experiment(device: str = "cuda") -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module]:

    FIRST_STAGE_OUTPUT_DIM = 32
    W_OUTPUT_DIM = 16
    SECOND_STAGE_A_OUTPUT_DIM = 16
    INTERNAL_WIDTH = 32 # Keep internal layers reasonably wide
    DROPOUT_RATE = 0.01

    class FeaturizerBase(nn.Module):
        """Standard featurizer structure with GELU activation."""
        def __init__(self, input_dim: int, output_dim: int, final_activation: str = "relu"):
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
            )
            # self.outbatchnorm = nn.BatchNorm1d(output_dim)
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
        def __init__(self, input_dim: int = 3, final_activation = "gelu"):
            super().__init__(input_dim=input_dim, output_dim=FIRST_STAGE_OUTPUT_DIM, final_activation=final_activation)

    # --- Outcome Proxy Featurizer (Input: 2D) ---
    class OutcomeProxyFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 1, final_activation = "gelu"):
            super().__init__(input_dim=input_dim, output_dim=W_OUTPUT_DIM, final_activation=final_activation)

    # --- Treatment Proxy Featurizer (Input: 2D) ---
    class TreatmentFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 1, final_activation = "gelu"):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_A_OUTPUT_DIM, final_activation=final_activation)


    # Instantiate the models
    first_stage_featurizer = FirstStageFeaturizer().to(device)
    outcome_proxy_featurizer = OutcomeProxyFeaturizer().to(device)
    treatment_featurizer = TreatmentFeaturizer().to(device)

    return first_stage_featurizer, treatment_featurizer, outcome_proxy_featurizer


def build_nets_for_dfpcl_dsprite_experiment_with_compressed_images(latent_dim: int = 8, device: str = "cuda") -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module, nn.Module, nn.Module]:
    """
    Builds the DFPCL featurizer networks based on Table 3 for the dSprite high-dimensional 
    image experiment. These networks include Spectral Normalization (SN) and 
    Batch Normalization (BN) for stability with high-dimensional input.
    
    Inputs: A (4096D), Z (3D), W (4096D).
    
    Returns: (phi_A1, psi_A2, psi_W, phi_Z)
    """

    # --- INPUT DIMENSIONS ---
    DIM_IMAGE = latent_dim 
    DIM_Z = 3 
    
    # --- Output/Internal Dimensions ---    
    FIRST_STAGE_OUTPUT_DIM = 128
    SECOND_STAGE_A_OUTPUT_DIM = 16
    OUT_DIM_W = 16
    
    # 1. Stage 1 Treatment Feature (phi_A1)
    class FirstStageFeaturizer(nn.Module):
        def __init__(self, input_dim: int = DIM_IMAGE + DIM_Z, output_dim: int = FIRST_STAGE_OUTPUT_DIM):
            super().__init__()
            self.net = nn.Sequential(
                # FC(4096, 1024), SN, ReLU
                nn.Linear(input_dim, 1024),
                nn.LayerNorm(1024),
                nn.ReLU(),
                # FC(1024, 512), SN, ReLU, BN
                nn.Linear(1024, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                # nn.BatchNorm1d(512), 
                # FC(512, 128), SN, ReLU
                nn.Linear(512, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                # FC(128, 32), SN, ReLU
                nn.Linear(128, output_dim), # Output dim is 32
                nn.LayerNorm(output_dim),
                nn.ReLU(),
            )
            self.output_dim = output_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    # 3. Outcome Proxy Feature (psi_W)
    class OutcomeProxyFeaturizer(nn.Module):
        def __init__(self, input_dim: int = DIM_IMAGE, output_dim: int = OUT_DIM_W):
            super().__init__()
            self.net = nn.Sequential(
                # FC(4096, 1024), SN, ReLU
                nn.Linear(input_dim, 1024),
                nn.LayerNorm(1024),
                nn.ReLU(),
                # FC(1024, 512), SN, ReLU, BN
                nn.Linear(1024, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                # FC(512, 128), SN, ReLU
                nn.Linear(512, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                # FC(128, 32), SN, ReLU
                nn.Linear(128, output_dim), # Output dim is 32
                nn.LayerNorm(output_dim),
                nn.ReLU(),
            )
            self.output_dim = output_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class TreatmentFeaturizer(nn.Module):
        def __init__(self, input_dim: int = DIM_IMAGE, output_dim: int = SECOND_STAGE_A_OUTPUT_DIM):
            super().__init__()
            self.net = nn.Sequential(
                # FC(4096, 1024), SN, ReLU
                nn.Linear(input_dim, 1024),
                nn.LayerNorm(1024),
                nn.ReLU(),
                # FC(1024, 512), SN, ReLU, BN
                nn.Linear(1024, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.BatchNorm1d(512), 
                # FC(512, 128), SN, ReLU
                nn.Linear(512, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                # FC(128, 32), SN, ReLU
                nn.Linear(128, output_dim), # Output dim is 32
                nn.LayerNorm(output_dim),
                nn.ReLU(),
            )
            self.output_dim = output_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


    # Instantiate the models
    first_stage_featurizer = FirstStageFeaturizer().to(device)
    outcome_proxy_featurizer = OutcomeProxyFeaturizer().to(device)
    treatment_featurizer = TreatmentFeaturizer().to(device)

    return first_stage_featurizer, treatment_featurizer, outcome_proxy_featurizer


def build_nets_for_outcome_pcl_net_synthetic_cate_experiment(device: str = "cuda") -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module]:
    FIRST_STAGE_OUTPUT_DIM = 128
    W_OUTPUT_DIM = 16
    SECOND_STAGE_A_OUTPUT_DIM = 4
    SECOND_STAGE_V_OUTPUT_DIM = 8
    INTERNAL_WIDTH = 64 # Keep internal layers reasonably wide
    DROPOUT_RATE = 0.01
    class FeaturizerBase(nn.Module):
        """Standard featurizer structure with GELU activation."""
        def __init__(self, input_dim: int, output_dim: int, final_activation: str = "relu"):
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
            )
            # self.outbatchnorm = nn.BatchNorm1d(output_dim)
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
        def __init__(self, input_dim: int = 5, final_activation = "gelu"):
            super().__init__(input_dim=input_dim, output_dim=FIRST_STAGE_OUTPUT_DIM, final_activation=final_activation)

    # --- Outcome Proxy Featurizer (Input: 2D) ---
    class OutcomeProxyFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 3, final_activation = "gelu"):
            super().__init__(input_dim=input_dim, output_dim=W_OUTPUT_DIM, final_activation=final_activation)

    # --- Treatment Proxy Featurizer (Input: 2D) ---
    class TreatmentFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 1, final_activation = "gelu"):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_A_OUTPUT_DIM, final_activation=final_activation)
            
    class CovariateFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 1, final_activation = "gelu"):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_V_OUTPUT_DIM, final_activation=final_activation)

    def _small_init(m):
        if isinstance(m, nn.Linear):
            # We use a very small scale (0.01) to squash the initial output variance
            nn.init.kaiming_normal_(m.weight, a=0.2) # Use GELU/ReLU compatible init
            m.weight.data *= 0.2
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 0.2) # Scale down LayerNorm gains too
            
    # Instantiate the models
    first_stage_featurizer = FirstStageFeaturizer().to(device)
    outcome_proxy_featurizer = OutcomeProxyFeaturizer().to(device)
    treatment_featurizer = TreatmentFeaturizer().to(device)
    covariate_featurizer = CovariateFeaturizer().to(device)
    outcome_proxy_featurizer.apply(_small_init)
    return first_stage_featurizer, treatment_featurizer, covariate_featurizer, outcome_proxy_featurizer



