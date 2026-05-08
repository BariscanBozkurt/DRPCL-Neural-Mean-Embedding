import os
import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy.sparse import diags
from scipy.stats import norm
from scipy.spatial.distance import cdist
from itertools import product
import torch.utils.data as data
from dataclasses import dataclass
#################################################################################################
############## PROXY CAUSAL LEARNING DATA GENERATION FUNCTIONS ##################################
#################################################################################################


class PCL_Synthetic_High_Dim:
    """
    This data generation process is taken from the official code of the following paper (from Openreview):
    Doubly Robust Proximal Causal Learning for Continuous Treatments, Yong Wu, Yanwei Fu, Shouyan Wang, Xinwei Sun 
    """
    def __init__(self ,seeds = 43, size = 1000,dim_z = 1, dim_w = 3, dim_x = 10, type_ = 'quadratic'):
        self.seeds = seeds
        self.size = size
        self.dim_z = dim_z
        self.dim_w = dim_w
        self.dim_x = dim_x
        self.type_ = type_

    def Lambda(self, t):
        return (0.9 - 0.1) * np.exp(t) / (1 + np.exp(t)) + 0.1

    def generatate_high(self, totensor: bool = False,):
        np.random.seed(self.seeds)

        e1 = np.random.normal(0,1,self.size)
        e2 = np.random.normal(0,1,self.size)
        e3 = np.random.normal(0,1,self.size)
        vz = np.random.uniform(-1, 1, size= (self.size, self.dim_z))
        vw = np.random.uniform(-1, 1, size= (self.size, self.dim_w))

       

        Uz = e1 + e3
        Uw = e2 + e3

        Z = vz + 0.25* np.repeat(Uz.reshape(-1, 1), self.dim_z, axis=1)
        W = vw + 0.25* np.repeat(Uw.reshape(-1, 1), self.dim_w, axis=1)


        rho = 0.5
        k = [rho*np.ones(self.dim_x-1),np.ones(self.dim_x),rho*np.ones(self.dim_x-1)]
        offset = [-1,0,1]
        sigma = diags(k,offset).toarray()
        X = np.random.multivariate_normal(np.zeros(self.dim_x),sigma,size=[self.size,])

        theta_x = np.array([(1/(l**2)) for l in list(range(1,(self.dim_x+1)))])
        theta_w = np.array([(1/(l**2)) for l in list(range(1,(self.dim_w+1)))])
        theta_z = np.array([(1/(l**2)) for l in list(range(1,(self.dim_z+1)))])

        A = self.Lambda(3*X@theta_x + 3*Z@theta_z) + 0.25*Uw

        if self.type_ == 'quadratic':
            structure = 1.2*A + (A**2) 
            Y = structure + 1.2*(X@theta_x + W@theta_w) + A*X[:,0] + 0.25*Uz
        elif self.type_ == 'peaked':
            structure = 2*(A**4/600 + np.exp(-4*A**2) + A/10 -2) + 1.2*A
            Y = structure + 1.2*(X@theta_x + W@theta_w) + A*X[:,0] + 0.25*Uz
        elif self.type_ == 'sigmoid':
            structure = np.log(abs(16*A-8)+1)*np.sign(A-0.5) + 1.2*A
            Y = structure + 1.2*(X@theta_x + W@theta_w) + A*X[:,0] + 0.25*Uz
        
        return A[:, np.newaxis], Z, W, Y[:, np.newaxis], X 
        
    def generate_test(self,size,seed=43,totensor=False) -> None:
        np.random.seed(seed)
        e1 = np.random.normal(0,1,size)
        e2 = np.random.normal(0,1,size)
        e3 = np.random.normal(0,1,size)
        vz = np.random.uniform(-1, 1, size= (size, self.dim_z))
        vw = np.random.uniform(-1, 1, size= (size, self.dim_w))

        Uz = e1 + e3
        Uw = e2 + e3

        Z = vz + 0.25* np.repeat(Uz.reshape(-1, 1), self.dim_z, axis=1)
        W = vw + 0.25* np.repeat(Uw.reshape(-1, 1), self.dim_w, axis=1)

        rho =0.5 
        k = [rho*np.ones(self.dim_x-1),np.ones(self.dim_x),rho*np.ones(self.dim_x-1)]
        offset = [-1,0,1]
        sigma = diags(k,offset).toarray()
        X = np.random.multivariate_normal(np.zeros(self.dim_x),sigma,size=[size,])

        theta_x = np.array([(1/(l**2)) for l in list(range(1,(self.dim_x+1)))])
        theta_w = np.array([(1/(l**2)) for l in list(range(1,(self.dim_w+1)))])
        theta_z = np.array([(1/(l**2)) for l in list(range(1,(self.dim_z+1)))])

        A = self.Lambda(3*X@theta_x + 3*Z@theta_z) + 0.25*Uw

        if self.type_ == 'quadratic':
            structure = 1.2*A + (A**2) 

            Y = structure + 1.2*(X@theta_x + W@theta_w) + A*X[:,0] + 0.25*Uz
        elif self.type_ == 'peaked':
            structure = 2*(A**4/600 + np.exp(-4*A**2) + A/10 -2) + 1.2*A
            Y = structure + 1.2*(X@theta_x + W@theta_w) + A*X[:,0] + 0.25*Uz
        elif self.type_ == 'sigmoid':
            structure = np.log(abs(16*A-8)+1)*np.sign(A-0.5) + 1.2*A
            Y = structure + 1.2*(X@theta_x + W@theta_w) + A*X[:,0] + 0.25*Uz

        return A[:, np.newaxis], Z, W, Y[:, np.newaxis], X

        
    @staticmethod
    def generate_test_effect(a,b,c,type_,dim_z,dim_w,dim_x):
        A = np.linspace(a, b, c)
        e1 = np.random.normal(0,1,10000)
        e2 = np.random.normal(0,1,10000)
        e3 = np.random.normal(0,1,10000)
        vz = np.random.uniform(-1, 1, size= (10000,dim_z))
        vw = np.random.uniform(-1, 1, size= (10000,dim_w) )

        Uz = e1 + e3
        Uw = e2 + e3

        Z = vz + 0.25* np.repeat(Uz.reshape(-1, 1), dim_z, axis=1)
        W = vw + 0.25* np.repeat(Uw.reshape(-1, 1), dim_w, axis=1)

        rho = 0.5, 
        k = [rho*np.ones(dim_x-1),np.ones(dim_x),rho*np.ones(dim_x-1)]
        offset = [-1,0,1]
        sigma = diags(k,offset).toarray()
        X = np.random.multivariate_normal(np.zeros(dim_x),sigma,size=[10000,])

        theta_x = np.array([(1/(l**2)) for l in list(range(1,(dim_x+1)))])
        theta_w = np.array([(1/(l**2)) for l in list(range(1,(dim_w+1)))])
        theta_z = np.array([(1/(l**2)) for l in list(range(1,(dim_z+1)))])

        if type_ == 'quadratic':
            treatment = np.array([np.mean(1.2*a + (a**2) + 1.2*(X@theta_x + W@theta_w) + a*X[:,0] + 0.25*Uz) for a in A])
        elif type_ == 'peaked':
            treatment = np.array([np.mean(2*(a**4/600 + np.exp(-4*a**2) + a/10 -2) + 1.2*a + 1.2*(X@theta_x + W@theta_w) + a*X[:,0] + 0.25*Uz) for a in A])
        elif type_ == 'sigmoid':
            treatment = np.array([np.mean(np.log(abs(16*a-8)+1)*np.sign(a-0.5) + 1.2*a+ 1.2*(X@theta_x + W@theta_w) + a*X[:,0] + 0.25*Uz) for a in A])
        A = A.reshape(-1, 1)
        treatment = treatment.reshape(-1, 1)
        return A,treatment


class PCL_Synthetic_High_DimNew:
    """
    Improved Synthetic Data Generator.
    
    Changes from original:
    1. Replaced Quadratic Decay (1/j^2) with Dense Normalized Weights (1/sqrt(d)).
       - Old: Only first ~5 features mattered.
       - New: All 100 features matter equally.
    2. Fixed method name typos.
    """
    def __init__(self, seeds=43, size=1000, dim_z=10, dim_w=10, dim_x=100, type_='quadratic'):
        self.seeds = seeds
        self.size = size
        self.dim_z = dim_z
        self.dim_w = dim_w
        self.dim_x = dim_x
        self.type_ = type_

    def Lambda(self, t):
        return (0.9 - 0.1) * np.exp(t) / (1 + np.exp(t)) + 0.1

    def _get_dense_weights(self, dim):
        """
        Returns weights where every dimension matters equally.
        Scaled by 1/sqrt(dim) to keep dot-product variance = 1.0.
        """
        return np.ones(dim) / np.sqrt(dim)

    def generatate_high(self, totensor: bool = False):
        np.random.seed(self.seeds)

        # 1. Latent Variables
        e1 = np.random.normal(0, 1, self.size)
        e2 = np.random.normal(0, 1, self.size)
        e3 = np.random.normal(0, 1, self.size)
        
        # Unobserved Confounders
        Uz = e1 + e3
        Uw = e2 + e3
        
        # 2. Proxies (Z, W)
        # Noise
        vz = np.random.uniform(-1, 1, size=(self.size, self.dim_z))
        vw = np.random.uniform(-1, 1, size=(self.size, self.dim_w))
        
        # Signal injection (Broadcasting Uz across all Z dimensions)
        Z = vz + 0.25 * np.repeat(Uz.reshape(-1, 1), self.dim_z, axis=1)
        W = vw + 0.25 * np.repeat(Uw.reshape(-1, 1), self.dim_w, axis=1)

        # 3. High-Dim Covariates (X) with Correlation
        rho = 0.5
        k = [rho * np.ones(self.dim_x - 1), np.ones(self.dim_x), rho * np.ones(self.dim_x - 1)]
        offset = [-1, 0, 1]
        sigma = diags(k, offset).toarray()
        X = np.random.multivariate_normal(np.zeros(self.dim_x), sigma, size=[self.size, ])

        # 4. Coefficients (The Fix!)
        # Use dense weights instead of decaying ones
        theta_x = self._get_dense_weights(self.dim_x)
        theta_w = self._get_dense_weights(self.dim_w)
        theta_z = self._get_dense_weights(self.dim_z)

        # 5. Treatment Assignment
        # Now depends on ALL dimensions of X and Z
        A = self.Lambda(3 * X @ theta_x + 3 * Z @ theta_z) + 0.25 * Uw

        # 6. Outcome Generation
        if self.type_ == 'quadratic':
            structure = 1.2 * A + (A**2)
            # Bias depends on ALL X and W
            Y = structure + 1.2 * (X @ theta_x + W @ theta_w) + A * X[:, 0] + 0.25 * Uz
            
        elif self.type_ == 'peaked':
            structure = 2 * (A**4 / 600 + np.exp(-4 * A**2) + A / 10 - 2) + 1.2 * A
            Y = structure + 1.2 * (X @ theta_x + W @ theta_w) + A * X[:, 0] + 0.25 * Uz
            
        elif self.type_ == 'sigmoid':
            structure = np.log(np.abs(16 * A - 8) + 1) * np.sign(A - 0.5) + 1.2 * A
            Y = structure + 1.2 * (X @ theta_x + W @ theta_w) + A * X[:, 0] + 0.25 * Uz

        return A[:, np.newaxis], Z, W, Y[:, np.newaxis], X

    @staticmethod
    def generate_test_effect(a, b, c, type_, dim_z, dim_w, dim_x):
        """
        Generates the Ground Truth causal curve.
        Must match the logic of generate_data EXACTLY.
        """
        A_grid = np.linspace(a, b, c)
        
        # Monte Carlo Integration for Ground Truth
        n_mc = 20000 
        e1 = np.random.normal(0, 1, n_mc)
        e2 = np.random.normal(0, 1, n_mc)
        e3 = np.random.normal(0, 1, n_mc)
        
        Uz = e1 + e3
        
        # X generation
        rho = 0.5
        k = [rho * np.ones(dim_x - 1), np.ones(dim_x), rho * np.ones(dim_x - 1)]
        offset = [-1, 0, 1]
        sigma = diags(k, offset).toarray()
        X = np.random.multivariate_normal(np.zeros(dim_x), sigma, size=[n_mc, ])
        
        # W noise for confounding term in Y
        # Note: In Y equation, W is involved: 1.2 * (W @ theta_w)
        # We need to generate W to compute the expectation properly
        vw = np.random.uniform(-1, 1, size=(n_mc, dim_w))
        Uw = e2 + e3
        W = vw + 0.25 * np.repeat(Uw.reshape(-1, 1), dim_w, axis=1)

        # Dense Weights (Matching the instance method)
        theta_x = np.ones(dim_x) / np.sqrt(dim_x)
        theta_w = np.ones(dim_w) / np.sqrt(dim_w)
        
        # Precompute Confounding Baseline
        # E[Y|do(a)] = structure(a) + E[1.2(X theta_x + W theta_w) + a*X0 + 0.25Uz]
        # Since X, W, Uz are centered at 0, the linear terms average to 0.
        # However, let's compute it explicitly to be safe against random seed variance.
        confounding_baseline = 1.2 * (X @ theta_x + W @ theta_w) + 0.25 * Uz

        treatment_curve = []
        for a_val in A_grid:
            # The structural part depends only on a_val
            if type_ == 'quadratic':
                struct = 1.2 * a_val + (a_val**2)
            elif type_ == 'peaked':
                struct = 2 * (a_val**4 / 600 + np.exp(-4 * a_val**2) + a_val / 10 - 2) + 1.2 * a_val
            elif type_ == 'sigmoid':
                struct = np.log(np.abs(16 * a_val - 8) + 1) * np.sign(a_val - 0.5) + 1.2 * a_val
            
            # The interaction term a * X[:,0]
            interaction = a_val * X[:, 0]
            
            # E[Y_a]
            y_a = struct + interaction + confounding_baseline
            treatment_curve.append(np.mean(y_a))

        return A_grid.reshape(-1, 1), np.array(treatment_curve).reshape(-1, 1)

class dSprite_ProxyVariable_DatasetV2():
    ### This is based on "Update on dSprite experiment", see https://github.com/liyuan9988/DeepFeatureProxyVariable/tree/master
    def __init__(self,):
        pass
        
    def cal_weight(self,):
        weights = np.empty((64, 64))
        for i in range(64):
            for j in range(64):
                weights[i, j] = (np.abs(32 - j))
        return weights.reshape(64*64, 1) / 32


    def image_id(self, latent_bases: np.ndarray, posX_id_arr: np.ndarray, posY_id_arr: np.ndarray,
                 orientation_id_arr: np.ndarray,
                 scale_id_arr: np.ndarray):
        data_size = posX_id_arr.shape[0]
        color_id_arr = np.array([0] * data_size, dtype=int)
        shape_id_arr = np.array([2] * data_size, dtype=int)
        idx = np.c_[color_id_arr, shape_id_arr, scale_id_arr, orientation_id_arr, posX_id_arr, posY_id_arr]
        return idx.dot(latent_bases)
        
    
    def structural_func(self, image, weights):
        # return (image.dot(weights)[:, 0] ** 2 - 5000) / 1000
        return (image.dot(weights)[:, 0] ** 2 - 3000) / 500
    
    
    def generate_test_dsprite(self, data_path: str):
        dataset_zip = np.load(os.path.join(data_path, "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"),
                              allow_pickle=True, encoding="bytes")
        
        weights = self.cal_weight()
    
        imgs = dataset_zip['imgs']
        latents_values = dataset_zip['latents_values']
        metadata = dataset_zip['metadata'][()]

        latents_sizes = metadata[b'latents_sizes']
        latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                        np.array([1, ])))

        posX_id_arr = [0, 5, 10, 15, 20, 25, 30]
        posY_id_arr = [0, 5, 10, 15, 20, 25, 30]
        scale_id_arr = [0, 3, 5]
        orientation_arr = [0, 10, 20, 30]
        latent_idx_arr = []
        for posX, posY, scale, orientation in product(posX_id_arr, posY_id_arr, scale_id_arr, orientation_arr):
            latent_idx_arr.append([0, 2, scale, orientation, posX, posY])

        image_idx_arr = np.array(latent_idx_arr).dot(latents_bases)
        data_size = 7 * 7 * 3 * 4
        treatment = imgs[image_idx_arr].reshape((data_size, 64 * 64))
        structural = self.structural_func(treatment, weights)
        structural = structural[:, np.newaxis]
        return treatment, structural
    
    
    def generate_dsprite_pv(self, data_path: str, 
                            n_sample: int,
                            generate_test: bool = False,
                            rand_seed: int = 42, **kwargs):
        """
        Parameters
        ----------
        n_sample : int
            size of data
        rand_seed : int
            random seed
    
        Returns
        -------
        train_data : TrainDataSet
        """
        dataset_zip = np.load(os.path.join(data_path, "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"),
                              allow_pickle=True, encoding="bytes")
        # weights = np.load(os.path.join(data_path, "dsprite_mat.npy"))
        weights = self.cal_weight()
        imgs = dataset_zip['imgs']
        latents_values = dataset_zip['latents_values']
        metadata = dataset_zip['metadata'][()]
    
        latents_sizes = metadata[b'latents_sizes']
        latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                        np.array([1, ])))
    
        rng = default_rng(seed=rand_seed)
        posX_id_arr = rng.integers(32, size=n_sample)
        posY_id_arr = rng.integers(32, size=n_sample)
        scale_id_arr = rng.integers(6, size=n_sample)
        orientation_arr = rng.integers(40, size=n_sample)
        image_idx_arr = self.image_id(latents_bases, posX_id_arr, posY_id_arr, orientation_arr, scale_id_arr)
        treatment = imgs[image_idx_arr].reshape((n_sample, 64 * 64)).astype(np.float64)
        treatment += rng.normal(0.0, 0.1, treatment.shape)
        latent_feature = latents_values[image_idx_arr]  # (color, shape, scale, orientation, posX, posY)
        treatment_proxy = latent_feature[:, 2:5]  # (scale, orientation, posX)

        posX_id_proxy = np.array([16] * n_sample)
        scale_id_proxy = np.array([3] * n_sample)
        orientation_proxy = np.array([0] * n_sample)
        proxy_image_idx_arr = self.image_id(latents_bases, posX_id_proxy, posY_id_arr, orientation_proxy, scale_id_proxy)
        outcome_proxy = imgs[proxy_image_idx_arr].reshape((n_sample, 64 * 64)).astype(np.float64)
        outcome_proxy += rng.normal(0.0, 0.1, outcome_proxy.shape)

        structural = self.structural_func(treatment, weights)
        outcome = structural * (posY_id_arr - 15.5) ** 2 / 85.25 + rng.normal(0.0, 0.5, n_sample)
        outcome = outcome[:, np.newaxis]
        if generate_test:
            do_A, EY_do_A = self.generate_test_dsprite(data_path)
        else:
            do_A, EY_do_A = None, None
        return treatment, outcome, treatment_proxy, outcome_proxy, do_A, EY_do_A


class dSprite_ProxyVariable_DatasetV2_ATT:
    """
    Corrected Deep PCL dSprite benchmark, extended with an ATT evaluation target.
    Training DGP is unchanged relative to the corrected ATE setup.
    """

    def __init__(self):
        pass

    def cal_weight(self):
        weights = np.empty((64, 64))
        for i in range(64):
            for j in range(64):
                weights[i, j] = np.abs(32 - j)
        return weights.reshape(64 * 64, 1) / 32.0

    def image_id(
        self,
        latent_bases: np.ndarray,
        posX_id_arr: np.ndarray,
        posY_id_arr: np.ndarray,
        orientation_id_arr: np.ndarray,
        scale_id_arr: np.ndarray,
    ):
        data_size = posX_id_arr.shape[0]
        color_id_arr = np.zeros(data_size, dtype=int)
        shape_id_arr = np.full(data_size, 2, dtype=int)  # heart
        idx = np.c_[
            color_id_arr,
            shape_id_arr,
            scale_id_arr,
            orientation_id_arr,
            posX_id_arr,
            posY_id_arr,
        ]
        return idx.dot(latent_bases)

    def structural_func(self, image, weights):
        return (image.dot(weights)[:, 0] ** 2 - 3000.0) / 500.0

    def nearest_grid_id(self, x: float, n_grid: int = 32) -> int:
        return int(np.clip(np.round(x * (n_grid - 1)), 0, n_grid - 1))

    def get_anchor_ids(
        self,
        anchor_posX: float = 0.6,
        anchor_posY: float = 0.6,
        anchor_scale_id: int = 3,      # scale = 0.8
        anchor_orientation_id: int = 0 # rotation = 0
    ):
        return {
            "posX_id": self.nearest_grid_id(anchor_posX),
            "posY_id": self.nearest_grid_id(anchor_posY),
            "scale_id": anchor_scale_id,
            "orientation_id": anchor_orientation_id,
        }

    def get_clean_image(
        self,
        imgs: np.ndarray,
        latents_bases: np.ndarray,
        posX_id: int,
        posY_id: int,
        orientation_id: int,
        scale_id: int,
    ):
        idx = self.image_id(
            latents_bases,
            np.array([posX_id]),
            np.array([posY_id]),
            np.array([orientation_id]),
            np.array([scale_id]),
        )
        return imgs[idx].reshape(1, 64 * 64).astype(np.float64)

    def att_scaling_from_anchor(self, anchor_posY_id: int) -> float:
        return ((anchor_posY_id - 15.5) ** 2) / 85.25

    def generate_test_dsprite_ate(self, data_path: str):
        dataset_zip = np.load(
            os.path.join(data_path, "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"),
            allow_pickle=True,
            encoding="bytes",
        )

        weights = self.cal_weight()
        imgs = dataset_zip["imgs"]
        metadata = dataset_zip["metadata"][()]

        latents_sizes = metadata[b"latents_sizes"]
        latents_bases = np.concatenate(
            (latents_sizes[::-1].cumprod()[::-1][1:], np.array([1]))
        )

        posX_id_arr = [0, 5, 10, 15, 20, 25, 30]
        posY_id_arr = [0, 5, 10, 15, 20, 25, 30]
        scale_id_arr = [0, 3, 5]
        orientation_arr = [0, 10, 20, 30]

        latent_idx_arr = []
        for posX, posY, scale, orientation in product(
            posX_id_arr, posY_id_arr, scale_id_arr, orientation_arr
        ):
            latent_idx_arr.append([0, 2, scale, orientation, posX, posY])

        image_idx_arr = np.array(latent_idx_arr).dot(latents_bases)
        data_size = 7 * 7 * 3 * 4

        treatment = imgs[image_idx_arr].reshape((data_size, 64 * 64)).astype(np.float64)
        structural = self.structural_func(treatment, weights)[:, np.newaxis]
        return treatment, structural

    def generate_test_dsprite_att(
        self,
        data_path: str,
        anchor_posX: float = 0.6,
        anchor_posY: float = 0.6,
        anchor_scale_id: int = 3,
        anchor_orientation_id: int = 0,
    ):
        """
        Returns:
            do_A: test treatments a on the standard 588-point grid
            EY_att: oracle ATT curve f_ATT(a; a')
            A_anchor: fixed anchor treatment a'
            anchor_info: dict with the anchor latent IDs
        """
        dataset_zip = np.load(
            os.path.join(data_path, "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"),
            allow_pickle=True,
            encoding="bytes",
        )

        imgs = dataset_zip["imgs"]
        metadata = dataset_zip["metadata"][()]
        latents_sizes = metadata[b"latents_sizes"]
        latents_bases = np.concatenate(
            (latents_sizes[::-1].cumprod()[::-1][1:], np.array([1]))
        )

        weights = self.cal_weight()

        # Standard test grid for a
        do_A, structural = self.generate_test_dsprite_ate(data_path)

        # Fixed anchor a'
        anchor = self.get_anchor_ids(
            anchor_posX=anchor_posX,
            anchor_posY=anchor_posY,
            anchor_scale_id=anchor_scale_id,
            anchor_orientation_id=anchor_orientation_id,
        )

        A_anchor = self.get_clean_image(
            imgs,
            latents_bases,
            posX_id=anchor["posX_id"],
            posY_id=anchor["posY_id"],
            orientation_id=anchor["orientation_id"],
            scale_id=anchor["scale_id"],
        )

        # Oracle ATT curve: scaling fixed by anchor posY
        c_anchor = self.att_scaling_from_anchor(anchor["posY_id"])
        EY_att = c_anchor * structural

        return do_A, EY_att, A_anchor, anchor

    def generate_dsprite_pv(
        self,
        data_path: str,
        n_sample: int,
        generate_test: bool = False,
        rand_seed: int = 42,
        **kwargs,
    ):
        dataset_zip = np.load(
            os.path.join(data_path, "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"),
            allow_pickle=True,
            encoding="bytes",
        )
        weights = self.cal_weight()
        imgs = dataset_zip["imgs"]
        latents_values = dataset_zip["latents_values"]
        metadata = dataset_zip["metadata"][()]

        latents_sizes = metadata[b"latents_sizes"]
        latents_bases = np.concatenate(
            (latents_sizes[::-1].cumprod()[::-1][1:], np.array([1]))
        )

        rng = default_rng(seed=rand_seed)
        posX_id_arr = rng.integers(32, size=n_sample)
        posY_id_arr = rng.integers(32, size=n_sample)
        scale_id_arr = rng.integers(6, size=n_sample)
        orientation_arr = rng.integers(40, size=n_sample)

        image_idx_arr = self.image_id(
            latents_bases, posX_id_arr, posY_id_arr, orientation_arr, scale_id_arr
        )

        treatment = imgs[image_idx_arr].reshape((n_sample, 64 * 64)).astype(np.float64)
        treatment += rng.normal(0.0, 0.1, treatment.shape)

        latent_feature = latents_values[image_idx_arr]  # (color, shape, scale, rotation, posX, posY)
        treatment_proxy = latent_feature[:, 2:5]        # (scale, rotation, posX)

        posX_id_proxy = np.full(n_sample, 16, dtype=int)
        scale_id_proxy = np.full(n_sample, 3, dtype=int)
        orientation_proxy = np.zeros(n_sample, dtype=int)

        proxy_image_idx_arr = self.image_id(
            latents_bases,
            posX_id_proxy,
            posY_id_arr,
            orientation_proxy,
            scale_id_proxy,
        )

        outcome_proxy = imgs[proxy_image_idx_arr].reshape((n_sample, 64 * 64)).astype(np.float64)
        outcome_proxy += rng.normal(0.0, 0.1, outcome_proxy.shape)

        structural = self.structural_func(treatment, weights)
        outcome = structural * ((posY_id_arr - 15.5) ** 2) / 85.25 + rng.normal(0.0, 0.5, n_sample)
        outcome = outcome[:, np.newaxis]

        if generate_test:
            do_A, EY_do_A = self.generate_test_dsprite_ate(data_path)
        else:
            do_A, EY_do_A = None, None

        return treatment, outcome, treatment_proxy, outcome_proxy, do_A, EY_do_A

    def generate_dsprite_pv_att(
        self,
        data_path: str,
        n_sample: int,
        generate_test: bool = False,
        rand_seed: int = 42,
        anchor_posX: float = 0.6,
        anchor_posY: float = 0.6,
        anchor_scale_id: int = 3,
        anchor_orientation_id: int = 0,
        **kwargs,
    ):
        """
        Same training DGP as generate_dsprite_pv, but returns ATT test objects.
        """
        A, Y, Z, W, _, _ = self.generate_dsprite_pv(
            data_path=data_path,
            n_sample=n_sample,
            generate_test=False,
            rand_seed=rand_seed,
            **kwargs,
        )

        if generate_test:
            do_A, EY_do_A, A_anchor, anchor_info = self.generate_test_dsprite_att(
                data_path=data_path,
                anchor_posX=anchor_posX,
                anchor_posY=anchor_posY,
                anchor_scale_id=anchor_scale_id,
                anchor_orientation_id=anchor_orientation_id,
            )
        else:
            do_A, EY_do_A, A_anchor, anchor_info = None, None, None, None

        return A, Y, Z, W, do_A, EY_do_A, A_anchor, anchor_info


def generate_synthetic_PCL_ATE_data(size = 1000, beta = 1, sigma = 1, do_A_range = (-1, 2), do_A_size = 100, seed = 10):
    """
    This data generation process is taken from Appendix H of the following paper:
    Doubly Robust Proximal Causal Learning for Continuous Treatments, Yong Wu, Yanwei Fu, Shouyan Wang, Xinwei Sun 
    """
    np.random.seed(seed)
    
    U2 = np.random.uniform(-1, 2, size = size)
    U1 = np.random.uniform(0, 1, size = size) - ((U2 >= 0) & (U2 <= 1)).astype(int)
    U = np.c_[U1, U2]
    
    Z2 = U2 + np.random.uniform(-1, 1, size = size)
    Z1 = U1 + np.random.normal(0, sigma, size = size)
    Z = np.c_[Z1, Z2]

    W1 = U1 + np.random.uniform(-1, 1, size = size)
    W2 = U2 + np.random.normal(0, sigma, size = size)
    W = np.c_[W1, W2]
    
    A = U2 + np.random.normal(0, beta, size = size)
    Y = 3 * np.cos(2 * (0.3 * U1 + 0.3 * U2 + 0.2) + 1.5 * A) + np.random.normal(0, 1, size = size)

    A, Y = A.reshape(size, -1), Y.reshape(size, -1)
    do_A = np.linspace(do_A_range[0], do_A_range[1], do_A_size)

    EY_do_A = []
    for a_ in do_A:
        U2 = np.random.uniform(-1, 2, size = 10000)
        U1 = np.random.uniform(0, 1, size = 10000) - ((U2 >= 0) & (U2 <= 1)).astype(int)
        EY_do_A.append(np.mean(3 * np.cos(2 * ( 0.3 * U1 + 0.3 * U2 + 0.2) + 1.5 * a_)))
    EY_do_A = np.array(EY_do_A)

    do_A, EY_do_A = do_A.reshape(do_A_size, -1), EY_do_A.reshape(do_A_size, -1)
    return U, W, Z, A, Y, do_A, EY_do_A


def gaussian_pdf(x: np.ndarray, mean: np.ndarray, std: float) -> np.ndarray:
    z = (x - mean) / std
    return np.exp(-0.5 * z * z) / (np.sqrt(2.0 * np.pi) * std)


@dataclass
class SyntheticLowDimATTBenchmark:
    """
    Synthetic low-dimensional proxy-causal ATT benchmark.

    Observational DGP:
        U1 ~ Unif[-1, 2]
        V  ~ Unif[0, 1]
        U2 = V - 1{0 <= U1 <= 1}

        W1 = U2 + Unif[-1, 1]
        W2 = U1 + N(0, 1)

        Z1 = U2 + N(0, 1)
        Z2 = U1 + Unif[-1, 1)

        A  = U1 + N(0, sigma_A^2)
        Y  = 3 cos( 2(0.3 U2 + 0.3 U1 + 0.2) + 1.5 A ) + N(0, sigma_Y^2)

    Target:
        f_ATT(a; a') = E[Y(a) | A = a']

    We compute the ground-truth ATT curve by:
        1) integrating out U2 analytically given U1;
        2) integrating over the posterior density p(U1 | A=a') numerically.
    """

    u1_low: float = -1.0
    u1_high: float = 2.0
    a_noise_std: float = 1.0
    y_noise_std: float = 1.0
    posterior_grid_size: int = 4001

    # ------------------------------------------------------------------
    # Latent / structural parts
    # ------------------------------------------------------------------
    def _sample_latents(self, n: int, rng: np.random.Generator):
        u1 = rng.uniform(self.u1_low, self.u1_high, size=n)
        v = rng.uniform(0.0, 1.0, size=n)
        u2 = v - ((u1 >= 0.0) & (u1 <= 1.0)).astype(np.float64)
        return u1, u2

    def structural_mean(self, a: np.ndarray, u1: np.ndarray, u2: np.ndarray) -> np.ndarray:
        """
        Noise-free conditional mean Y(a) given U1,U2.
        """
        return 3.0 * np.cos(2.0 * (0.3 * u2 + 0.3 * u1 + 0.2) + 1.5 * a)

    def _mean_over_u2_given_u1(self, a: np.ndarray, u1: np.ndarray) -> np.ndarray:
        """
        Computes E[Y(a) | U1=u1] exactly by integrating out U2 analytically.

        Since U2 = V - 1{0 <= U1 <= 1}, with V ~ Unif[0,1], we have
            Y(a) = 3 cos(0.6 V + c(u1,a)) + noise,
        so
            E[Y(a) | U1=u1] = 3 * [sin(c+0.6) - sin(c)] / 0.6
        """
        a = np.asarray(a, dtype=np.float64).reshape(-1, 1)      # (n_a, 1)
        u1 = np.asarray(u1, dtype=np.float64).reshape(1, -1)    # (1, n_u)

        indicator = ((u1 >= 0.0) & (u1 <= 1.0)).astype(np.float64)
        c = 0.6 * (u1 - indicator) + 0.4 + 1.5 * a              # broadcasted
        return 3.0 * (np.sin(c + 0.6) - np.sin(c)) / 0.6        # (n_a, n_u)

    def posterior_u1_given_anchor(self, anchor_a: float):
        """
        Posterior density p(U1 | A = anchor_a) on a dense grid.

        Since U1 ~ Unif[-1,2] and A|U1=u1 ~ N(u1, sigma_A^2),
        the posterior on [-1,2] is proportional to the Gaussian likelihood.
        """
        u1_grid = np.linspace(self.u1_low, self.u1_high, self.posterior_grid_size)
        post_unnorm = gaussian_pdf(anchor_a, mean=u1_grid, std=self.a_noise_std)
        post = post_unnorm / np.trapezoid(post_unnorm, u1_grid)
        return u1_grid, post

    # ------------------------------------------------------------------
    # Observational data
    # ------------------------------------------------------------------
    def generate_observational_data(self, n: int, seed: int = 0):
        rng = default_rng(seed)
        u1, u2 = self._sample_latents(n, rng)

        # outcome proxy W
        w1 = u2 + rng.uniform(-1.0, 1.0, size=n)
        w2 = u1 + rng.normal(0.0, 1.0, size=n)

        # treatment proxy Z
        z1 = u2 + rng.normal(0.0, 1.0, size=n)
        z2 = u1 + rng.uniform(-1.0, 1.0, size=n)

        # observed treatment and outcome
        a = u1 + rng.normal(0.0, self.a_noise_std, size=n)
        y = self.structural_mean(a, u1, u2) + rng.normal(0.0, self.y_noise_std, size=n)

        W = np.column_stack([w1, w2]).astype(np.float64)
        Z = np.column_stack([z1, z2]).astype(np.float64)
        A = a.reshape(-1, 1).astype(np.float64)
        Y = y.reshape(-1, 1).astype(np.float64)

        return {
            "U1": u1.reshape(-1, 1).astype(np.float64),
            "U2": u2.reshape(-1, 1).astype(np.float64),
            "W": W,
            "Z": Z,
            "A": A,
            "Y": Y,
        }

    # ------------------------------------------------------------------
    # Ground-truth curves
    # ------------------------------------------------------------------
    def generate_ate_curve(self, do_A_range=(-1.0, 2.0), do_A_size: int = 100):
        """
        Optional: population-level dose-response for comparison.
        """
        do_A = np.linspace(do_A_range[0], do_A_range[1], do_A_size, dtype=np.float64)
        u1_grid = np.linspace(self.u1_low, self.u1_high, self.posterior_grid_size)
        prior_u1 = np.ones_like(u1_grid) / (self.u1_high - self.u1_low)

        mean_surface = self._mean_over_u2_given_u1(do_A, u1_grid)   # (n_a, n_u)
        ey_ate = np.trapezoid(mean_surface * prior_u1.reshape(1, -1), u1_grid, axis=1)

        return do_A.reshape(-1, 1), ey_ate.reshape(-1, 1)

    def generate_att_curve(
        self,
        anchor_a: float,
        do_A_range=(-1.0, 2.0),
        do_A_size: int = 100,
    ):
        """
        Ground-truth ATT curve:
            f_ATT(a; a') = E[Y(a) | A = a']
        """
        do_A = np.linspace(do_A_range[0], do_A_range[1], do_A_size, dtype=np.float64)
        u1_grid, post_u1 = self.posterior_u1_given_anchor(anchor_a)

        mean_surface = self._mean_over_u2_given_u1(do_A, u1_grid)   # (n_a, n_u)
        ey_att = np.trapezoid(mean_surface * post_u1.reshape(1, -1), u1_grid, axis=1)

        return do_A.reshape(-1, 1), ey_att.reshape(-1, 1)

    # ------------------------------------------------------------------
    # Full benchmark object
    # ------------------------------------------------------------------
    def generate_dataset(
        self,
        n: int,
        seed: int = 0,
        anchor_a: float = 0.25,
        do_A_range=(-1.0, 2.0),
        do_A_size: int = 100,
        include_ate_curve: bool = False,
    ):
        obs = self.generate_observational_data(n=n, seed=seed)
        do_A, EY_att = self.generate_att_curve(
            anchor_a=anchor_a,
            do_A_range=do_A_range,
            do_A_size=do_A_size,
        )

        out = {
            "A": obs["A"],
            "Y": obs["Y"],
            "Z": obs["Z"],
            "W": obs["W"],
            "A_anchor": np.array([[anchor_a]], dtype=np.float64),
            "do_A": do_A,
            "EY_att": EY_att,
        }

        if include_ate_curve:
            _, EY_ate = self.generate_ate_curve(
                do_A_range=do_A_range,
                do_A_size=do_A_size,
            )
            out["EY_ate"] = EY_ate

        return out


def read_legalized_abortion_and_crime_dataset(data_path: str,
                                              return_test: bool = False,
                                              seed: int = 0):
    seed_str = str(seed)
    folder_path_train = data_path + '/train'
    folder_path_effect = data_path + '/test'
    
    train_name = f'main_ab_seed{seed_str}.npz'
    train_path = f'{folder_path_train}/{train_name}'
    train_data = np.load(train_path)
    W, Z, A, Y = train_data['train_w'], train_data['train_z'], train_data['train_a'], train_data['train_y']

    effect_name = f'do_A_ab_seed{seed_str}.npz'
    effect_path = f'{folder_path_effect}/{effect_name}'
    effect_data = np.load(effect_path)
    do_A, EY_do_A = effect_data['do_A'], effect_data['gt_EY_do_A']
    if return_test:
        W_test, Z_test, A_test, Y_test = train_data['test_w'], train_data['test_z'], train_data['test_a'], train_data['test_y']
        return W, Z, A, Y, W_test, Z_test, A_test, Y_test, do_A, EY_do_A
    else:
        return W, Z, A, Y, do_A, EY_do_A


def read_deaner_dataset(data_path: str,
                        id_: str,
                        seed: int,
                        return_test: bool = False):
    id_path = id_ + "_80_N"
    data_path = os.path.join(data_path, id_path)
    npz_train_file = f"main_edu_{id_}_80_seed{seed}.npz"
    npz_effect_file = f"do_A_edu_{id_}_80_seed{seed}.npz"
    train_data = np.load(os.path.join(data_path, npz_train_file))
    effect_data = np.load(os.path.join(data_path, npz_effect_file))
    W, Z, A, Y = train_data['train_w'], train_data['train_z'], train_data['train_a'], train_data['train_y']
    do_A, EY_do_A = effect_data['do_A'], effect_data['gt_EY_do_A'].reshape(-1, 1)
    filtered_indices = ((A > -0.1) & (A < 2.0)).reshape(-1)
    doA_filtered_indices = ((do_A > -0.1) & (do_A < 2.0)).reshape(-1)
    W, Z, A, Y, do_A, EY_do_A = W[filtered_indices], Z[filtered_indices], A[filtered_indices], Y[filtered_indices], do_A[doA_filtered_indices], EY_do_A[doA_filtered_indices]

    if return_test:
        W_test, Z_test, A_test, Y_test = train_data['test_w'], train_data['test_z'], train_data['test_a'], train_data['test_y']
        return W, Z, A, Y, W_test, Z_test, A_test, Y_test, do_A, EY_do_A
    else:
        return W, Z, A, Y, do_A, EY_do_A


def generate_demand_experiment_data(size = 1000, epsilon_std = 1.0, do_A_range = (10, 30), do_A_size = 10, seed = 10):

    def psi(t: np.ndarray) -> np.ndarray:
        return 2 * ((t - 5) ** 4 / 600 + np.exp(-4 * (t - 5) ** 2) + t / 10 - 2)

    def cal_outcome(price, views, demand):
        return np.clip(np.exp((views - price) / 10.0), None, 5.0) * price - 5 * psi(demand)

    np.random.seed(seed)
    
    demand = np.random.uniform(0, 10, size)
    cost1 = 2 * np.sin(demand * np.pi * 2 / 10) + np.random.normal(0, epsilon_std, size)
    cost2 = 2 * np.cos(demand * np.pi * 2 / 10) + np.random.normal(0, epsilon_std, size)
    price = 35 + (cost1 + 3) * psi(demand) + cost2 + np.random.normal(0, epsilon_std, size)
    views = 7 * psi(demand) + 45 + np.random.normal(0, epsilon_std, size)
    outcome = cal_outcome(price, views, demand) + np.random.normal(0, epsilon_std, size)

    A, Z, W, Y, U = price[:, np.newaxis], np.c_[cost1, cost2], views[:, np.newaxis], outcome[:, np.newaxis], demand[:, np.newaxis]
    
    do_A = np.linspace(do_A_range[0], do_A_range[1], do_A_size)
    EY_do_A = []
    for a_ in do_A:
        demand = np.random.uniform(0, 10, 10000)
        views = 7 * psi(demand) + 45 + np.random.normal(0, epsilon_std, 10000)
        EY_do_A.append(np.mean(cal_outcome(a_, views, demand)))

    EY_do_A = np.array(EY_do_A)

    do_A, EY_do_A = do_A.reshape(do_A_size, -1), EY_do_A.reshape(do_A_size, -1)

    return U, W, Z, A, Y, do_A, EY_do_A


def generate_synthetic_CATE_data(data_size: int,
                                 sigma: float = 0.5,
                                 uniform_noise_upper_bound: float = 0.5,
                                 uniform_noise_lower_bound: float = -0.5,
                                 covariate_v_range = (-0.5, 0.5), covariate_v_size: int = 100,
                                 seed: int = 42):
    """
    TODO: This one was from one of Liyuans paper that is taken from another papers. Find those papers and add citation here!
    """
    rng = default_rng(seed=seed)
    V = rng.uniform(low=-0.5, high=0.5, size=(data_size,))
    U1 = 1 + 2 * V + rng.uniform(low=-0.5, high=0.5, size=(data_size,))
    U2 = 1 + 2 * V + rng.uniform(low=-0.5, high=0.5, size=(data_size,))
    U3 = (V - 1) ** 2 + rng.uniform(low=-0.5, high=0.5, size=(data_size,))
    U = np.c_[U1, U2, U3]
    prob = 1.0 / (1.0 + np.exp(-0.5 * (V + U1 + U2 + U3)))
    A = (rng.random(data_size) < prob).astype(float)
    Y = V * U1 * U2 * U3 + rng.normal(0.0, 0.25, size=(data_size, ))
    Y *= A
    
    V = V.reshape(-1, 1)
    A, Y = A.reshape(-1, 1), Y.reshape(-1, 1)

    Z2 = U2 + rng.uniform(uniform_noise_lower_bound, uniform_noise_upper_bound, size = data_size)
    Z1 = U1 + rng.normal(0, sigma, size = data_size)
    Z3 = U3 + rng.uniform(uniform_noise_lower_bound, uniform_noise_upper_bound, size = data_size)
    Z = np.c_[Z1, Z2, Z3]

    W1 = U1 + rng.uniform(uniform_noise_lower_bound, uniform_noise_upper_bound, size = data_size)
    W2 = U2 + rng.normal(0, sigma, size = data_size)
    W3 = U3 + rng.normal(0, sigma / 2, size = data_size)
    W = np.c_[W1, W2, W3]

    # covariate_v_test = np.array([-0.4, -0.2, 0.0, 0.2, 0.4]).reshape(-1, 1)
    covariate_v_test = np.linspace(covariate_v_range[0], covariate_v_range[1], covariate_v_size).reshape(-1, 1)
    # do_A = np.array([1, 1, 1, 1, 1]).reshape(-1, 1)  # only test A = 1
    do_A = np.ones(shape = covariate_v_size).reshape(-1, 1)
    EY_do_A_CATE = covariate_v_test * ((1 + 2 * covariate_v_test) ** 2) * ((covariate_v_test - 1) ** 2)
    
    return U, W, Z, V, A, Y, covariate_v_test, do_A, EY_do_A_CATE