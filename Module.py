
"""
Project 2 : 
"""
from __future__ import annotations
import os, math, random
from dataclasses import dataclass
from typing import Tuple
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm.auto import tqdm
from mamba_ssm import Mamba
try:
    from pytorch_msssim import ssim
except ImportError:
    ssim = None

import sys

sys.path.insert(
    0,
    "/mnt/c/Users/lenovo/BasicSR"
)

from basicsr.metrics.psnr_ssim import (
    calculate_psnr,
    calculate_ssim
)
def seed_everything(seed: int = 123, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
#-----------------------------------------
def seed_worker(worker_id):
    worker_seed = 123 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# -----------------------------
# Utilities: Metrics & helpers
# -----------------------------

class EMA:
    """
    Exponential Moving Average for model parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999
    ):
        self.decay = float(decay)
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    param.detach().clone()
                )

    @torch.no_grad()
    def update(
        self,
        model: nn.Module
    ):
        for name, param in model.named_parameters():

            if not param.requires_grad:
                continue

            if name not in self.shadow:
                raise KeyError(
                    f"EMA parameter '{name}' was not "
                    "initialized in shadow weights."
                )

            new_average = (
                self.decay * self.shadow[name]
                + (1.0 - self.decay) * param.detach()
            )

            self.shadow[name] = (
                new_average.clone()
            )

    @torch.no_grad()
    def store(
        self,
        model: nn.Module
    ):
        """
        Save current non-EMA model parameters before
        temporarily copying EMA weights into the model.
        """
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = (
                    param.detach().clone()
                )

    @torch.no_grad()
    def copy_to(
        self,
        model: nn.Module
    ):
        """
        Temporarily copy EMA weights into the model.
        """
        for name, param in model.named_parameters():

            if (
                param.requires_grad
                and name in self.shadow
            ):
                param.data.copy_(
                    self.shadow[name].data
                )

    @torch.no_grad()
    def restore(
        self,
        model: nn.Module
    ):
        """
        Restore the original non-EMA model weights.
        """
        for name, param in model.named_parameters():

            if (
                param.requires_grad
                and name in self.backup
            ):
                param.data.copy_(
                    self.backup[name].data
                )

        self.backup = {}

    def state_dict(self):
        """
        Return the EMA state for checkpoint saving.
        """
        return {
            "decay": self.decay,
            "shadow": {
                name: tensor.detach().cpu().clone()
                for name, tensor in self.shadow.items()
            }
        }

    def load_state_dict(
        self,
        state_dict,
        device=None
    ):
        """
        Restore EMA shadow weights from checkpoint.
        """

        self.decay = float(
            state_dict.get(
                "decay",
                self.decay
            )
        )

        if "shadow" not in state_dict:
            raise KeyError(
                "EMA checkpoint does not contain "
                "the 'shadow' entry."
            )

        self.shadow = {}

        for name, tensor in (
            state_dict["shadow"].items()
        ):
            if device is not None:
                tensor = tensor.to(device)

            self.shadow[name] = (
                tensor.detach().clone()
            )




@torch.no_grad()
def batch_psnr_ssim_basic_sr(
    pred: torch.Tensor,
    gt: torch.Tensor
) -> Tuple[float, float]:
    """
    Compute PSNR and SSIM exactly in the same way used
    by the final testing cell.

    Input:
        pred, gt: (N,1,H,W), normally in [0,1]

    Procedure:
        clamp to [0,1]
        -> convert each image to uint8 using round()
        -> BasicSR calculate_psnr/calculate_ssim
    """

    if pred.dim() != 4 or gt.dim() != 4:
        raise ValueError(
            "Expected pred and gt with shape (N,C,H,W)."
        )

    if pred.shape != gt.shape:
        raise ValueError(
            f"Shape mismatch: pred={tuple(pred.shape)}, "
            f"gt={tuple(gt.shape)}."
        )

    pred = (
        pred.clamp(0.0, 1.0)
        .detach()
        .cpu()
        .numpy()
    )

    gt = (
        gt.clamp(0.0, 1.0)
        .detach()
        .cpu()
        .numpy()
    )

    psnr_values = []
    ssim_values = []

    for i in range(pred.shape[0]):

        # grayscale CT image
        pred01 = pred[i, 0]
        gt01 = gt[i, 0]

        pred_u8 = (
            pred01 * 255.0
        ).round().clip(
            0,
            255
        ).astype(np.uint8)

        gt_u8 = (
            gt01 * 255.0
        ).round().clip(
            0,
            255
        ).astype(np.uint8)

        psnr_value = calculate_psnr(
            pred_u8,
            gt_u8,
            crop_border=0,
            test_y_channel=False
        )

        ssim_value = calculate_ssim(
            pred_u8,
            gt_u8,
            crop_border=0,
            test_y_channel=False
        )

        psnr_values.append(
            float(psnr_value)
        )

        ssim_values.append(
            float(ssim_value)
        )

    return (
        float(np.mean(psnr_values)),
        float(np.mean(ssim_values))
    )
# -----------------------------
# Network building blocks
# -----------------------------
#--------------------------------------------------------------
class GHMFullMWT2D(nn.Module):
    """
    Full 2D GHM multiwavelet transform using MATLAB-style repeated-row preprocessing.

    MATLAB preprocessing:
        vector sample = [x, x / sqrt(2)]

    Input:
        x: (N, C, H, W)

    Output:
        bands: (N, 16*C, H/2, W/2)
    """
    def __init__(self):
        super().__init__()
        
        s2 = math.sqrt(2.0)

        H0 = torch.tensor([
            [ 3.0 / (5.0 * s2),  4.0 / 5.0],
            [-1.0 / 20.0,       -3.0 / (10.0 * s2)],
        ], dtype=torch.float32)

        H1 = torch.tensor([
            [ 3.0 / (5.0 * s2),  0.0],
            [ 9.0 / 20.0,        1.0 / s2],
        ], dtype=torch.float32)

        H2 = torch.tensor([
            [ 0.0,               0.0],
            [ 9.0 / 20.0,       -3.0 / (10.0 * s2)],
        ], dtype=torch.float32)

        H3 = torch.tensor([
            [ 0.0,        0.0],
            [-1.0 / 20.0, 0.0],
        ], dtype=torch.float32)

        G0 = (1.0 / 10.0) * torch.tensor([
            [-0.5,        -3.0 / s2],
            [ 1.0 / s2,    3.0],
        ], dtype=torch.float32)

        G1 = (1.0 / 10.0) * torch.tensor([
            [ 9.0 / 2.0,  -10.0 / s2],
            [-9.0 / s2,     0.0],
        ], dtype=torch.float32)

        G2 = (1.0 / 10.0) * torch.tensor([
            [ 9.0 / 2.0,  -3.0 / s2],
            [ 9.0 / s2,   -3.0],
        ], dtype=torch.float32)

        G3 = (1.0 / 10.0) * torch.tensor([
            [-0.5,       0.0],
            [-1.0 / s2,  0.0],
        ], dtype=torch.float32)

        self.register_buffer("H_mats", torch.stack([H0, H1, H2, H3], dim=0))
        self.register_buffer("G_mats", torch.stack([G0, G1, G2, G3], dim=0))

        # Fixed MATLAB-style repeated-row scale.
        # It is stored with the model but is not trainable.
        self.register_buffer(
            "pre_scale",
            torch.tensor(
                1.0 / math.sqrt(2.0),
                dtype=torch.float32
            )
        )

    def _get_pre_scale(self):
        """
        Return the fixed MATLAB-style preprocessing scale.
    
        Fixed value:
            1 / sqrt(2) = 0.70710678...
        """
        return self.pre_scale

    def _preprocess_width_matlab(self, x):
        """
        MATLAB-style repeated-row preprocessing along width,
        with a fixed scale of 1/sqrt(2).

    
        Input : (N,C,H,W)
        Output: (N,C,H,W,2)
        """
        pre_scale = self._get_pre_scale()
        v1 = x
        v2 = x * pre_scale
        return torch.stack([v1, v2], dim=-1)

    def _preprocess_height_matlab(self, x):
        
        pre_scale = self._get_pre_scale()
        v1 = x.permute(0, 1, 3, 2).contiguous()
        v2 = v1 * pre_scale
        return torch.stack([v1, v2], dim=-1)

    def _mfilter_width(self, x):
        """
        Width transform.

        Input : (N,C,H,W)
        Output: (N,4C,H,W/2)
        """
        N, C, H, W = x.shape

        if W % 2 != 0:
            raise ValueError(f"W must be divisible by 2. Got W={W}")

        v = self._preprocess_width_matlab(x)  # (N,C,H,W,2)

        Lout = W // 2
        base = 2 * torch.arange(Lout, device=x.device)
        offs = torch.arange(4, device=x.device)
        idx = (base[:, None] + offs[None, :]) % W

        gathered = v[:, :, :, idx, :]  # (N,C,H,Lout,4,2)

        low  = torch.einsum("nchlkq,kpq->nchlp", gathered, self.H_mats)
        high = torch.einsum("nchlkq,kpq->nchlp", gathered, self.G_mats)

        out = torch.stack(
            [low[..., 0], low[..., 1], high[..., 0], high[..., 1]],
            dim=2
        )  # (N,C,4,H,W/2)

        return out.reshape(N, C * 4, H, Lout)

    def _mfilter_height(self, x):
        """
        Height transform.

        Input : (N,C,H,W)
        Output: (N,4C,H/2,W)
        """
        N, C, H, W = x.shape

        if H % 2 != 0:
            raise ValueError(f"H must be divisible by 2. Got H={H}")

        v = self._preprocess_height_matlab(x)  # (N,C,W,H,2)

        Lout = H // 2
        base = 2 * torch.arange(Lout, device=x.device)
        offs = torch.arange(4, device=x.device)
        idx = (base[:, None] + offs[None, :]) % H

        gathered = v[:, :, :, idx, :]  # (N,C,W,Lout,4,2)

        low  = torch.einsum("ncwlkq,kpq->ncwlp", gathered, self.H_mats)
        high = torch.einsum("ncwlkq,kpq->ncwlp", gathered, self.G_mats)

        out = torch.stack(
            [low[..., 0], low[..., 1], high[..., 0], high[..., 1]],
            dim=2
        )  # (N,C,4,W,H/2)

        out = out.permute(0, 1, 2, 4, 3).contiguous()
        return out.reshape(N, C * 4, Lout, W)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"GHMFullMWT2D expects (N,C,H,W), got {tuple(x.shape)}")

        N, C, H, W = x.shape

        if H % 2 != 0 or W % 2 != 0:
            raise ValueError(f"H and W must be divisible by 2. Got H={H}, W={W}")

        xw = self._mfilter_width(x)      # (N,4C,H,W/2)
        bands = self._mfilter_height(xw) # (N,16C,H/2,W/2)

        return bands
#----------------------------------------------------
class GHMInverseMWT2D(nn.Module):
    """
    Numerical inverse of GHMFullMWT2D using pseudo-inverse
    matrices for the corresponding 1D GHM analysis transform.

    The inverse assumes the exact same fixed MATLAB-style
    preprocessing scale used by GHMFullMWT2D:

        pre_scale = 1 / sqrt(2)

    Input:
        bands:
            (N, 16*C, H/2, W/2)

    Output:
        reconstructed:
            (N, C, H, W)

    Notes:
        - H_mats and G_mats must come from the corresponding
          GHMFullMWT2D instance.
        - pre_scale is fixed and non-trainable.
        - The pseudo-inverse matrices are cached after their
          first construction for each spatial size/device/dtype.
    """

    def __init__(
        self,
        H_mats: torch.Tensor,
        G_mats: torch.Tensor
    ):
        super().__init__()

        if H_mats.shape != (4, 2, 2):
            raise ValueError(
                "H_mats must have shape (4,2,2), "
                f"received {tuple(H_mats.shape)}."
            )

        if G_mats.shape != (4, 2, 2):
            raise ValueError(
                "G_mats must have shape (4,2,2), "
                f"received {tuple(G_mats.shape)}."
            )

        self.register_buffer(
            "H_mats",
            H_mats.detach().clone()
        )

        self.register_buffer(
            "G_mats",
            G_mats.detach().clone()
        )

        # Fixed MATLAB-style repeated-row scale.
        #
        # It exactly matches GHMFullMWT2D and is not
        # a trainable parameter.
        self.register_buffer(
            "pre_scale",
            torch.tensor(
                1.0 / math.sqrt(2.0),
                dtype=torch.float32
            )
        )

        # Numerical pseudo-inverse matrices are built once
        # for each spatial size/device/dtype combination.
        self._pinv_cache = {}

    def _get_pre_scale(self) -> float:
        """
        Return the fixed preprocessing scale as a Python float.

        Fixed value:
            1 / sqrt(2) = 0.70710678...
        """
        return float(
            self.pre_scale.detach().cpu().item()
        )

    def _build_pinv_1d(
        self,
        length_out: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Build the pseudo-inverse matrix for one 1D GHM
        analysis operation.

        Forward analysis dimensions:
            input signal:
                2 * length_out

            output coefficients:
                4 * length_out

        Pseudo-inverse dimensions:
            (4 * length_out) -> (2 * length_out)
        """

        if length_out <= 0:
            raise ValueError(
                "length_out must be positive, "
                f"received {length_out}."
            )

        pre_scale = self._get_pre_scale()

        cache_key = (
            int(length_out),
            str(device),
            str(dtype),
            pre_scale
        )

        if cache_key in self._pinv_cache:
            return self._pinv_cache[
                cache_key
            ]

        length_in = 2 * length_out

        H_mats = self.H_mats.to(
            device=device,
            dtype=dtype
        )

        G_mats = self.G_mats.to(
            device=device,
            dtype=dtype
        )

        # Complete linear analysis matrix:
        #
        # A:
        #   (4*length_out, 2*length_out)
        analysis_matrix = torch.zeros(
            (
                4 * length_out,
                length_in
            ),
            device=device,
            dtype=dtype
        )

        base = 2 * torch.arange(
            length_out,
            device=device
        )

        offsets = torch.arange(
            4,
            device=device
        )

        indices = (
            base[:, None]
            + offsets[None, :]
        ) % length_in

        # Construct the analysis matrix column by column
        # by applying the exact forward operation to basis
        # vectors.
        for input_index in range(length_in):

            basis_signal = torch.zeros(
                length_in,
                device=device,
                dtype=dtype
            )

            basis_signal[
                input_index
            ] = 1.0

            repeated_component_1 = (
                basis_signal
            )

            repeated_component_2 = (
                basis_signal * pre_scale
            )

            vector_signal = torch.stack(
                [
                    repeated_component_1,
                    repeated_component_2
                ],
                dim=-1
            )

            gathered = vector_signal[
                indices,
                :
            ]

            low = torch.einsum(
                "lkq,kpq->lp",
                gathered,
                H_mats
            )

            high = torch.einsum(
                "lkq,kpq->lp",
                gathered,
                G_mats
            )

            coefficients = torch.stack(
                [
                    low[:, 0],
                    low[:, 1],
                    high[:, 0],
                    high[:, 1]
                ],
                dim=0
            )

            analysis_matrix[
                :,
                input_index
            ] = coefficients.reshape(-1)

        # Build the numerical inverse on CPU for broader
        # linear-algebra compatibility, then return it to
        # the requested device and dtype.
        pseudo_inverse = torch.linalg.pinv(
            analysis_matrix
            .detach()
            .cpu()
            .to(torch.float64)
        )

        pseudo_inverse = pseudo_inverse.to(
            device=device,
            dtype=dtype
        )

        self._pinv_cache[
            cache_key
        ] = pseudo_inverse

        return pseudo_inverse

    def _inverse_height(
        self,
        bands: torch.Tensor
    ) -> torch.Tensor:
        """
        Reverse GHMFullMWT2D._mfilter_height.

        Input:
            bands:
                (N, 4*Cw, H2, W2)

        Output:
            width_coefficients:
                (N, Cw, 2*H2, W2)
        """

        n, channels_times_four, h2, w2 = (
            bands.shape
        )

        if channels_times_four % 4 != 0:
            raise ValueError(
                "The channel count entering inverse height "
                "must be divisible by 4. "
                f"Received {channels_times_four}."
            )

        channels_width = (
            channels_times_four // 4
        )

        reconstructed_height = 2 * h2

        pinv_height = self._build_pinv_1d(
            length_out=h2,
            device=bands.device,
            dtype=bands.dtype
        )

        # Restore explicit coefficient-component dimension:
        #
        # (N,4*Cw,H2,W2)
        #       ->
        # (N,Cw,4,H2,W2)
        bands_grouped = bands.reshape(
            n,
            channels_width,
            4,
            h2,
            w2
        )

        # Each spatial width position gets one coefficient
        # vector of length 4*H2.
        coefficient_vectors = (
            bands_grouped
            .permute(0, 1, 4, 2, 3)
            .contiguous()
            .reshape(
                -1,
                4 * h2
            )
        )

        reconstructed_vectors = (
            coefficient_vectors
            @ pinv_height.transpose(0, 1)
        )

        width_coefficients = (
            reconstructed_vectors
            .reshape(
                n,
                channels_width,
                w2,
                reconstructed_height
            )
            .permute(0, 1, 3, 2)
            .contiguous()
        )

        return width_coefficients

    def _inverse_width(
        self,
        width_coefficients: torch.Tensor
    ) -> torch.Tensor:
        """
        Reverse GHMFullMWT2D._mfilter_width.

        Input:
            width_coefficients:
                (N, 4*C, H, W2)

        Output:
            reconstructed:
                (N, C, H, 2*W2)
        """

        n, channels_times_four, h, w2 = (
            width_coefficients.shape
        )

        if channels_times_four % 4 != 0:
            raise ValueError(
                "The channel count entering inverse width "
                "must be divisible by 4. "
                f"Received {channels_times_four}."
            )

        output_channels = (
            channels_times_four // 4
        )

        reconstructed_width = 2 * w2

        pinv_width = self._build_pinv_1d(
            length_out=w2,
            device=width_coefficients.device,
            dtype=width_coefficients.dtype
        )

        # (N,4*C,H,W2)
        #       ->
        # (N,C,4,H,W2)
        width_grouped = (
            width_coefficients.reshape(
                n,
                output_channels,
                4,
                h,
                w2
            )
        )

        # Every image row gets a coefficient vector with
        # length 4*W2.
        coefficient_vectors = (
            width_grouped
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .reshape(
                -1,
                4 * w2
            )
        )

        reconstructed_vectors = (
            coefficient_vectors
            @ pinv_width.transpose(0, 1)
        )

        reconstructed = (
            reconstructed_vectors.reshape(
                n,
                output_channels,
                h,
                reconstructed_width
            )
        )

        return reconstructed

    def forward(
        self,
        bands: torch.Tensor
    ) -> torch.Tensor:

        if bands.dim() != 4:
            raise ValueError(
                "GHMInverseMWT2D expects input with shape "
                "(N,16*C,H/2,W/2), received "
                f"{tuple(bands.shape)}."
            )

        n, band_channels, h2, w2 = (
            bands.shape
        )

        if band_channels % 16 != 0:
            raise ValueError(
                "The GHM inverse input channel count must be "
                "divisible by 16. "
                f"Received {band_channels}."
            )

        if h2 <= 0 or w2 <= 0:
            raise ValueError(
                "The GHM coefficient spatial dimensions must "
                f"be positive. Received H={h2}, W={w2}."
            )

        if not torch.isfinite(bands).all():
            raise ValueError(
                "Non-finite values were found in the GHM "
                "inverse input coefficients."
            )

        # Reverse the second forward operation first:
        # height inverse, then width inverse.
        width_coefficients = self._inverse_height(
            bands
        )

        reconstructed = self._inverse_width(
            width_coefficients
        )

        expected_output_shape = (
            n,
            band_channels // 16,
            h2 * 2,
            w2 * 2
        )

        if tuple(reconstructed.shape) != (
            expected_output_shape
        ):
            raise RuntimeError(
                "Unexpected inverse-GHM output shape. "
                f"Received {tuple(reconstructed.shape)}, "
                f"expected {expected_output_shape}."
            )

        return reconstructed
#-------------------------------------------------------
class HaarDWT2D(nn.Module):
    """
    Fixed orthonormal 2D Haar wavelet transform.

    Input:
        x: (N, C, H, W)

    Output:
        bands: (N, 4*C, H/2, W/2)

    Band order for each input channel:
        0: LL
        1: LH
        2: HL
        3: HH

    The transform contains no trainable parameters.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        if x.dim() != 4:
            raise ValueError(
                "HaarDWT2D expects input with shape "
                f"(N,C,H,W), received {tuple(x.shape)}."
            )

        n, c, h, w = x.shape

        if h % 2 != 0 or w % 2 != 0:
            raise ValueError(
                "Input height and width must be divisible by 2 "
                "for Haar decomposition. "
                f"Received H={h}, W={w}."
            )

        # Four spatial samples in every 2×2 block.
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        # Orthonormal 2D Haar decomposition.
        ll = (
            x00 + x01 + x10 + x11
        ) * 0.5

        lh = (
            -x00 - x01 + x10 + x11
        ) * 0.5

        hl = (
            -x00 + x01 - x10 + x11
        ) * 0.5

        hh = (
            x00 - x01 - x10 + x11
        ) * 0.5

        # For one-channel CT input:
        # (N,4,H/2,W/2)
        bands = torch.cat(
            [
                ll,
                lh,
                hl,
                hh
            ],
            dim=1
        )

        return bands
#----------------------------------------------------------
# --------------------------------------------------------------
class HaarInverseDWT2D(nn.Module):
    """
    Fixed inverse orthonormal 2D Haar wavelet transform.

    Input:
        bands: (N, 4*C, H/2, W/2)

    Band order:
        LL, LH, HL, HH

    Output:
        reconstructed: (N, C, H, W)

    The transform contains no trainable parameters.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        bands: torch.Tensor
    ) -> torch.Tensor:

        if bands.dim() != 4:
            raise ValueError(
                "HaarInverseDWT2D expects input with shape "
                "(N,4*C,H/2,W/2), received "
                f"{tuple(bands.shape)}."
            )

        n, band_channels, h2, w2 = bands.shape

        if band_channels % 4 != 0:
            raise ValueError(
                "The Haar inverse input channel count must be "
                "divisible by 4. "
                f"Received {band_channels}."
            )

        output_channels = band_channels // 4

        # The forward Haar transform concatenates:
        # [LL, LH, HL, HH]
        ll, lh, hl, hh = torch.split(
            bands,
            output_channels,
            dim=1
        )

        # Inverse orthonormal Haar reconstruction.
        x00 = (
            ll - lh - hl + hh
        ) * 0.5

        x01 = (
            ll - lh + hl - hh
        ) * 0.5

        x10 = (
            ll + lh - hl - hh
        ) * 0.5

        x11 = (
            ll + lh + hl + hh
        ) * 0.5

        reconstructed = torch.zeros(
            (
                n,
                output_channels,
                h2 * 2,
                w2 * 2
            ),
            device=bands.device,
            dtype=bands.dtype
        )

        reconstructed[
            :,
            :,
            0::2,
            0::2
        ] = x00

        reconstructed[
            :,
            :,
            0::2,
            1::2
        ] = x01

        reconstructed[
            :,
            :,
            1::2,
            0::2
        ] = x10

        reconstructed[
            :,
            :,
            1::2,
            1::2
        ] = x11

        if not torch.isfinite(
            reconstructed
        ).all():
            raise ValueError(
                "Non-finite values were found in the "
                "inverse-Haar output."
            )

        return reconstructed

#--------------------------------------------------------------
class MambaBlock2D(nn.Module):
    """
    2D -> sequence -> Mamba -> 2D
    input/output: (N, C, H, W)
    """
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W)
        n, c, h, w = x.shape
        residual = x

        # -> (N, H*W, C)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = self.mamba(x)

        # -> (N, C, H, W)
        x = x.transpose(1, 2).reshape(n, c, h, w)

        return x + residual



#-----------------------------------------
class ConvResBlock(nn.Module):
    """
    Simple residual conv block for low-frequency branch.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)

#---------------------------------------------------------------

# generator--------------
class LayerNorm2D(nn.Module):
    """
    Channel-wise LayerNorm for 2D feature maps.

    Input/output:
        (N, C, H, W)

    Normalization is applied over the channel dimension
    independently at every spatial location.
    """

    def __init__(
        self,
        channels: int,
        eps: float = 1e-6
    ):
        super().__init__()

        self.norm = nn.LayerNorm(
            channels,
            eps=eps
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        if x.dim() != 4:
            raise ValueError(
                "LayerNorm2D expects a 4D tensor "
                f"(N,C,H,W), received {tuple(x.shape)}."
            )

        # (N,C,H,W) -> (N,H,W,C)
        x = x.permute(
            0,
            2,
            3,
            1
        ).contiguous()

        x = self.norm(x)

        # (N,H,W,C) -> (N,C,H,W)
        x = x.permute(
            0,
            3,
            1,
            2
        ).contiguous()

        return x
#---------------------------------------
class GatedDconvFeedForward(nn.Module):
    """
    Restormer-inspired gated depthwise-convolutional
    feed-forward network.

    Input/output:
        (N, C, H, W)

    Processing:
        1x1 expansion
        -> depthwise 3x3 convolution
        -> split into two feature groups
        -> GELU gating
        -> 1x1 projection
    """

    def __init__(
        self,
        channels: int,
        expansion_factor: float = 2.0,
        bias: bool = False
    ):
        super().__init__()

        if channels <= 0:
            raise ValueError(
                "channels must be positive. "
                f"Received {channels}."
            )

        if expansion_factor <= 0:
            raise ValueError(
                "expansion_factor must be positive. "
                f"Received {expansion_factor}."
            )

        hidden_channels = int(
            channels * expansion_factor
        )

        if hidden_channels < 1:
            raise ValueError(
                "The expanded channel count must be positive."
            )

        self.channels = int(channels)
        self.hidden_channels = int(
            hidden_channels
        )

        # Produce two feature groups for gated interaction.
        self.project_in = nn.Conv2d(
            channels,
            hidden_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )

        # Spatial local processing without expensive
        # full channel mixing.
        self.depthwise_conv = nn.Conv2d(
            hidden_channels * 2,
            hidden_channels * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_channels * 2,
            bias=bias
        )

        self.project_out = nn.Conv2d(
            hidden_channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        if x.dim() != 4:
            raise ValueError(
                "GatedDconvFeedForward expects a 4D tensor, "
                f"received {tuple(x.shape)}."
            )

        if x.size(1) != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, "
                f"received {x.size(1)}."
            )

        expanded = self.project_in(x)

        expanded = self.depthwise_conv(
            expanded
        )

        feature_a, feature_b = torch.chunk(
            expanded,
            chunks=2,
            dim=1
        )

        gated = (
            F.gelu(feature_a)
            * feature_b
        )

        output = self.project_out(
            gated
        )

        return output
#--------------------------------------
class ResidualGDFNBlock(nn.Module):
    """
    Pre-normalized residual GDFN refinement block.

    output = input + scale * GDFN(LayerNorm(input))
    """

    def __init__(
        self,
        channels: int,
        expansion_factor: float = 2.0,
        residual_scale: float = 0.1,
        bias: bool = False
    ):
        super().__init__()

        if residual_scale <= 0.0:
            raise ValueError(
                "residual_scale must be positive. "
                f"Received {residual_scale}."
            )

        self.channels = int(channels)
        self.residual_scale = float(
            residual_scale
        )

        self.norm = LayerNorm2D(
            channels=channels
        )

        self.gdfn = GatedDconvFeedForward(
            channels=channels,
            expansion_factor=expansion_factor,
            bias=bias
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        if x.dim() != 4:
            raise ValueError(
                "ResidualGDFNBlock expects a 4D tensor, "
                f"received {tuple(x.shape)}."
            )

        if x.size(1) != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, "
                f"received {x.size(1)}."
            )

        refinement = self.gdfn(
            self.norm(x)
        )

        output = (
            x
            + self.residual_scale
            * refinement
        )

        return output
#--------------------------------------
class AdaptiveGHMFrequencyCollaborativeRefinement(nn.Module):
    """
    Adaptive GHM frequency-spatial collaborative refinement.

    This block operates after LF-guided alignment.

    Input:
        bands:
            (N, B, C, H, W)

    Main idea:
        1. Build explicit frequency context from the
           Low / Mixed / High GHM band groups.

        2. Refine every band locally using a spatial GDFN path.

        3. Generate a frequency-conditioned correction for
           every band using the shared GHM frequency context.

        4. Learn an adaptive gate between:
               local spatial correction
               frequency-guided correction

        5. Inject the collaborative correction residually
           with a conservative learnable strength.

    Output:
        (N, B, C, H, W)
    """

    def __init__(
        self,
        channels: int,
        low_band_indices,
        mixed_band_indices,
        high_band_indices,
        expansion_factor: float = 2.0,
        init_scale: float = 0.10,
        bias: bool = False
    ):
        super().__init__()

        if channels <= 0:
            raise ValueError(
                f"channels must be positive. Received {channels}."
            )

        if not 0.0 <= init_scale <= 1.0:
            raise ValueError(
                "init_scale must be in [0,1]. "
                f"Received {init_scale}."
            )

        self.channels = int(channels)

        self.low_band_indices = tuple(
            int(i) for i in low_band_indices
        )

        self.mixed_band_indices = tuple(
            int(i) for i in mixed_band_indices
        )

        self.high_band_indices = tuple(
            int(i) for i in high_band_indices
        )

        # -------------------------------------------------
        # 1. Frequency-group context
        #
        # Mean Low features   : C
        # Mean Mixed features : C
        # Mean High features  : C
        #
        # concatenation:
        #       3C -> C
        # -------------------------------------------------
        self.frequency_context = nn.Sequential(
            nn.Conv2d(
                channels * 3,
                channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias
            ),
            nn.GELU(),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=channels,
                bias=bias
            ),
            nn.GELU(),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias
            )
        )

        # -------------------------------------------------
        # 2. Local spatial branch
        #
        # This preserves the useful role of the original
        # GDFN, but produces a correction instead of a
        # complete residual block.
        # -------------------------------------------------
        self.local_norm = LayerNorm2D(
            channels=channels
        )

        self.local_ffn = GatedDconvFeedForward(
            channels=channels,
            expansion_factor=expansion_factor,
            bias=bias
        )

        # -------------------------------------------------
        # 3. Frequency-conditioned branch
        #
        # Each band feature is combined with the common
        # Low/Mixed/High frequency context.
        #
        # 2C -> C
        # -------------------------------------------------
        self.frequency_adapter = nn.Sequential(
            nn.Conv2d(
                channels * 2,
                channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias
            ),
            nn.GELU(),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=channels,
                bias=bias
            ),
            nn.GELU(),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias
            )
        )

        # -------------------------------------------------
        # 4. Adaptive collaboration gate
        #
        # The gate decides, for every spatial position and
        # channel, how much to use:
        #
        # frequency-guided correction
        # versus
        # local spatial correction.
        # -------------------------------------------------
        self.collaboration_gate = nn.Sequential(
            nn.Conv2d(
                channels * 3,
                channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.GELU(),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=channels,
                bias=True
            ),

            nn.Sigmoid()
        )

        # -------------------------------------------------
        # Conservative residual strength.
        #
        # A single scale is deliberately used in the first
        # experiment to avoid adding extra Low/Mixed/High
        # assumptions.
        # -------------------------------------------------
        self.collaboration_scale = nn.Parameter(
            torch.tensor(
                init_scale,
                dtype=torch.float32
            )
        )

    def forward(
        self,
        bands: torch.Tensor
    ) -> torch.Tensor:

        if bands.dim() != 5:
            raise ValueError(
                "AdaptiveGHMFrequencyCollaborativeRefinement "
                "expects (N,B,C,H,W), received "
                f"{tuple(bands.shape)}."
            )

        n, b, c, h, w = bands.shape

        if c != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, "
                f"received {c}."
            )

        # -------------------------------------------------
        # Validate band indices
        # -------------------------------------------------
        all_indices = (
            self.low_band_indices
            + self.mixed_band_indices
            + self.high_band_indices
        )

        if len(all_indices) != b:
            raise ValueError(
                "The number of configured frequency-band "
                f"indices ({len(all_indices)}) does not match "
                f"the input number of bands ({b})."
            )

        if sorted(all_indices) != list(range(b)):
            raise ValueError(
                "Low/Mixed/High band indices must form an "
                "exact partition of all input bands."
            )

        # -------------------------------------------------
        # 1. Explicit GHM frequency-group summaries
        # -------------------------------------------------
        low_context = bands[
            :,
            self.low_band_indices,
            :,
            :,
            :
        ].mean(
            dim=1
        )

        mixed_context = bands[
            :,
            self.mixed_band_indices,
            :,
            :,
            :
        ].mean(
            dim=1
        )

        high_context = bands[
            :,
            self.high_band_indices,
            :,
            :,
            :
        ].mean(
            dim=1
        )

        # -------------------------------------------------
        # 2. Learn shared frequency context
        #
        # (N,3C,H,W) -> (N,C,H,W)
        # -------------------------------------------------
        frequency_context = self.frequency_context(
            torch.cat(
                [
                    low_context,
                    mixed_context,
                    high_context
                ],
                dim=1
            )
        )

        # -------------------------------------------------
        # 3. Flatten bands for shared processing
        #
        # (N,B,C,H,W)
        # ->
        # (N*B,C,H,W)
        # -------------------------------------------------
        bands_flat = bands.reshape(
            n * b,
            c,
            h,
            w
        )

        normalized_flat = self.local_norm(
            bands_flat
        )

        # -------------------------------------------------
        # 4. Local spatial correction
        # -------------------------------------------------
        local_delta = self.local_ffn(
            normalized_flat
        )

        # -------------------------------------------------
        # 5. Broadcast the learned GHM frequency context
        #    to every band.
        # -------------------------------------------------
        context_flat = (
            frequency_context
            .unsqueeze(1)
            .expand(
                n,
                b,
                c,
                h,
                w
            )
            .contiguous()
            .reshape(
                n * b,
                c,
                h,
                w
            )
        )

        # -------------------------------------------------
        # 6. Frequency-guided correction
        # -------------------------------------------------
        frequency_delta = self.frequency_adapter(
            torch.cat(
                [
                    normalized_flat,
                    context_flat
                ],
                dim=1
            )
        )

        # -------------------------------------------------
        # 7. Adaptive collaboration
        #
        # Gate ≈ 0:
        #     prefer local spatial refinement
        #
        # Gate ≈ 1:
        #     prefer frequency-guided refinement
        # -------------------------------------------------
        gate = self.collaboration_gate(
            torch.cat(
                [
                    local_delta,
                    frequency_delta,
                    context_flat
                ],
                dim=1
            )
        )

        collaborative_delta = (
            (1.0 - gate) * local_delta
            + gate * frequency_delta
        )

        scale = torch.clamp(
            self.collaboration_scale,
            min=0.0,
            max=1.0
        )

        refined_flat = (
            bands_flat
            + scale * collaborative_delta
        )

        refined_bands = refined_flat.reshape(
            n,
            b,
            c,
            h,
            w
        )

        return refined_bands
#--------------------------------------

class CNNMambaGatedFusion(nn.Module):
    """
    Adaptive fusion between local CNN features and global Mamba features.

    The gate determines the relative contribution of each branch:

        fused = gate * CNN + (1 - gate) * Mamba
    """

    def __init__(
        self,
        channels: int = 144
    ):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Conv2d(
                channels * 2,
                channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.GELU(),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.Sigmoid()
        )

        self.refine = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.GELU(),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

    def forward(
        self,
        cnn_features: torch.Tensor,
        mamba_features: torch.Tensor
    ) -> torch.Tensor:

        if cnn_features.shape != mamba_features.shape:
            raise ValueError(
                "CNN and Mamba feature shapes must match. "
                f"CNN={tuple(cnn_features.shape)}, "
                f"Mamba={tuple(mamba_features.shape)}."
            )

        joint = torch.cat(
            [
                cnn_features,
                mamba_features
            ],
            dim=1
        )

        gate = self.gate(joint)

        fused = (
            gate * cnn_features
            + (1.0 - gate) * mamba_features
        )

        fused = fused + self.refine(fused)

        return fused
#---------------------------------------------------
class CNNMambaAverageFusion(nn.Module):
    """
    Parameter-free average fusion between CNN and Mamba features.

    fused = 0.5 * CNN + 0.5 * Mamba
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        cnn_features: torch.Tensor,
        mamba_features: torch.Tensor
    ) -> torch.Tensor:

        if cnn_features.shape != mamba_features.shape:
            raise ValueError(
                "CNN and Mamba feature shapes must match. "
                f"CNN={tuple(cnn_features.shape)}, "
                f"Mamba={tuple(mamba_features.shape)}."
            )

        fused = 0.5 * (
            cnn_features + mamba_features
        )

        return fused
#---------------------------------------------------
class HFAlignmentModule(nn.Module):
    """
    Low-frequency-guided residual alignment with separate
    learnable strengths for mixed-frequency and high-frequency
    wavelet bands.

    The feature extraction, delta prediction, and gating weights
    are shared by all aligned bands. Only the residual alignment
    strength differs between:

        mixed-frequency bands
        high-frequency bands

    Inputs:
        feat_hf:
            Mixed- or high-frequency band feature.
            Shape: (N, C, H, W)

        feat_lf:
            Shared low-frequency structural guide.
            Shape: (N, C, H, W)

        band_group:
            Either:
                "mixed"
                "high"

    Output:
        aligned:
            Residually corrected band feature with the same shape.
    """

    def __init__(
        self,
        channels: int,
        init_mixed_scale: float = 0.10,
        init_high_scale: float = 0.05
    ):
        super().__init__()

        if not 0.0 <= init_mixed_scale <= 1.0:
            raise ValueError(
                "init_mixed_scale must be in [0,1]. "
                f"Received {init_mixed_scale}."
            )

        if not 0.0 <= init_high_scale <= 1.0:
            raise ValueError(
                "init_high_scale must be in [0,1]. "
                f"Received {init_high_scale}."
            )

        self.channels = int(channels)

        # Shared feature-extraction network for both groups.
        self.feat = nn.Sequential(
            nn.Conv2d(
                channels * 2,
                channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(inplace=True)
        )

        # Shared residual-correction prediction.
        self.delta_head = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Shared spatial-channel gate prediction.
        self.gate_head = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Separate learnable alignment strengths.
        self.mixed_scale = nn.Parameter(
            torch.tensor(
                init_mixed_scale,
                dtype=torch.float32
            )
        )

        self.high_scale = nn.Parameter(
            torch.tensor(
                init_high_scale,
                dtype=torch.float32
            )
        )

    def forward(
        self,
        feat_hf: torch.Tensor,
        feat_lf: torch.Tensor,
        band_group: str
    ) -> torch.Tensor:

        if feat_hf.dim() != 4 or feat_lf.dim() != 4:
            raise ValueError(
                "HFAlignmentModule expects two 4D tensors. "
                f"HF={tuple(feat_hf.shape)}, "
                f"LF={tuple(feat_lf.shape)}."
            )

        if feat_hf.shape != feat_lf.shape:
            raise ValueError(
                "Band feature and low-frequency guide must "
                "have identical shapes. "
                f"Band={tuple(feat_hf.shape)}, "
                f"LF={tuple(feat_lf.shape)}."
            )

        if feat_hf.size(1) != self.channels:
            raise ValueError(
                f"Expected {self.channels} feature channels, "
                f"received {feat_hf.size(1)}."
            )

        if band_group not in ("mixed", "high"):
            raise ValueError(
                "band_group must be either 'mixed' or 'high'. "
                f"Received '{band_group}'."
            )

        joint = torch.cat(
            [
                feat_hf,
                feat_lf
            ],
            dim=1
        )

        hidden = self.feat(joint)

        delta = self.delta_head(
            hidden
        )

        gate = torch.sigmoid(
            self.gate_head(hidden)
        )

        if band_group == "mixed":
            scale = torch.clamp(
                self.mixed_scale,
                min=0.0,
                max=1.0
            )
        else:
            scale = torch.clamp(
                self.high_scale,
                min=0.0,
                max=1.0
            )

        aligned = (
            feat_hf
            + scale * gate * delta
        )

        return aligned
# ---------------------------------------------------
class FullResolutionRefinement(nn.Module):
    """
    Lightweight full-resolution residual refinement.

    The module operates after inverse wavelet reconstruction.

    Inputs:
        denoised_coarse:
            Coarse denoised image reconstructed from
            predicted wavelet noise coefficients.
            Shape: (N,1,H,W)

        noisy:
            Original noisy image.
            Shape: (N,1,H,W)

    Processing:
        concat(coarse, noisy)
            -> 3x3 Conv
            -> GELU
            -> GDFN
            -> GDFN
            -> 3x3 Conv

    Output:
        A one-channel residual correction map at the
        original spatial resolution.
    """

    def __init__(
        self,
        hidden_channels: int = 48
    ):
        super().__init__()

        if hidden_channels <= 0:
            raise ValueError(
                "hidden_channels must be positive. "
                f"Received {hidden_channels}."
            )

        self.hidden_channels = int(
            hidden_channels
        )

        # -------------------------------------------------
        # Two image channels enter the refiner:
        #
        #   channel 0: coarse denoised image
        #   channel 1: original noisy image
        # -------------------------------------------------
        self.input_projection = nn.Sequential(
            nn.Conv2d(
                2,
                hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),
            nn.GELU()
        )

        # -------------------------------------------------
        # Lightweight full-resolution feature refinement.
        # -------------------------------------------------
        self.refinement = nn.Sequential(
            ResidualGDFNBlock(
                channels=hidden_channels,
                expansion_factor=2.0,
                residual_scale=0.1,
                bias=False
            ),

            ResidualGDFNBlock(
                channels=hidden_channels,
                expansion_factor=2.0,
                residual_scale=0.1,
                bias=False
            )
        )

        # -------------------------------------------------
        # Predict one residual correction map.
        # -------------------------------------------------
        self.output_projection = nn.Conv2d(
            hidden_channels,
            1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )

        # -------------------------------------------------
        # Conservative initialization:
        #
        # At initialization:
        #
        #     correction ≈ 0
        #
        # Therefore the new model initially behaves like
        # the original coarse reconstruction.
        # -------------------------------------------------
        nn.init.zeros_(
            self.output_projection.weight
        )

        if (
            self.output_projection.bias
            is not None
        ):
            nn.init.zeros_(
                self.output_projection.bias
            )

    def forward(
        self,
        denoised_coarse: torch.Tensor,
        noisy: torch.Tensor
    ) -> torch.Tensor:

        if (
            denoised_coarse.dim() != 4
            or noisy.dim() != 4
        ):
            raise ValueError(
                "FullResolutionRefinement expects "
                "two 4D tensors."
            )

        if denoised_coarse.shape != noisy.shape:
            raise ValueError(
                "denoised_coarse and noisy must have "
                "identical shapes. "
                f"Coarse={tuple(denoised_coarse.shape)}, "
                f"Noisy={tuple(noisy.shape)}."
            )

        if denoised_coarse.size(1) != 1:
            raise ValueError(
                "The current full-resolution refinement "
                "expects one-channel CT images."
            )

        refinement_input = torch.cat(
            [
                denoised_coarse,
                noisy
            ],
            dim=1
        )

        features = self.input_projection(
            refinement_input
        )

        features = self.refinement(
            features
        )

        correction = self.output_projection(
            features
        )

        return correction
#--------------------------------------------------
class MedMultiwaveletDenoisingGenerator(nn.Module):

    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        wavelet_type: str = "ghm",
        band_feature_channels: int = 40,
        feature_channels: int = 144,
        cnn_blocks: int = 6,
        mamba_blocks: int = 6,
        branch_mode: str = "dual",
        fusion_type: str = "gated",
        use_alignment: bool = True,
        use_band_refinement: bool = True,
        use_full_res_refinement: bool = True,
    ):
        super().__init__()

        if in_ch != 1:
            raise ValueError(
                "The current shared-band implementation expects "
                "single-channel CT images. "
                f"Received in_ch={in_ch}."
            )

        self.in_ch = in_ch
        self.out_ch = out_ch
        
        self.wavelet_type = wavelet_type.lower()
        
        self.branch_mode = branch_mode.lower()
        self.fusion_type = fusion_type.lower()
        
        self.use_alignment = bool(use_alignment)
        self.use_band_refinement = bool(
            use_band_refinement
        )
        self.use_full_res_refinement = bool(
            use_full_res_refinement
        )

        if self.branch_mode not in (
            "cnn",
            "mamba",
            "dual"
        ):
            raise ValueError(
                "branch_mode must be 'cnn', 'mamba', or 'dual'. "
                f"Received '{branch_mode}'."
            )
        
        if self.branch_mode == "dual":
        
            if self.fusion_type not in (
                "average",
                "gated"
            ):
                raise ValueError(
                    "For branch_mode='dual', fusion_type must be "
                    "'average' or 'gated'. "
                    f"Received '{fusion_type}'."
                )
        
        else:
        
            if self.fusion_type != "none":
                raise ValueError(
                    "For CNN-only or Mamba-only experiments, "
                    "fusion_type must be 'none'. "
                    f"Received '{fusion_type}'."
                )
               

        

        self.band_feature_channels = band_feature_channels
        self.feature_channels = feature_channels

        # -------------------------------------------------
        # 1. wavelet decomposition
        #
        # Input:
        #   (N,1,H,W)
        #
        # Output:
        #   (N,16,H/2,W/2)
        # -------------------------------------------------
        # -------------------------------------------------
        # 1. Wavelet decomposition and reconstruction
        # -------------------------------------------------
        if self.wavelet_type == "ghm":
        
            self.dwt = GHMFullMWT2D()
        
            self.idwt = GHMInverseMWT2D(
                H_mats=self.dwt.H_mats,
                G_mats=self.dwt.G_mats
            )
        
            self.expected_num_bands = 16
        
        elif self.wavelet_type == "haar":
        
            self.dwt = HaarDWT2D()
            self.idwt = HaarInverseDWT2D()
        
            self.expected_num_bands = 4
        
        else:
            raise ValueError(
                "Unsupported wavelet_type. "
                "Expected 'ghm' or 'haar', "
                f"received '{wavelet_type}'."
            )
        # -------------------------------------------------
        # GHM band organization
        #
        # Final GHM band ordering:
        #
        #  0: W-L0 × H-L0
        #  1: W-L0 × H-L1
        #  2: W-L0 × H-H0
        #  3: W-L0 × H-H1
        #
        #  4: W-L1 × H-L0
        #  5: W-L1 × H-L1
        #  6: W-L1 × H-H0
        #  7: W-L1 × H-H1
        #
        #  8: W-H0 × H-L0
        #  9: W-H0 × H-L1
        # 10: W-H0 × H-H0
        # 11: W-H0 × H-H1
        #
        # 12: W-H1 × H-L0
        # 13: W-H1 × H-L1
        # 14: W-H1 × H-H0
        # 15: W-H1 × H-H1

        # -------------------------------------------------
        # Wavelet-dependent band organization
        # -------------------------------------------------
        if self.wavelet_type == "ghm":
        
            # Low-low GHM components.
            self.low_band_indices = (
                0,
                1,
                4,
                5
            )
        
            # Mixed-frequency components:
            # one low-direction component and one
            # high-direction component.
            self.mixed_band_indices = (
                2,
                3,
                6,
                7,
                8,
                9,
                12,
                13
            )
        
            # High-high GHM components.
            self.high_band_indices = (
                10,
                11,
                14,
                15
            )
        
        elif self.wavelet_type == "haar":
        
            # Haar ordering:
            # 0 = LL
            # 1 = LH
            # 2 = HL
            # 3 = HH
            self.low_band_indices = (
                0,
            )
        
            # LH and HL are mixed-frequency bands.
            self.mixed_band_indices = (
                1,
                2
            )
        
            # HH is the high-high band.
            self.high_band_indices = (
                3,
            )
        # -------------------------------------------------
        # 2. Shared encoder applied independently to every band
        #
        # Output:
        #   (N,16,F,H/2,W/2)
        # -------------------------------------------------
        self.band_encoder = SharedBandEncoder(
            feature_channels=band_feature_channels
        )

        # -------------------------------------------------
        #
        # -------------------------------------------------
        

        
        # -------------------------------------------------
        # 4. Common frequency stem
        #
        # For F=32:
        #   input channels = 3 × 2 × 32 = 192
        #   output channels = 96
        # -------------------------------------------------
        self.band_projection = nn.Sequential(
            nn.Conv2d(
                band_feature_channels,
                feature_channels,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.GELU(),
        
            nn.Conv2d(
                feature_channels,
                feature_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.GELU()
        )

        # -------------------------------------------------
        # -------------------------------------------------
        # Branch construction
        # -------------------------------------------------
        
        if self.branch_mode in (
            "cnn",
            "dual"
        ):
            self.cnn_branch = LocalCNNBranch(
                channels=feature_channels,
                num_blocks=cnn_blocks
            )
        else:
            self.cnn_branch = None
        
        
        if self.branch_mode in (
            "mamba",
            "dual"
        ):
            self.mamba_branch = GlobalMambaBranch(
                channels=feature_channels,
                num_blocks=mamba_blocks,
                d_state=16,
                d_conv=4,
                expand=2
            )
        else:
            self.mamba_branch = None

        # -------------------------------------------------
        # -------------------------------------------------
        # 6. CNN-Mamba fusion
        # -------------------------------------------------
        # -------------------------------------------------
        # CNN-Mamba fusion
        # Used only for dual-branch experiments.
        # -------------------------------------------------
        
        if self.branch_mode == "dual":
        
            if self.fusion_type == "gated":
        
                self.fusion = CNNMambaGatedFusion(
                    channels=feature_channels
                )
        
            elif self.fusion_type == "average":
        
                self.fusion = CNNMambaAverageFusion()
        
        else:
        
            self.fusion = None
        # -------------------------------------------------
        # 7. Low-frequency structural guide
        #
        # Four GHM low-low band features are concatenated:
        #
        #   4 × feature_channels
        #
        # and projected into one shared structural guide:
        #
        #   feature_channels
        # -------------------------------------------------
        if self.use_alignment:

            self.low_frequency_guide = nn.Sequential(
                nn.Conv2d(
                    feature_channels * len(
                        self.low_band_indices
                    ),
                    feature_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0
                ),
                nn.GELU(),
        
                nn.Conv2d(
                    feature_channels,
                    feature_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.GELU()
            )
        
        else:
        
            self.low_frequency_guide = None

        # -------------------------------------------------
        # 8. Shared low-frequency-guided alignment
        #
        # The same alignment weights are applied to all
        # mixed/high-frequency GHM bands. This preserves
        # parameter efficiency and avoids giving each band
        # an independent large correction network.
        # -------------------------------------------------
        if self.use_alignment:

            self.hf_alignment = HFAlignmentModule(
                channels=feature_channels,
                init_mixed_scale=0.10,
                init_high_scale=0.05
            )
        
        else:
        
            self.hf_alignment = None
        
        # -------------------------------------------------
        if self.use_band_refinement:

            self.band_refinement = (
                AdaptiveGHMFrequencyCollaborativeRefinement(
                    channels=feature_channels,
                    low_band_indices=self.low_band_indices,
                    mixed_band_indices=self.mixed_band_indices,
                    high_band_indices=self.high_band_indices,
                    expansion_factor=2.0,
                    init_scale=0.10,
                    bias=False
                )
            )
        
        else:
        
            self.band_refinement = None
        coefficient_hidden_channels = 40
        
        self.noise_band_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        feature_channels,
                        coefficient_hidden_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1
                    ),
                    nn.GELU(),
        
                    nn.Conv2d(
                        coefficient_hidden_channels,
                        1,
                        kernel_size=3,
                        stride=1,
                        padding=1
                    )
                )
                for _ in range(
                    self.expected_num_bands
                )
            ]
        )

        # -------------------------------------------------
        # 10. Full-resolution spatial refinement
        #
        # This is applied only after inverse wavelet
        # reconstruction and therefore operates directly
        # at the original image resolution H x W.
        # -------------------------------------------------
        if self.use_full_res_refinement:

            self.full_res_refinement = (
                FullResolutionRefinement(
                    hidden_channels=48
                )
            )
        
            self.full_res_scale = nn.Parameter(
                torch.tensor(
                    0.08,
                    dtype=torch.float32
                )
            )
        
        else:
        
            self.full_res_refinement = None
            self.register_parameter(
                "full_res_scale",
                None
            )

           

    def forward(
        self,
        noisy: torch.Tensor
    ) -> torch.Tensor:

        if noisy.dim() != 4:
            raise ValueError(
                "MedMultiwaveletDenoisingGenerator expects "
                f"(N,C,H,W), received {tuple(noisy.shape)}."
            )

        n, c, h, w = noisy.shape

        if c != self.in_ch:
            raise ValueError(
                f"Expected {self.in_ch} input channels, "
                f"received {c}."
            )

        if h % 2 != 0 or w % 2 != 0:
            raise ValueError(
                "Input height and width must be divisible by 2. "
                f"Received H={h}, W={w}."
            )

        # -------------------------------------------------
        # 1. wavelet decomposition
        # -------------------------------------------------
        bands = self.dwt(noisy)

        if bands.size(1) != self.expected_num_bands:
            raise RuntimeError(
                f"{self.wavelet_type.upper()} decomposition must "
                f"produce {self.expected_num_bands} bands for a "
                "one-channel input. "
                f"Received {bands.size(1)}."
            )

        # -------------------------------------------------
        # 2. Shared per-band encoding
        #
        # encoded_bands:
        #   (N,16,F,H/2,W/2)
        # -------------------------------------------------
        encoded_bands = self.band_encoder(
            bands
        )

        # -------------------------------------------------
        # ---------------------------------------
        # Flatten the band dimension
        #
        # (N,16,32,H/2,W/2)
        # ->
        # (N*16,32,H/2,W/2)
        # ---------------------------------------
        encoded_flat = encoded_bands.reshape(
            n * self.expected_num_bands,
            self.band_feature_channels,
            h // 2,
            w // 2
        )
        projected_flat = self.band_projection(
            encoded_flat
        )
        # -------------------------------------------------
        # -------------------------------------------------
        # Branch processing
        # -------------------------------------------------
        
        if self.branch_mode == "cnn":
        
            fused_flat = self.cnn_branch(
                projected_flat
            )
        
        elif self.branch_mode == "mamba":
        
            fused_flat = self.mamba_branch(
                projected_flat
            )
        
        elif self.branch_mode == "dual":
        
            cnn_flat = self.cnn_branch(
                projected_flat
            )
        
            mamba_flat = self.mamba_branch(
                projected_flat
            )
        
            fused_flat = self.fusion(
                cnn_flat,
                mamba_flat
            )
        
        else:
        
            raise RuntimeError(
                f"Unexpected branch_mode: "
                f"{self.branch_mode}"
            )

        # -------------------------------------------------
        # ---------------------------------------
        # Restore band dimension
        #
        # (N*16,96,H/2,W/2)
        # ->
        # (N,16,96,H/2,W/2)
        # ---------------------------------------
        fused_bands = fused_flat.reshape(
            n,
            self.expected_num_bands,
            self.feature_channels,
            h // 2,
            w // 2
        )

        # -------------------------------------------------
        # 8. Build the shared low-frequency structural guide
        #
        # Selected GHM low-low features:
        #     indices = (0, 1, 4, 5)
        #
        # Each feature:
        #     (N, C, H/2, W/2)
        #
        # After concatenation:
        #     (N, 4C, H/2, W/2)
        # -------------------------------------------------
        low_frequency_guide = None

        if self.use_alignment:
        
            low_feature_list = [
                fused_bands[
                    :,
                    band_index,
                    :,
                    :,
                    :
                ]
                for band_index in self.low_band_indices
            ]
        
            low_features_concat = torch.cat(
                low_feature_list,
                dim=1
            )
        
            low_frequency_guide = (
                self.low_frequency_guide(
                    low_features_concat
                )
            )
        # -------------------------------------------------
        # Predict one wavelet noise coefficient map per band
        # -------------------------------------------------
        predicted_band_list = []
        # -------------------------------------------------
        # 9. Align mixed/high-frequency GHM bands
        #
        # Low-low bands remain unchanged.
        # Mixed/high-frequency bands receive residual
        # refinement guided by the low-frequency structure.
        # -------------------------------------------------
        aligned_band_feature_list = []

        for band_index in range(
            self.expected_num_bands
        ):
            band_features = fused_bands[
                :,
                band_index,
                :,
                :,
                :
            ]

            if self.use_alignment:

                if band_index in self.mixed_band_indices:
            
                    band_features = self.hf_alignment(
                        feat_hf=band_features,
                        feat_lf=low_frequency_guide,
                        band_group="mixed"
                    )
            
                elif band_index in self.high_band_indices:
            
                    band_features = self.hf_alignment(
                        feat_hf=band_features,
                        feat_lf=low_frequency_guide,
                        band_group="high"
                    )

            aligned_band_feature_list.append(
                band_features
            )

        aligned_fused_bands = torch.stack(
            aligned_band_feature_list,
            dim=1
        )

        # -------------------------------------------------
        # Shared GDFN refinement for all aligned bands
        #
        # (N,B,C,H/2,W/2)
        #       ->
        # (N*B,C,H/2,W/2)
        #       ->
        # shared refinement
        #       ->
        # (N,B,C,H/2,W/2)
        # -------------------------------------------------
        if self.use_band_refinement:

            refined_fused_bands = self.band_refinement(
                aligned_fused_bands
            )
        
        else:
        
            refined_fused_bands = (
                aligned_fused_bands
            )
        
        for band_index, head in enumerate(
            self.noise_band_heads
        ):
            band_features = refined_fused_bands[
                :,
                band_index,
                :,
                :,
                :
            ]
        
            predicted_band = head(
                band_features
            )
        
            predicted_band_list.append(
                predicted_band
            )
        
        # -------------------------------------------------
        # Reassemble the 16 predicted wavelet coefficient maps
        #
        # Output:
        #   (N,16,H/2,W/2)
        # -------------------------------------------------
        predicted_noise_bands = torch.cat(
            predicted_band_list,
            dim=1
        )
        
        # -------------------------------------------------
        # Reconstruct predicted spatial noise using
        # the selected inverse wavelet transform
        #
        # Output:
        #   (N,1,H,W)
        # -------------------------------------------------
        predicted_noise = self.idwt(
            predicted_noise_bands
        )
        
        # -------------------------------------------------
        # 10. Coarse residual denoising
        #
        # This is exactly the output of the original
        # baseline architecture.
        # -------------------------------------------------
        denoised_coarse = (
            noisy
            - predicted_noise
        )

        if self.use_full_res_refinement:

            full_res_correction = (
                self.full_res_refinement(
                    denoised_coarse=denoised_coarse,
                    noisy=noisy
                )
            )
        
            full_res_scale = torch.clamp(
                self.full_res_scale,
                min=0.0,
                max=1.0
            )
        
            denoised = (
                denoised_coarse
                + full_res_scale
                * full_res_correction
            )
        
        else:
        
            denoised = denoised_coarse
        
        return denoised

#------------------------------------------------------------------------------
#--------------------------------------------------------------
class LSGANPatchDiscriminator(nn.Module):
    """discriminator"""
    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()

        # --- BEGIN: ---
        def block(ic, oc, k=4, s=2, p=1):
            layers = [nn.Conv2d(ic, oc, k, s, p)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        # --- END ---

        layers = []
        ch = base_ch
        layers += block(in_ch, ch)
        layers += block(ch, ch*2)
        layers += block(ch*2, ch*4)
        layers += block(ch*4, ch*8)
        layers += [nn.Conv2d(ch*8, 1, kernel_size=4, stride=1, padding=0)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)  # (B,1,h,w)

#--------------------------------------------------------------------------
class SharedBandEncoder(nn.Module):
    """
    Encode every wavelet band independently using the same CNN weights.

    Input:
        bands: (N, B, H, W)

    Output:
        encoded: (N, B, F, H, W)

    where:
        B = number of wavelet bands
        F = feature channels generated for each band
    """

    def __init__(
        self,
        feature_channels: int = 40
    ):
        super().__init__()

        self.feature_channels = feature_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(
                1,
                feature_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.GELU(),

            nn.Conv2d(
                feature_channels,
                feature_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

    def forward(
        self,
        bands: torch.Tensor
    ) -> torch.Tensor:

        if bands.dim() != 4:
            raise ValueError(
                "SharedBandEncoder expects bands with shape "
                f"(N,B,H,W), received {tuple(bands.shape)}."
            )

        n, b, h, w = bands.shape

        # Treat every band as an independent one-channel image.
        bands_flat = bands.reshape(
            n * b,
            1,
            h,
            w
        )

        encoded_flat = self.encoder(
            bands_flat
        )

        encoded = encoded_flat.reshape(
            n,
            b,
            self.feature_channels,
            h,
            w
        )

        return encoded
#---------------------------------------------------

#-------------------------------------------------------------
class LocalCNNBranch(nn.Module):
    """
    CNN branch for local noise patterns, edges, and fine structures.

    Input/output:
        (N, C, H, W)
    """

    def __init__(
        self,
        channels: int = 144,
        num_blocks: int = 4
    ):
        super().__init__()

        self.blocks = nn.Sequential(
            *[
                ConvResBlock(channels)
                for _ in range(num_blocks)
            ]
        )

        self.conv_after = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        residual = x

        features = self.blocks(x)
        features = self.conv_after(features)

        return residual + features
#-----------------------------------------------
class GlobalMambaBranch(nn.Module):
    """
    Mamba branch for global spatial context and long-range dependencies.

    Input/output:
        (N, C, H, W)
    """

    def __init__(
        self,
        channels: int = 144,
        num_blocks: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2
    ):
        super().__init__()

        self.blocks = nn.Sequential(
            *[
                MambaBlock2D(
                    dim=channels,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand
                )
                for _ in range(num_blocks)
            ]
        )

        self.conv_after = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        residual = x

        features = self.blocks(x)
        features = self.conv_after(features)

        return residual + features
#-----------------------------------------------------------

#-------------------------------------------------------------------------------
# -----------------------------
# Losses
# -----------------------------

#------------------------------------------------------
def lsgan_d_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """LSGAN discriminator loss: 0.5*( (D(real)-1)^2 + (D(fake)-0)^2 )"""
    return 0.5 * (
        F.mse_loss(d_real, torch.ones_like(d_real)) +
        F.mse_loss(d_fake, torch.zeros_like(d_fake))
    )

def lsgan_g_loss(d_fake: torch.Tensor) -> torch.Tensor:
    """LSGAN generator loss: 0.5*(D(fake)-1)^2"""
    return 0.5 * F.mse_loss(d_fake, torch.ones_like(d_fake))
#-------------------------------------------------------------------


# ----------------------------------------------------------------
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
    def forward(self, x, y):
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
#-------------------------------------------------------------------------------
class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        if ssim is None:
            raise ImportError("Install pytorch-msssim")

        return 1.0 - ssim(
            x,
            y,
            data_range=1.0,
            size_average=True
        )

#----------------------------------------------------------------------
#----------------------------------------------
def _reflect_pad_if_needed(img, size):
    h, w = img.shape[:2]
    if h >= size and w >= size:
        return img
    pad_h = max(0, size - h)
    pad_w = max(0, size - w)
    return np.pad(
        img,
        ((pad_h//2, pad_h - pad_h//2),
         (pad_w//2, pad_w - pad_w//2)),
        mode="reflect"
    )
    

def _augment_clean(
    clean: np.ndarray,
    use_hflip: bool = True,
    use_vflip: bool = True,
    use_rot: bool = True
) -> np.ndarray:

    if use_hflip and random.random() < 0.5:
        clean = np.fliplr(clean)

    if use_vflip and random.random() < 0.5:
        clean = np.flipud(clean)

    if use_rot and random.random() < 0.5:
        k = random.choice([1, 2, 3])
        clean = np.rot90(clean, k)

    return clean.copy()
# -----------------------------
# Synthetic Gaussian CT denoising dataset
# -----------------------------
class CTGaussianDenoisingDataset(Dataset):
    """
    On-the-fly synthetic Gaussian denoising dataset.

    The dataset reads clean grayscale CT images and dynamically
    generates noisy-clean training pairs.

    Training:
        - Random or full-size clean crop.
        - Geometric augmentation.
        - Random Gaussian noise level sampled from sigma_range.

    Validation:
        - Center crop.
        - Fixed Gaussian noise level.
        - Deterministic noise generated from val_seed + image index.

    Returns:
        noisy_t: Tensor with shape (1, patch_size, patch_size)
        clean_t: Tensor with shape (1, patch_size, patch_size)

    Both tensors are in the normalized intensity range [0,1].
    """

    def __init__(
        self,
        root: str,
        patch_size: int = 352,
        sigma_range=(0.0, 25.0),
        validation_sigma: float = 25.0,
        training: bool = True,
        use_hflip: bool = True,
        use_vflip: bool = True,
        use_rot: bool = True,
        val_seed: int = 123,
        exts=(".png", ".jpg", ".jpeg", ".tif", ".tiff")
    ):
        super().__init__()

        # -------------------------------------------------
        # Collect image paths
        # -------------------------------------------------
        self.files = []

        for dp, _, fns in os.walk(root):
            for filename in fns:
                extension = os.path.splitext(filename)[1].lower()

                if extension in exts:
                    self.files.append(
                        os.path.join(dp, filename)
                    )

        self.files.sort()

        if len(self.files) == 0:
            raise RuntimeError(
                f"No supported CT images were found in: {root}"
            )

        # -------------------------------------------------
        # Configuration
        # -------------------------------------------------
        self.patch_size = int(patch_size)
        self.training = bool(training)

        self.use_hflip = bool(use_hflip)
        self.use_vflip = bool(use_vflip)
        self.use_rot = bool(use_rot)

        self.val_seed = int(val_seed)

        # -------------------------------------------------
        # Noise configuration
        #
        # Sigma values are supplied in the conventional
        # image-intensity scale [0,255].
        # -------------------------------------------------
        if len(sigma_range) != 2:
            raise ValueError(
                "sigma_range must contain exactly two values: "
                "(minimum_sigma, maximum_sigma)."
            )

        self.sigma_min = float(sigma_range[0])
        self.sigma_max = float(sigma_range[1])

        self.validation_sigma = float(validation_sigma)

        if self.sigma_min < 0:
            raise ValueError(
                "Minimum Gaussian noise sigma cannot be negative."
            )

        if self.sigma_max < self.sigma_min:
            raise ValueError(
                "sigma_range maximum must be greater than or "
                "equal to its minimum."
            )

        if self.validation_sigma < 0:
            raise ValueError(
                "validation_sigma cannot be negative."
            )

        # GHM requires even spatial dimensions because it
        # reduces H and W by a factor of two.
        if self.patch_size % 2 != 0:
            raise ValueError(
                "patch_size must be divisible by 2 for the "
                f"GHM decomposition. Received {self.patch_size}."
            )

    def __len__(self):
        return len(self.files)

    def _crop_clean(
        self,
        clean: np.ndarray
    ) -> np.ndarray:
        """
        Extract a square patch from the clean CT image.

        Training:
            random crop.

        Validation:
            deterministic center crop.
        """

        # Pad only when the input image is smaller than patch_size.
        clean = _reflect_pad_if_needed(
            clean,
            self.patch_size
        )

        h, w = clean.shape[:2]

        if self.training:
            top = random.randint(
                0,
                h - self.patch_size
            )

            left = random.randint(
                0,
                w - self.patch_size
            )

        else:
            top = (
                h - self.patch_size
            ) // 2

            left = (
                w - self.patch_size
            ) // 2

        clean_patch = clean[
            top:top + self.patch_size,
            left:left + self.patch_size
        ]

        if clean_patch.shape != (
            self.patch_size,
            self.patch_size
        ):
            raise RuntimeError(
                "Clean patch size mismatch. "
                f"Received {clean_patch.shape}, expected "
                f"({self.patch_size}, {self.patch_size})."
            )

        return clean_patch

    def _generate_training_noise(
        self,
        clean_t: torch.Tensor
    ):
        """
        Generate training Gaussian noise using NumPy default_rng,
        consistent with the final test noise formulation.
    
        A new random realization is generated every time an image
        is sampled during training.
        """
    
        sigma_255 = random.uniform(
            self.sigma_min,
            self.sigma_max
        )
    
        sigma_normalized = (
            float(sigma_255) / 255.0
        )
    
        # Deterministic across complete repeated experiments
        # because np.random is seeded by seed_everything().
        random_seed = int(
            np.random.randint(
                0,
                2**32 - 1
            )
        )
    
        rng = np.random.default_rng(
            random_seed
        )
    
        noise_np = rng.normal(
            loc=0.0,
            scale=sigma_normalized,
            size=tuple(clean_t.shape)
        ).astype(np.float32)
    
        noise_t = torch.from_numpy(
            noise_np
        )
    
        return (
            noise_t,
            float(sigma_255)
        )
    def _generate_validation_noise(
        self,
        clean_t: torch.Tensor,
        idx: int
    ):
        """
        Deterministic Gaussian noise identical in formulation
        to the final testing protocol.
    
        sigma is expressed on the [0,255] scale.
    
        Noise realization:
            NumPy default_rng(val_seed + image_index)
        """
    
        sigma_255 = float(
            self.validation_sigma
        )
    
        sigma_normalized = (
            sigma_255 / 255.0
        )
    
        image_seed = (
            self.val_seed
            + int(idx)
        )
    
        rng = np.random.default_rng(
            image_seed
        )
    
        noise_np = rng.normal(
            loc=0.0,
            scale=sigma_normalized,
            size=tuple(clean_t.shape)
        ).astype(np.float32)
    
        noise_t = torch.from_numpy(
            noise_np
        )
    
        return (
            noise_t,
            sigma_255
        )
    def __getitem__(
        self,
        idx: int
    ):
        clean_path = self.files[idx]
    
        # -------------------------------------------------
        # 1. Read clean grayscale CT image
        # -------------------------------------------------
        clean = cv2.imread(
            clean_path,
            cv2.IMREAD_GRAYSCALE
        )

        if clean is None:
            raise FileNotFoundError(
                f"Could not read CT image: {clean_path}"
            )

        # -------------------------------------------------
        # 2. Extract clean patch
        # -------------------------------------------------
        clean = self._crop_clean(clean)

        # -------------------------------------------------
        # 3. Geometric augmentation
        #
        # Applied only to training data.
        # Noise is generated after augmentation.
        # -------------------------------------------------
        if self.training:
            clean = _augment_clean(
                clean,
                use_hflip=self.use_hflip,
                use_vflip=self.use_vflip,
                use_rot=self.use_rot
            )
        else:
            clean = clean.copy()

        # -------------------------------------------------
        # 4. Normalize clean image to [0,1]
        #
        # Input PNG values:
        #     [0,255]
        #
        # Tensor values:
        #     [0,1]
        # -------------------------------------------------
        clean_t = torch.from_numpy(
            clean.astype(np.float32) / 255.0
        ).unsqueeze(0)

        # clean_t shape:
        # (1, patch_size, patch_size)

        # -------------------------------------------------
        # 5. Generate synthetic Gaussian noise
        # -------------------------------------------------
        if self.training:
            noise_t, _ = (
                self._generate_training_noise(
                    clean_t
                )
            )
        else:
            noise_t, _ = (
                self._generate_validation_noise(
                    clean_t,
                    idx
                )
            )

        # -------------------------------------------------
        # 6. Form noisy image
        # -------------------------------------------------
        noisy_t = clean_t + noise_t

        # Keep the simulated image in the valid normalized
        # image intensity range.
        noisy_t = torch.clamp(
            noisy_t,
            min=0.0,
            max=1.0
        )

        # -------------------------------------------------
        # 7. Final validation
        # -------------------------------------------------
        if noisy_t.shape != clean_t.shape:
            raise RuntimeError(
                "Noisy and clean tensors must have identical "
                f"shapes. Noisy={tuple(noisy_t.shape)}, "
                f"clean={tuple(clean_t.shape)}."
            )

        if not torch.isfinite(noisy_t).all():
            raise RuntimeError(
                f"Non-finite values found in noisy image: {clean_path}"
            )

        if not torch.isfinite(clean_t).all():
            raise RuntimeError(
                f"Non-finite values found in clean image: {clean_path}"
            )

        # Training and validation loops can initially keep
        # the same two-value unpacking structure:
        #
        # for noisy, clean in loader:
        return noisy_t, clean_t
#---------------------------------------------------------


# -----------------------------
# Training / Validation loops
# -----------------------------

@dataclass
class TrainConfig:
    # =====================================================
    # Data paths
    # =====================================================
    train_dir: str
    val_dir: str

    # =====================================================
    experiment_name: str = "experiment"

    wavelet_type: str = "ghm"
    
    # Branch configuration:
    # "cnn"   -> CNN branch only
    # "mamba" -> Mamba branch only
    # "dual"  -> CNN + Mamba
    branch_mode: str = "dual"
    
    # Fusion is used only when branch_mode="dual".
    # "none"    -> no fusion
    # "average" -> simple average fusion
    # "gated"   -> adaptive gated fusion
    fusion_type: str = "gated"
    
    use_alignment: bool = True
    use_band_refinement: bool = True
    use_full_res_refinement: bool = True
    
    save_root: str = "./runs/denoising"

    # =====================================================
    # Denoising data protocol
    # =====================================================
    patch_size: int = 88

    train_sigma_min: float = 0.0
    train_sigma_max: float = 25.0
    validation_sigma: float = 25.0

    # =====================================================
    # DataLoader
    # =====================================================
    batch_size: int = 2
    num_workers: int = 0
    pin_memory: bool = True
    drop_last: bool = True

    # =====================================================
    # Training
    # =====================================================
    epochs: int = 30

    lr_g: float = 1.5e-4
    lr_d: float = 1e-4

    beta1: float = 0.9
    beta2: float = 0.999

    # =====================================================
    # Loss weights
    # =====================================================
    content_weight: float = 1.0

    charbonnier_weight: float = 1.15
    ssim_weight: float = 0.15

    adv_weight: float = 0.002
    gan_warmup_epochs: int = 1

    # =====================================================
    # Discriminator and EMA
    # =====================================================
    disc_channels: int = 64
    ema_decay: float = 0.999

    # =====================================================
    # Reproducibility
    # =====================================================
    seed: int = 123
    deterministic: bool = True
    validation_seed: int = 123

    # =====================================================
    # Device
    # =====================================================
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    @property
    def save_dir(self) -> str:
        """
        Produce an isolated directory for every experiment.

        Examples:
            ./runs/denoising/GHM_Denoising
            ./runs/denoising/Haar_Denoising
        """
        return os.path.join(
            self.save_root,
            self.experiment_name
        )
    


@torch.no_grad()
def validate_denoising(
    generator: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluate the denoising generator using noisy-clean image pairs.

    Args:
        generator:
            Denoising model.

        loader:
            Validation DataLoader returning:
                noisy, clean

        device:
            CPU or CUDA device.

    Returns:
        Mean PSNR and mean SSIM over the validation set.
    """
    generator.eval()

    total_psnr = 0.0
    total_ssim = 0.0
    total_images = 0

    for noisy, clean in loader:
        noisy = noisy.to(
            device,
            non_blocking=True
        )

        clean = clean.to(
            device,
            non_blocking=True
        )

        denoised = generator(noisy).clamp(0.0, 1.0)

        batch_psnr, batch_ssim = (
            batch_psnr_ssim_basic_sr(
                denoised,
                clean
            )
        )

        batch_size = noisy.size(0)

        total_psnr += batch_psnr * batch_size
        total_ssim += batch_ssim * batch_size
        total_images += batch_size

    if total_images == 0:
        raise RuntimeError(
            "The validation DataLoader returned no images."
        )

    mean_psnr = total_psnr / total_images
    mean_ssim = total_ssim / total_images

    return float(mean_psnr), float(mean_ssim)


def train(cfg: TrainConfig):
    # =====================================================
    # Reproducibility
    # =====================================================
    seed_everything(
        seed=cfg.seed,
        deterministic=cfg.deterministic
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(cfg.seed)

    print("======================================")
    print("Denoising experiment configuration")
    print("======================================")
    print("Experiment:", cfg.experiment_name)
    print("Wavelet:", cfg.wavelet_type)

    print("Branch mode:", cfg.branch_mode)
    print("Fusion type:", cfg.fusion_type)
    print("Alignment enabled:", cfg.use_alignment)
    print(
        "Frequency-collaborative refinement:",
        cfg.use_band_refinement
    )
    print(
        "Full-resolution refinement:",
        cfg.use_full_res_refinement
    )
    print("Patch size:", cfg.patch_size)
    print(
        "Training sigma range:",
        f"[{cfg.train_sigma_min}, {cfg.train_sigma_max}]"
    )
    print("Validation sigma:", cfg.validation_sigma)
    print("Seed:", cfg.seed)
    print("Deterministic:", cfg.deterministic)
    print("Save directory:", cfg.save_dir)

    # =====================================================
    # Device information
    # =====================================================
    if torch.cuda.is_available():
        print(
            "GPU Detected:",
            torch.cuda.get_device_name(0),
            "- CUDA:",
            torch.version.cuda
        )
    else:
        print("No GPU detected. Training will run on CPU.")

    os.makedirs(
        cfg.save_dir,
        exist_ok=True
    )

    

    # =====================================================
    # Denoising datasets
    # =====================================================
    train_dataset = CTGaussianDenoisingDataset(
        root=cfg.train_dir,
        patch_size=cfg.patch_size,
        sigma_range=(
            cfg.train_sigma_min,
            cfg.train_sigma_max
        ),
        validation_sigma=cfg.validation_sigma,
        training=True,
        use_hflip=True,
        use_vflip=True,
        use_rot=True,
        val_seed=cfg.validation_seed
    )

    val_dataset = CTGaussianDenoisingDataset(
        root=cfg.val_dir,
        patch_size=cfg.patch_size,
        sigma_range=(
            cfg.train_sigma_min,
            cfg.train_sigma_max
        ),
        validation_sigma=cfg.validation_sigma,
        training=False,
        use_hflip=False,
        use_vflip=False,
        use_rot=False,
        val_seed=cfg.validation_seed
    )

    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))

    # =====================================================
    # Initial DataLoaders
    # =====================================================
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last,
        worker_init_fn=seed_worker,
        generator=loader_generator
    )

    
    # =====================================================
    # Dataset batch sanity check
    # =====================================================
    noisy_batch, clean_batch = next(
        iter(train_loader)
    )

    print("\n====== Denoising Batch Sanity Check ======")
    print("Noisy batch shape:", tuple(noisy_batch.shape))
    print("Clean batch shape:", tuple(clean_batch.shape))

    print(
        "Noisy range:",
        float(noisy_batch.min()),
        float(noisy_batch.max())
    )

    print(
        "Clean range:",
        float(clean_batch.min()),
        float(clean_batch.max())
    )

    expected_shape = (
        cfg.batch_size,
        1,
        cfg.patch_size,
        cfg.patch_size
    )

    assert noisy_batch.shape == clean_batch.shape, (
        "Noisy and clean batch shapes must match. "
        f"Noisy={tuple(noisy_batch.shape)}, "
        f"clean={tuple(clean_batch.shape)}."
    )

    assert tuple(noisy_batch.shape) == expected_shape, (
        f"Unexpected batch shape: {tuple(noisy_batch.shape)}. "
        f"Expected {expected_shape}."
    )

    assert torch.isfinite(noisy_batch).all(), (
        "Non-finite values found in noisy batch."
    )

    assert torch.isfinite(clean_batch).all(), (
        "Non-finite values found in clean batch."
    )

    print("✅ Denoising batch sanity check passed.")

    # =====================================================
    # Generator-dataset integration sanity check
    # =====================================================
    with torch.no_grad():
    
        noisy_dbg = noisy_batch.to(
            cfg.device,
            non_blocking=cfg.pin_memory
        )
    
        clean_dbg = clean_batch.to(
            cfg.device,
            non_blocking=cfg.pin_memory
        )
    
        gen_tmp = MedMultiwaveletDenoisingGenerator(
            in_ch=1,
            out_ch=1,
            wavelet_type=cfg.wavelet_type,
            band_feature_channels=40,
            feature_channels=144,
            cnn_blocks=6,
            mamba_blocks=6,
            branch_mode=cfg.branch_mode,
            fusion_type=cfg.fusion_type,
            use_alignment=cfg.use_alignment,
            use_band_refinement=cfg.use_band_refinement,
            use_full_res_refinement=cfg.use_full_res_refinement
        ).to(cfg.device)
    
        gen_tmp.eval()

        # -------------------------------------------------
        # Verify requested architecture
        # -------------------------------------------------
        if cfg.branch_mode == "cnn":
        
            assert gen_tmp.cnn_branch is not None
            assert gen_tmp.mamba_branch is None
            assert gen_tmp.fusion is None
        
        elif cfg.branch_mode == "mamba":
        
            assert gen_tmp.cnn_branch is None
            assert gen_tmp.mamba_branch is not None
            assert gen_tmp.fusion is None
        
        elif cfg.branch_mode == "dual":
        
            assert gen_tmp.cnn_branch is not None
            assert gen_tmp.mamba_branch is not None
            assert gen_tmp.fusion is not None
        
        print(
            "✅ Requested branch architecture was created correctly."
        )

        
        
        if cfg.use_band_refinement:

            assert isinstance(
                gen_tmp.band_refinement,
                AdaptiveGHMFrequencyCollaborativeRefinement
            )
        
            initial_collaboration_scale = float(
                torch.clamp(
                    gen_tmp.band_refinement.collaboration_scale,
                    min=0.0,
                    max=1.0
                )
                .detach()
                .cpu()
                .item()
            )
        
            print(
                "Initial frequency-collaboration scale:",
                initial_collaboration_scale
            )
        
        else:
        
            assert gen_tmp.band_refinement is None
        
            initial_collaboration_scale = None
        
            print(
                "Frequency-collaborative refinement disabled."
            )
        if gen_tmp.use_alignment:

            assert gen_tmp.hf_alignment is not None, (
                "Alignment is enabled, but hf_alignment was not created."
            )
        
            assert gen_tmp.low_frequency_guide is not None, (
                "Alignment is enabled, but low_frequency_guide was not created."
            )
        
        else:
        
            assert gen_tmp.hf_alignment is None, (
                "Alignment is disabled, but hf_alignment still exists."
            )
        
            assert gen_tmp.low_frequency_guide is None, (
                "Alignment is disabled, but low_frequency_guide still exists."
            )
        
            alignment_parameter_names = [
                name
                for name, _ in gen_tmp.named_parameters()
                if (
                    name.startswith("hf_alignment.")
                    or name.startswith("low_frequency_guide.")
                )
            ]
        
            assert len(alignment_parameter_names) == 0, (
                "Alignment is disabled, but alignment-related trainable "
                f"parameters still exist: {alignment_parameter_names}"
            )
        
            print(
                "✅ Alignment modules and parameters are completely absent."
            )

        if gen_tmp.use_alignment:
        
            initial_mixed_alignment_scale = float(
                torch.clamp(
                    gen_tmp.hf_alignment.mixed_scale,
                    min=0.0,
                    max=1.0
                )
                .detach()
                .cpu()
                .item()
            )
        
            initial_high_alignment_scale = float(
                torch.clamp(
                    gen_tmp.hf_alignment.high_scale,
                    min=0.0,
                    max=1.0
                )
                .detach()
                .cpu()
                .item()
            )
        
            print(
                "Initial mixed-band alignment scale:",
                initial_mixed_alignment_scale
            )
        
            print(
                "Initial high-band alignment scale:",
                initial_high_alignment_scale
            )
        
        else:
        
            print(
                "Alignment disabled: "
                "alignment parameters are not active."
            )
    
        print(
            "\n====== Generator-Dataset Integration Check ======"
        )
    
        # -------------------------------------------------
        # 1.  decomposition
        # -------------------------------------------------
        bands_dbg = gen_tmp.dwt(
            noisy_dbg
        )
    
        expected_half_size = (
            cfg.patch_size // 2,
            cfg.patch_size // 2
        )
    
        print(
            "Input noisy shape:",
            tuple(noisy_dbg.shape)
        )
    
        print(
            f"{cfg.wavelet_type.upper()} bands shape:",
            tuple(bands_dbg.shape)
        )
    
        expected_num_bands = gen_tmp.expected_num_bands
        
        assert bands_dbg.shape == (
            cfg.batch_size,
            expected_num_bands,
            expected_half_size[0],
            expected_half_size[1]
        ), (
            f"Unexpected {cfg.wavelet_type.upper()} output shape. "
            f"Received {tuple(bands_dbg.shape)}."
        )
    
        # -------------------------------------------------
        # 2. Shared band encoding
        # -------------------------------------------------
        encoded_dbg = gen_tmp.band_encoder(
            bands_dbg
        )
    
        print(
            "Encoded bands shape:",
            tuple(encoded_dbg.shape)
        )
    
        assert encoded_dbg.shape == (
            cfg.batch_size,
            expected_num_bands,
            40,
            expected_half_size[0],
            expected_half_size[1]
        ), (
            "Unexpected encoded-band shape. "
            f"Received {tuple(encoded_dbg.shape)}."
        )
    
        # -------------------------------------------------
        # -------------------------------------------------
        # 3. Flatten independent GHM bands
        # -------------------------------------------------
        encoded_flat_dbg = encoded_dbg.reshape(
            cfg.batch_size * expected_num_bands,
            40,
            expected_half_size[0],
            expected_half_size[1]
        )
        
        print(
            "Flattened encoded bands:",
            tuple(encoded_flat_dbg.shape)
        )
        
        assert encoded_flat_dbg.shape == (
            cfg.batch_size * expected_num_bands,
            40,
            expected_half_size[0],
            expected_half_size[1]
        )
        
        # -------------------------------------------------
        # 4. Shared band projection
        # -------------------------------------------------
        projected_flat_dbg = gen_tmp.band_projection(
            encoded_flat_dbg
        )
        
        print(
            f"Projected {cfg.wavelet_type.upper()} bands:",
            tuple(projected_flat_dbg.shape)
        )
        
        assert projected_flat_dbg.shape == (
            cfg.batch_size * expected_num_bands,
            144,
            expected_half_size[0],
            expected_half_size[1]
        )
        
        # -------------------------------------------------
        # 5. Shared CNN and Mamba paths
        # -------------------------------------------------
        
        cnn_flat_dbg = None
        mamba_flat_dbg = None
        
        if gen_tmp.branch_mode == "cnn":
        
            fused_flat_dbg = gen_tmp.cnn_branch(
                projected_flat_dbg
            )
        
            assert gen_tmp.mamba_branch is None
            assert gen_tmp.fusion is None
        
            print(
                "✅ CNN-only branch check passed."
            )
        
        elif gen_tmp.branch_mode == "mamba":
        
            fused_flat_dbg = gen_tmp.mamba_branch(
                projected_flat_dbg
            )
        
            assert gen_tmp.cnn_branch is None
            assert gen_tmp.fusion is None
        
            print(
                "✅ Mamba-only branch check passed."
            )
        
        else:
        
            cnn_flat_dbg = gen_tmp.cnn_branch(
                projected_flat_dbg
            )
        
            mamba_flat_dbg = gen_tmp.mamba_branch(
                projected_flat_dbg
            )
        
            fused_flat_dbg = gen_tmp.fusion(
                cnn_flat_dbg,
                mamba_flat_dbg
            )
        
            print(
                "✅ Dual-branch fusion check passed."
            )
        
        print(
            f"Fused {cfg.wavelet_type.upper()} bands:",
            tuple(fused_flat_dbg.shape)
        )
        
        assert fused_flat_dbg.shape == (
            cfg.batch_size * expected_num_bands,
            144,
            expected_half_size[0],
            expected_half_size[1]
        )
        
        # -------------------------------------------------
        # 6. Restore explicit band dimension
        # -------------------------------------------------
        fused_bands_dbg = fused_flat_dbg.reshape(
            cfg.batch_size,
            expected_num_bands,
            144,
            expected_half_size[0],
            expected_half_size[1]
        )
        # -------------------------------------------------
        #
        # -------------------------------------------------
        # 7. Build low-frequency guide only when
        #    alignment is enabled
        # -------------------------------------------------
        low_feature_list_dbg = None
        low_features_concat_dbg = None
        low_frequency_guide_dbg = None
        
        if gen_tmp.use_alignment:
        
            low_feature_list_dbg = [
                fused_bands_dbg[
                    :,
                    band_index,
                    :,
                    :,
                    :
                ]
                for band_index in gen_tmp.low_band_indices
            ]
        
            low_features_concat_dbg = torch.cat(
                low_feature_list_dbg,
                dim=1
            )
        
            print(
                f"Concatenated low-frequency "
                f"{cfg.wavelet_type.upper()} features:",
                tuple(low_features_concat_dbg.shape)
            )
        
            assert low_features_concat_dbg.shape == (
                cfg.batch_size,
                144 * len(gen_tmp.low_band_indices),
                expected_half_size[0],
                expected_half_size[1]
            ), (
                "Unexpected concatenated low-frequency "
                f"feature shape: {tuple(low_features_concat_dbg.shape)}."
            )
        
            low_frequency_guide_dbg = (
                gen_tmp.low_frequency_guide(
                    low_features_concat_dbg
                )
            )
        
            print(
                "Low-frequency structural guide:",
                tuple(low_frequency_guide_dbg.shape)
            )
        
            assert low_frequency_guide_dbg.shape == (
                cfg.batch_size,
                144,
                expected_half_size[0],
                expected_half_size[1]
            ), (
                "Unexpected low-frequency guide shape: "
                f"{tuple(low_frequency_guide_dbg.shape)}."
            )
        
            assert torch.isfinite(
                low_frequency_guide_dbg
            ).all(), (
                "Non-finite values found in "
                "low-frequency structural guide."
            )
        
        else:
        
            print(
                "Alignment disabled: "
                "low-frequency guide was not computed."
            )
        
        
        # -------------------------------------------------
        # 8. Apply alignment only when enabled
        # -------------------------------------------------
        aligned_band_feature_list_dbg = []
        
        for band_index in range(
            expected_num_bands
        ):
            band_features_dbg = fused_bands_dbg[
                :,
                band_index,
                :,
                :,
                :
            ]
        
            if gen_tmp.use_alignment:
        
                if band_index in gen_tmp.mixed_band_indices:
        
                    band_features_dbg = gen_tmp.hf_alignment(
                        feat_hf=band_features_dbg,
                        feat_lf=low_frequency_guide_dbg,
                        band_group="mixed"
                    )
        
                elif band_index in gen_tmp.high_band_indices:
        
                    band_features_dbg = gen_tmp.hf_alignment(
                        feat_hf=band_features_dbg,
                        feat_lf=low_frequency_guide_dbg,
                        band_group="high"
                    )
        
            aligned_band_feature_list_dbg.append(
                band_features_dbg
            )
        
        
        aligned_fused_bands_dbg = torch.stack(
            aligned_band_feature_list_dbg,
            dim=1
        )

        # -------------------------------------------------
        # Verify alignment behavior
        # -------------------------------------------------
        if not gen_tmp.use_alignment:
        
            assert torch.equal(
                aligned_fused_bands_dbg,
                fused_bands_dbg
            ), (
                "Alignment is disabled, but band features "
                "were unexpectedly modified."
            )
        
            print("✅ Alignment bypass check passed.")
        
        else:
        
            for low_index in gen_tmp.low_band_indices:
        
                assert torch.equal(
                    aligned_fused_bands_dbg[:, low_index],
                    fused_bands_dbg[:, low_index]
                ), (
                    f"Low-frequency band {low_index} "
                    "was unexpectedly modified."
                )
        
            print("✅ Alignment selective-application check passed.")
        
        print(
            f"Aligned {cfg.wavelet_type.upper()} "
            f"band features:",
            tuple(aligned_fused_bands_dbg.shape)
        )
        
        assert aligned_fused_bands_dbg.shape == (
            cfg.batch_size,
            expected_num_bands,
            144,
            expected_half_size[0],
            expected_half_size[1]
        ), (
            "Unexpected aligned-band feature shape: "
            f"{tuple(aligned_fused_bands_dbg.shape)}."
        )
        
        assert torch.isfinite(
            aligned_fused_bands_dbg
        ).all(), (
            "Non-finite values found after alignment stage."
        )
        
        
        # -------------------------------------------------
        # 7. Predict 16 GHM coefficient maps
        # -------------------------------------------------
        predicted_band_list_dbg = []
        # -------------------------------------------------
    
        if gen_tmp.use_band_refinement:

            refined_fused_bands_dbg = (
                gen_tmp.band_refinement(
                    aligned_fused_bands_dbg
                )
            )
        
        else:
        
            refined_fused_bands_dbg = (
                aligned_fused_bands_dbg
            )
        
            assert gen_tmp.band_refinement is None
        
            print(
                "✅ Frequency-collaborative refinement bypassed."
            )
        
        print(
            f"Refined {cfg.wavelet_type.upper()} "
            f"band features:",
            tuple(refined_fused_bands_dbg.shape)
        )
        
        assert refined_fused_bands_dbg.shape == (
            cfg.batch_size,
            expected_num_bands,
            144,
            expected_half_size[0],
            expected_half_size[1]
        ), (
            "Unexpected refined-band feature shape: "
            f"{tuple(refined_fused_bands_dbg.shape)}."
        )
        
        assert torch.isfinite(
            refined_fused_bands_dbg
        ).all(), (
            "Non-finite values found after band refinement."
        )
        
        for band_index, head in enumerate(
            gen_tmp.noise_band_heads
        ):
            predicted_band_dbg = head(
                refined_fused_bands_dbg[
                    :,
                    band_index,
                    :,
                    :,
                    :
                ]
            )
        
            predicted_band_list_dbg.append(
                predicted_band_dbg
            )
        
        predicted_noise_bands_dbg = torch.cat(
            predicted_band_list_dbg,
            dim=1
        )
        
        print(
            f"Predicted {cfg.wavelet_type.upper()} noise bands:",
            tuple(predicted_noise_bands_dbg.shape)
        )
        
        assert predicted_noise_bands_dbg.shape == (
            cfg.batch_size,
            expected_num_bands,
            expected_half_size[0],
            expected_half_size[1]
        )
        
        # -------------------------------------------------
        # 8. Inverse-GHM predicted noise
        # -------------------------------------------------
        predicted_noise_dbg = gen_tmp.idwt(
            predicted_noise_bands_dbg
        )
        
        print(
            "Predicted spatial noise:",
            tuple(predicted_noise_dbg.shape)
        )
        
        assert predicted_noise_dbg.shape == noisy_dbg.shape
    
        # -------------------------------------------------
        # -------------------------------------------------
        # Full-resolution refinement sanity check
        # -------------------------------------------------
        denoised_coarse_dbg = (
            noisy_dbg
            - predicted_noise_dbg
        )
        
        full_res_correction_dbg = None
        full_res_scale_dbg = None
        
        if gen_tmp.use_full_res_refinement:
        
            assert gen_tmp.full_res_refinement is not None
            assert gen_tmp.full_res_scale is not None
        
            full_res_correction_dbg = (
                gen_tmp.full_res_refinement(
                    denoised_coarse=denoised_coarse_dbg,
                    noisy=noisy_dbg
                )
            )
        
            assert (
                full_res_correction_dbg.shape
                == noisy_dbg.shape
            ), (
                "Unexpected full-resolution correction "
                f"shape: {tuple(full_res_correction_dbg.shape)}."
            )
        
            assert torch.isfinite(
                full_res_correction_dbg
            ).all(), (
                "Non-finite values found in "
                "full-resolution correction."
            )
        
            full_res_scale_dbg = torch.clamp(
                gen_tmp.full_res_scale,
                min=0.0,
                max=1.0
            )
        
            denoised_refined_dbg = (
                denoised_coarse_dbg
                + full_res_scale_dbg
                * full_res_correction_dbg
            )
        
            assert (
                denoised_refined_dbg.shape
                == noisy_dbg.shape
            )
        
            assert torch.isfinite(
                denoised_refined_dbg
            ).all()
        
            print(
                "✅ Full-resolution refinement check passed."
            )
        
        else:
        
            assert gen_tmp.full_res_refinement is None
            assert gen_tmp.full_res_scale is None
        
            denoised_refined_dbg = (
                denoised_coarse_dbg
            )
        
            print(
                "✅ Full-resolution refinement bypassed."
            )
    
        # -------------------------------------------------
        # 4. Full generator output
        # -------------------------------------------------
        denoised_dbg = gen_tmp(
            noisy_dbg
        )
    
        print(
            "Denoised output shape:",
            tuple(denoised_dbg.shape)
        )
    
        print(
            "Raw denoised range:",
            float(denoised_dbg.min()),
            float(denoised_dbg.max())
        )
    
        assert denoised_dbg.shape == clean_dbg.shape, (
            "Denoised output and clean target must have "
            "identical shapes. "
            f"Denoised={tuple(denoised_dbg.shape)}, "
            f"clean={tuple(clean_dbg.shape)}."
        )
    
        assert torch.isfinite(
            denoised_dbg
        ).all(), (
            "Non-finite values found in generator output."
        )
    
        print(
            "✅ Dataset and generator are correctly connected."
        )
        print(
            "   Wavelet:",
            cfg.wavelet_type.upper()
        )
        print(
            "   Branch mode:",
            cfg.branch_mode
        )
        print(
            "   Fusion:",
            cfg.fusion_type
        )
        print(
            "   Alignment:",
            cfg.use_alignment
        )
        print(
            "   Frequency refinement:",
            cfg.use_band_refinement
        )
        print(
            "   Full-resolution refinement:",
            cfg.use_full_res_refinement
        )

    # Free temporary generator before training model creation.
    del gen_tmp
    del bands_dbg
    del encoded_dbg
    
    del encoded_flat_dbg
    del projected_flat_dbg
    del cnn_flat_dbg
    del mamba_flat_dbg
    del fused_flat_dbg
    del fused_bands_dbg
    del low_feature_list_dbg
    del low_features_concat_dbg
    del low_frequency_guide_dbg
    del aligned_band_feature_list_dbg
    del band_features_dbg
    del aligned_fused_bands_dbg
    del predicted_band_list_dbg
    del predicted_band_dbg
    del predicted_noise_bands_dbg
    del predicted_noise_dbg

    del denoised_coarse_dbg
    del full_res_correction_dbg
    del full_res_scale_dbg
    del denoised_refined_dbg
    
    del denoised_dbg
    del noisy_dbg
    del clean_dbg
    
    del refined_fused_bands_dbg
    
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

        # =====================================================
    # Reset randomness after sanity checks
    #
    # This ensures that temporary checks do not alter the
    # actual experiment initialization or DataLoader order.
    # =====================================================
    seed_everything(
        seed=cfg.seed,
        deterministic=cfg.deterministic
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(cfg.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last,
        worker_init_fn=seed_worker,
        generator=loader_generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False
    )
    
   

    # ===================== Models =====================
    # =====================================================
    # Denoising generator
    # =====================================================
    

    gen = MedMultiwaveletDenoisingGenerator(
        in_ch=1,
        out_ch=1,
        wavelet_type=cfg.wavelet_type,
        band_feature_channels=40,
        feature_channels=144,
        cnn_blocks=6,
        mamba_blocks=6,
        branch_mode=cfg.branch_mode,
        fusion_type=cfg.fusion_type,
        use_alignment=cfg.use_alignment,
        use_band_refinement=cfg.use_band_refinement,
        use_full_res_refinement=cfg.use_full_res_refinement
    ).to(cfg.device)

   
    disc_patch = LSGANPatchDiscriminator(
        in_ch=1,
        base_ch=cfg.disc_channels
    ).to(cfg.device)

       
    


    # ===================== Losses & Optimizers =====================
    
    charb = CharbonnierLoss().to(cfg.device) 
    
    ssim_loss_func = SSIMLoss().to(cfg.device)
    


    opt_g = torch.optim.Adam(gen.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2))
    opt_d_patch = torch.optim.Adam(disc_patch.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2))


    # =====================================================
    # =====================================================
    # Resume training from checkpoint
    # =====================================================
    checkpoint_path = os.path.join(
        cfg.save_dir,
        "last.pt"
    )

    start_epoch = 1
    best_psnr = -1.0
    checkpoint = None

    if os.path.exists(checkpoint_path):

        print(
            f"✅ Loading checkpoint from "
            f"{checkpoint_path}..."
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=cfg.device
        )

        # -------------------------------------------------
        # Restore normal generator training weights
        # -------------------------------------------------
        if "gen" not in checkpoint:
            raise KeyError(
                "The checkpoint does not contain "
                "generator weights under key 'gen'."
            )

        gen.load_state_dict(
            checkpoint["gen"],
            strict=True
        )

        # -------------------------------------------------
        # Restore discriminator weights
        # -------------------------------------------------
        if "disc_patch" in checkpoint:
            disc_patch.load_state_dict(
                checkpoint["disc_patch"],
                strict=True
            )

        # -------------------------------------------------
        # Restore optimizer states
        # -------------------------------------------------
        if "opt_g" in checkpoint:
            opt_g.load_state_dict(
                checkpoint["opt_g"]
            )

        if "opt_d_patch" in checkpoint:
            opt_d_patch.load_state_dict(
                checkpoint["opt_d_patch"]
            )

        # -------------------------------------------------
        # Restore epoch and best validation score
        # -------------------------------------------------
        start_epoch = (
            int(
                checkpoint.get(
                    "epoch",
                    0
                )
            )
            + 1
        )

        best_psnr = float(
            checkpoint.get(
                "best_psnr",
                -1.0
            )
        )

        print(
            f"✅ Training resumed from Epoch "
            f"{start_epoch}. "
            f"Current best PSNR: "
            f"{best_psnr:.4f}"
        )


    # =====================================================
    # Initialize or restore EMA
    # =====================================================
    ema = EMA(
        gen,
        decay=cfg.ema_decay
    )

    if (
        checkpoint is not None
        and "ema" in checkpoint
    ):
        ema.load_state_dict(
            checkpoint["ema"],
            device=cfg.device
        )

        print(
            "✅ EMA state restored from checkpoint."
        )

    elif checkpoint is not None:
        print(
            "⚠️ The checkpoint does not contain an EMA state. "
            "EMA was initialized from the loaded generator weights."
        )

    else:
        print(
            "✅ EMA initialized from the new generator weights."
        )


    # =====================================================
    # Logging paths
    # =====================================================
    log_path = os.path.join(
        cfg.save_dir,
        "log.txt"
    )
    

        
    # ===================== CSV: debug per-iter =====================
    debug_path = os.path.join(cfg.save_dir, 'train_debug_iter.csv')
    if not os.path.exists(debug_path) and start_epoch == 1:
        with open(debug_path, 'w', encoding='utf-8') as f:
            f.write(
            "epoch,iteration,"
            "generator_loss,discriminator_loss,"
            "content_loss,charbonnier_loss,ssim_loss,"
            "adversarial_loss,"
            "denoised_min,denoised_max,denoised_mean,"
            "noisy_min,noisy_max,"
            "clean_min,clean_max\n"
        )

    # ===================== CSV: train =====================
    train_loss_path = os.path.join(cfg.save_dir, 'train_loss_epoch.csv')
    if not os.path.exists(train_loss_path) and start_epoch == 1:
        with open(train_loss_path, 'w', encoding='utf-8') as f:
           f.write(
                "epoch,"
                "wavelet_type,"
                "branch_mode,"
                "fusion_type,"
                "use_alignment,"
                "use_band_refinement,"
                "use_full_res_refinement,"
                "generator_loss,discriminator_loss,"
                "content_loss,adversarial_loss,"
                "denoising_psnr,denoising_ssim,"
                "mixed_alignment_scale,"
                "high_alignment_scale\n"
            )
           


    # ===================== CSV: val =====================
    val_metrics_path = os.path.join(cfg.save_dir, 'val_metrics_epoch.csv')
    if not os.path.exists(val_metrics_path) and start_epoch == 1:
        with open(val_metrics_path, 'w', encoding='utf-8') as f:
            f.write(
                "epoch,denoising_psnr,denoising_ssim\n"
            )


    # ===================== Epoch Loop =====================
    
    for epoch in range(start_epoch, cfg.epochs + 1):
    
        gen.train()
        disc_patch.train()

        epoch_loss_g = 0.0
        epoch_loss_d = 0.0
        epoch_content = 0.0
        epoch_adv = 0.0
        epoch_train_psnr = 0.0
        epoch_train_ssim = 0.0
        num_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{cfg.epochs}"
        )

        for noisy, clean in pbar:
            noisy = noisy.to(
                cfg.device,
                non_blocking=cfg.pin_memory
            )

            clean = clean.to(
                cfg.device,
                non_blocking=cfg.pin_memory
            )

            use_gan = (
                cfg.adv_weight > 0.0
                and epoch > cfg.gan_warmup_epochs
            )

            # =================================================
            # 1. Train the discriminator
            # =================================================
            if use_gan:
                with torch.no_grad():
                    denoised_detached = gen(noisy).clamp(
                        0.0,
                        1.0
                    )

                clean_for_discriminator = clean.clamp(
                    0.0,
                    1.0
                )

                discriminator_real = disc_patch(
                    clean_for_discriminator
                )

                discriminator_fake = disc_patch(
                    denoised_detached
                )

                loss_d = lsgan_d_loss(
                    discriminator_real,
                    discriminator_fake
                )

                opt_d_patch.zero_grad(
                    set_to_none=True
                )

                loss_d.backward()
                opt_d_patch.step()

            else:
                loss_d = torch.zeros(
                    (),
                    device=cfg.device
                )

            # =================================================
            # 2. Train the denoising generator
            # =================================================
            denoised = gen(noisy)

            denoised_clamped = denoised.clamp(
                0.0,
                1.0
            )

            clean_clamped = clean.clamp(
                0.0,
                1.0
            )

            loss_charbonnier = charb(
                denoised_clamped,
                clean_clamped
            )

            
            loss_ssim = ssim_loss_func(
                denoised_clamped,
                clean_clamped
            )

            loss_content = (
                cfg.charbonnier_weight * loss_charbonnier
                + cfg.ssim_weight * loss_ssim
            )

            if use_gan:
                discriminator_fake_for_generator = disc_patch(
                    denoised_clamped
                )

                loss_adversarial = lsgan_g_loss(
                    discriminator_fake_for_generator
                )
            else:
                loss_adversarial = torch.zeros(
                    (),
                    device=cfg.device
                )

            loss_generator = (
                cfg.content_weight * loss_content
                + cfg.adv_weight * loss_adversarial
            )

            # =================================================
            # 3. Output and input statistics
            # =================================================
            with torch.no_grad():
                denoised_min = float(
                    denoised.min().item()
                )

                denoised_max = float(
                    denoised.max().item()
                )

                denoised_mean = float(
                    denoised.mean().item()
                )

                noisy_min = float(
                    noisy.min().item()
                )

                noisy_max = float(
                    noisy.max().item()
                )

                clean_min = float(
                    clean.min().item()
                )

                clean_max = float(
                    clean.max().item()
                )

            iter_in_epoch = num_batches + 1

            with open(
                debug_path,
                "a",
                encoding="utf-8"
            ) as file:
                file.write(
                    f"{epoch},{iter_in_epoch},"
                    f"{loss_generator.item():.6f},"
                    f"{loss_d.item():.6f},"
                    f"{loss_content.item():.6f},"
                    f"{loss_charbonnier.item():.6f},"
                    f"{loss_ssim.item():.6f},"
                    f"{loss_adversarial.item():.6f},"
                    f"{denoised_min:.6f},"
                    f"{denoised_max:.6f},"
                    f"{denoised_mean:.6f},"
                    f"{noisy_min:.6f},"
                    f"{noisy_max:.6f},"
                    f"{clean_min:.6f},"
                    f"{clean_max:.6f}\n"
                )

            # =================================================
            # 4. Numerical stability checks
            # =================================================
            if not torch.isfinite(loss_generator):
                print(
                    "❌ Non-finite generator loss at "
                    f"epoch {epoch}, iteration {iter_in_epoch}. "
                    "The generator update was skipped."
                )

                opt_g.zero_grad(
                    set_to_none=True
                )

                continue

            if (
                denoised_min < -1.0
                or denoised_max > 2.0
            ):
                print(
                    "⚠️ Denoised output is outside the "
                    "expected training range at "
                    f"epoch {epoch}, iteration {iter_in_epoch}: "
                    f"min={denoised_min:.3f}, "
                    f"max={denoised_max:.3f}"
                )

            # =================================================
            # 5. Generator optimization
            # =================================================
            opt_g.zero_grad(
                set_to_none=True
            )

            loss_generator.backward()

            torch.nn.utils.clip_grad_norm_(
                gen.parameters(),
                max_norm=1.0
            )

            opt_g.step()
            ema.update(gen)

            # =================================================
            # 6. Accumulate epoch statistics
            # =================================================
            epoch_loss_d += loss_d.item()
            epoch_loss_g += loss_generator.item()
            epoch_content += loss_content.item()
            epoch_adv += loss_adversarial.item()
            

            with torch.no_grad():
                batch_psnr, batch_ssim = (
                    batch_psnr_ssim_basic_sr(
                        denoised_clamped,
                        clean
                    )
                )

                epoch_train_psnr += batch_psnr
                epoch_train_ssim += batch_ssim

            num_batches += 1

            pbar.set_postfix({
                "loss_d": f"{loss_d.item():.4f}",
                "loss_g": f"{loss_generator.item():.4f}",
                "content": f"{loss_content.item():.4f}",
                "charb": f"{loss_charbonnier.item():.4f}",
                "ssim_loss": f"{loss_ssim.item():.4f}",
                "adv": f"{loss_adversarial.item():.4f}",
            })

        

        # ===== 4) Averages per epoch (train) =====
        mean_loss_g      = epoch_loss_g      / max(1, num_batches)
        mean_loss_d      = epoch_loss_d      / max(1, num_batches)
        mean_content     = epoch_content     / max(1, num_batches)
        mean_adv         = epoch_adv         / max(1, num_batches)
        mean_train_psnr  = epoch_train_psnr  / max(1, num_batches)
        mean_train_ssim  = epoch_train_ssim  / max(1, num_batches)

        if cfg.use_alignment:
            current_mixed_alignment_scale = float(
                torch.clamp(
                    gen.hf_alignment.mixed_scale,
                    min=0.0,
                    max=1.0
                ).detach().cpu().item()
            )
        
            current_high_alignment_scale = float(
                torch.clamp(
                    gen.hf_alignment.high_scale,
                    min=0.0,
                    max=1.0
                ).detach().cpu().item()
            )
        
        else:
        
            current_mixed_alignment_scale = float("nan")
            current_high_alignment_scale = float("nan")
        


        # ✅ صف واحد فقط في CSV، مطابق للهيدر
        with open(
            train_loss_path,
            "a",
            encoding="utf-8"
        ) as f:
            f.write(
                f"{epoch},"
                f"{cfg.wavelet_type},"
                f"{cfg.branch_mode},"
                f"{cfg.fusion_type},"
                f"{cfg.use_alignment},"
                f"{cfg.use_band_refinement},"
                f"{cfg.use_full_res_refinement},"
                f"{mean_loss_g:.6f},"
                f"{mean_loss_d:.6f},"
                f"{mean_content:.6f},"
                f"{mean_adv:.6f},"
                f"{mean_train_psnr:.6f},"
                f"{mean_train_ssim:.6f},"
                f"{current_mixed_alignment_scale:.6f},"
                f"{current_high_alignment_scale:.6f}\n"
            )

        train_message = (
            f"[DENOISING TRAIN] Epoch {epoch}: "
            f"PSNR={mean_train_psnr:.3f}, "
            f"SSIM={mean_train_ssim:.4f}, "
            f"generator_loss={mean_loss_g:.6f}, "
            f"discriminator_loss={mean_loss_d:.6f}, "
            f"content_loss={mean_content:.6f}, "
            f"adversarial_loss={mean_adv:.6f}"
        )
        
        if cfg.use_alignment:
        
            train_message += (
                f", mixed_alignment_scale="
                f"{current_mixed_alignment_scale:.6f}"
                f", high_alignment_scale="
                f"{current_high_alignment_scale:.6f}"
            )
        
        print(train_message)

        # ===== 5) Validation (using EMA weights) =====
        ema.store(gen)
        ema.copy_to(gen)
        val_psnr, val_ssim = validate_denoising(gen, val_loader, cfg.device)
        ema.restore(gen)

        # حفظ val في CSV
        with open(val_metrics_path, 'a', encoding='utf-8') as f_val:
            f_val.write(f"{epoch},{val_psnr:.6f},{val_ssim:.6f}\n")

        # حفظ log نصي
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"{epoch}\t{val_psnr:.3f}\t{val_ssim:.3f}\n")

        # =====================================================
        # =====================================================
        # Update best validation score
        # =====================================================
        is_best = val_psnr > best_psnr
        
        if is_best:
            best_psnr = val_psnr
        
        
        # =====================================================
        # Save last resumable checkpoint
        #
        # gen:
        #     normal training generator weights
        #
        # ema:
        #     exponential moving average shadow weights
        # =====================================================
        checkpoint_state = {
            "gen": gen.state_dict(),
            "ema": ema.state_dict(),
        
            "disc_patch": disc_patch.state_dict(),
        
            "opt_g": opt_g.state_dict(),
            "opt_d_patch": opt_d_patch.state_dict(),
        
            "cfg": cfg.__dict__,
        
            "epoch": epoch,
            "best_psnr": best_psnr,
        }
        
        torch.save(
            checkpoint_state,
            os.path.join(
                cfg.save_dir,
                "last.pt"
            )
        )
        
        
        # =====================================================
        # Save best checkpoint
        # =====================================================
        if is_best:
        
            torch.save(
                checkpoint_state,
                os.path.join(
                    cfg.save_dir,
                    "best.pt"
                )
            )
        
            # -------------------------------------------------
            # Save a separate EMA generator for final testing
            # -------------------------------------------------
            ema.store(gen)
            ema.copy_to(gen)
        
            torch.save(
                {
                    "gen": gen.state_dict(),
                    "cfg": cfg.__dict__,
                    "epoch": epoch,
                    "best_psnr": best_psnr,
                    "weights_type": "ema"
                },
                os.path.join(
                    cfg.save_dir,
                    "best_ema_generator.pt"
                )
            )
        
            ema.restore(gen)
        

        

        print(
            f"[DENOISING VAL] Epoch {epoch}: "
            f"PSNR={val_psnr:.3f}, "
            f"SSIM={val_ssim:.4f}, "
            f"best_PSNR={best_psnr:.3f}"
        )
