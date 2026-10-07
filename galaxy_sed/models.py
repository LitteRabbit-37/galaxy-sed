"""Neural network architectures.

All models work in standardised coordinates (see ``galaxy_sed.data``):

* ``DirectMLP``        photometry -> all physical parameters. Its redshift
                       output feeds the GLO inversion.
* ``GLOGenerator``     latent code -> full SED and parameters. Trained by
                       generative latent optimisation, then inverted against
                       the photometry of a new galaxy.
* ``ParameterHeads``   one small network per physical parameter, reading the
                       photometry alone or the photometry plus a
                       reconstructed SED.
* ``CWGANGenerator``   noise + photometry -> full SED and parameters, trained
                       adversarially against ``CWGANCritic``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

SED_LENGTH = 223
N_PARAMS = 18
N_INPUTS = 40  # 20 standardised fluxes + 20 mask bits


class DirectMLP(nn.Module):
    """Photometry to all parameters in one network."""

    def __init__(
        self, n_in: int = N_INPUTS, n_out: int = N_PARAMS, dropout: float = 0.1
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _sed_upsampler(in_channels: int) -> nn.Sequential:
    """Transposed-convolution stack: a (batch, in_channels, 1) code to 224 samples.

    Lengths 1 -> 7 -> 14 -> 28 -> 56 -> 112 -> 224, the last sample is dropped
    to match the 223-point wavelength grid.
    """
    return nn.Sequential(
        nn.ConvTranspose1d(in_channels, 512, kernel_size=7, stride=1, padding=0),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm1d(64),
        nn.ReLU(inplace=True),
        nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm1d(32),
        nn.ReLU(inplace=True),
        nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1),
    )


def _parameter_mlp(in_features: int, n_params: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 64),
        nn.ReLU(inplace=True),
        nn.Linear(64, n_params),
    )


class GLOGenerator(nn.Module):
    """Latent code (on the unit sphere) to SED and parameters."""

    def __init__(
        self, latent_dim: int = 32, sed_len: int = SED_LENGTH, n_params: int = N_PARAMS
    ):
        super().__init__()
        self.sed_len = sed_len
        self.trunk = _sed_upsampler(latent_dim)
        self.params_head = _parameter_mlp(latent_dim, n_params)

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sed = self.trunk(latent.unsqueeze(-1)).squeeze(1)[:, : self.sed_len]
        return sed, self.params_head(latent)


class LatentCodes(nn.Module):
    """One trainable latent code per training galaxy, kept on the unit sphere.

    Only needed during GLO training, a new galaxy gets its code by inversion.
    """

    def __init__(self, n_samples: int, latent_dim: int, seed: int):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        codes = torch.randn(n_samples, latent_dim, generator=generator)
        self.codes = nn.Parameter(codes / codes.norm(dim=1, keepdim=True))

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.codes[indices]


class ParamSubNet(nn.Module):
    """Small MLP for one parameter: a value, a (value, log variance) pair or class logits."""

    def __init__(self, n_in: int, n_out: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ParameterHeads(nn.Module):
    """One independent ``ParamSubNet`` per parameter.

    Each spec (see ``parameter_specs``) says whether its parameter is a
    regression target or a classification over a small grid of values. With
    ``uncertainty``, regression heads output a mean and a log variance and are
    trained by Gaussian negative log likelihood, so every galaxy gets its own
    error bar, classification heads carry theirs in the softmax.
    """

    def __init__(
        self,
        n_in: int,
        specs: list[dict[str, Any]],
        uncertainty: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.specs = specs
        self.uncertainty = uncertainty
        heads = []
        for spec in specs:
            if spec["type"] == "cls":
                n_out = spec["n_classes"]
            else:
                n_out = 2 if uncertainty else 1
            heads.append(ParamSubNet(n_in, n_out, dropout))
        self.heads = nn.ModuleList(heads)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [head(x) for head in self.heads]


def parameter_specs(
    train_params_physical: np.ndarray,
    param_names: list[str],
    param_mean: np.ndarray,
    param_std: np.ndarray,
    param_log10: np.ndarray,
    max_classes: int,
) -> list[dict[str, Any]]:
    """Decide, per parameter, between a regression and a classification head.

    Simulation libraries often sample a parameter on a handful of values. When
    the training set holds between 2 and ``max_classes`` distinct values, the
    parameter becomes a classification over those values (in standardised
    coordinates), which is more natural than regressing a three-level variable.
    """
    specs = []
    for j, name in enumerate(param_names):
        grid = np.unique(np.round(train_params_physical[:, j], decimals=8))
        if 2 <= grid.size <= max_classes:
            raw = np.log10(grid) if param_log10[j] else grid
            # Levels are stored at float32 precision, the precision of the targets.
            levels = ((raw - float(param_mean[j])) / float(param_std[j])).astype(
                np.float32
            )
            order = np.argsort(levels)
            specs.append(
                {
                    "index": j,
                    "name": name,
                    "type": "cls",
                    "n_classes": int(grid.size),
                    "levels_std": [float(v) for v in levels[order]],
                    "grid_physical": [float(v) for v in grid[order]],
                }
            )
        else:
            specs.append({"index": j, "name": name, "type": "reg", "n_classes": 0})
    return specs


class CWGANGenerator(nn.Module):
    """Noise and photometry to SED and parameters (conditional generator)."""

    def __init__(
        self,
        latent_dim: int = 32,
        cond_dim: int = N_INPUTS,
        sed_len: int = SED_LENGTH,
        n_params: int = N_PARAMS,
    ):
        super().__init__()
        self.sed_len = sed_len
        self.trunk = _sed_upsampler(latent_dim + cond_dim)
        self.params_head = _parameter_mlp(latent_dim + cond_dim, n_params)

    def forward(
        self, noise: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([noise, cond], dim=1)
        sed = self.trunk(combined.unsqueeze(-1)).squeeze(1)[:, : self.sed_len]
        return sed, self.params_head(combined)


class CWGANCritic(nn.Module):
    """Scores an (SED, parameters, photometry) triple, only used during training."""

    def __init__(
        self,
        sed_len: int = SED_LENGTH,
        n_params: int = N_PARAMS,
        cond_dim: int = N_INPUTS,
    ):
        super().__init__()
        self.sed_body = nn.Sequential(
            nn.Conv1d(1, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        with torch.no_grad():
            flat = self.sed_body(torch.zeros(1, 1, sed_len)).flatten(1).shape[1]
        self.param_body = nn.Sequential(
            nn.Linear(n_params, 64), nn.LeakyReLU(0.2, inplace=True)
        )
        self.condition_body = nn.Sequential(
            nn.Linear(cond_dim, 64), nn.LeakyReLU(0.2, inplace=True)
        )
        self.head = nn.Sequential(
            nn.Linear(flat + 128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(
        self, sed: torch.Tensor, params: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        features = torch.cat(
            [
                self.sed_body(sed.unsqueeze(1)).flatten(1),
                self.param_body(params),
                self.condition_body(cond),
            ],
            dim=1,
        )
        return self.head(features).squeeze(1)


MODEL_CLASSES = {
    cls.__name__: cls
    for cls in (DirectMLP, GLOGenerator, ParameterHeads, CWGANGenerator, CWGANCritic)
}
