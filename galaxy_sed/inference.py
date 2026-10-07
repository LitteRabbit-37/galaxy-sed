"""Inference with trained models.

Pipeline for a galaxy observed in the 20 bands:

1. ``predict_redshift``: the direct MLP estimates the redshift.
2. ``invert_glo``: a latent code is searched so that the generator's SED,
   observed at that redshift, reproduces the photometry, the search is
   repeated from several random starts and the results are averaged.
3. ``predict_parameter_heads``: the per-parameter networks read the
   photometry, or the photometry plus the reconstructed SED, and return a
   value and an error bar for each parameter.

``sample_cwgan`` is the adversarial alternative to steps 2 and 3: it averages
SEDs and parameters generated from several noise draws.

All arrays are in the standardised coordinates of ``galaxy_sed.data``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from galaxy_sed.io import seed_everything
from galaxy_sed.models import CWGANGenerator, GLOGenerator, ParameterHeads
from galaxy_sed.training import LOG_VAR_RANGE


@torch.no_grad()
def predict_batched(
    model: nn.Module, inputs: np.ndarray, device: torch.device, batch_size: int = 65536
) -> np.ndarray:
    """Forward pass of a feed-forward model over a large array."""
    model.eval()
    outputs = []
    for start in range(0, len(inputs), batch_size):
        x = torch.tensor(
            inputs[start : start + batch_size], dtype=torch.float32, device=device
        )
        outputs.append(model(x).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def predict_redshift(
    mlp_predictions_std: np.ndarray,
    param_names: list[str],
    preprocessing: dict[str, Any],
) -> np.ndarray:
    """Physical redshift from the direct MLP output, clipped to the training range."""
    j = param_names.index("redshift")
    z = (
        mlp_predictions_std[:, j] * preprocessing["param_std"][j]
        + preprocessing["param_mean"][j]
    )
    low, high = preprocessing["redshift_range"]
    return np.clip(z, low, high).astype(np.float32)


# --------------------------------------------------------------------------- #
# GLO inversion
# --------------------------------------------------------------------------- #
def batched_linear_interp(
    x_grid: torch.Tensor, y_grid: torch.Tensor, x_query: torch.Tensor
) -> torch.Tensor:
    """Differentiable linear interpolation, one set of query points per galaxy.

    ``x_grid`` (n_wave,) ascending, ``y_grid`` (n_galaxies, n_wave)
    ``x_query`` (n_galaxies, n_bands).
    """
    upper = torch.searchsorted(x_grid, x_query).clamp(1, x_grid.shape[0] - 1)
    lower = upper - 1
    fraction = (x_query - x_grid[lower]) / (x_grid[upper] - x_grid[lower] + 1e-12)
    rows = (
        torch.arange(y_grid.shape[0], device=y_grid.device)
        .unsqueeze(1)
        .expand_as(upper)
    )
    y_lower = y_grid[rows, lower]
    return y_lower + fraction * (y_grid[rows, upper] - y_lower)


def _invert_once(
    generator: GLOGenerator,
    tensors: dict[str, torch.Tensor],
    latent_dim: int,
    seed: int,
    steps: int,
    learning_rate: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """One restart: optimise one latent code per galaxy against its photometry.

    The loss compares, in standardised log flux, the observed fluxes with the
    fluxes the generated SED would produce in the same bands at the galaxy's
    redshift. Masked bands do not contribute.
    """
    torch.manual_seed(seed)
    n = tensors["X_flux"].shape[0]
    device = tensors["X_flux"].device
    latent = torch.randn(n, latent_dim, device=device)
    latent = (
        (latent / latent.norm(dim=1, keepdim=True).clamp(min=1e-12))
        .detach()
        .requires_grad_(True)
    )
    optimizer = torch.optim.Adam([latent], lr=learning_rate, betas=(0.5, 0.999))
    loss = torch.zeros((), device=device)
    for _ in range(steps):
        sed_std, _ = generator(latent)
        sed_flux = torch.pow(10.0, sed_std * tensors["sed_std"] + tensors["sed_mean"])
        band_flux = batched_linear_interp(
            tensors["wave_asc"], sed_flux[:, tensors["ascending"]], tensors["lam_rest"]
        )
        band_std = (
            torch.log10(band_flux.clamp(min=1e-30)) - tensors["flux_mean"]
        ) / tensors["flux_std"].clamp(min=1e-12)
        error = (band_std - tensors["X_flux"]) * tensors["X_mask"]
        loss = error.square().sum() / tensors["X_mask"].sum().clamp(min=1.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():  # stay on the unit sphere, as during training
            latent.data = latent.data / latent.data.norm(dim=1, keepdim=True).clamp(
                min=1e-12
            )
    with torch.no_grad():
        sed_std, params_std = generator(latent)
    return sed_std.cpu().numpy(), params_std.cpu().numpy(), float(loss.item())


def invert_glo(
    generator: GLOGenerator,
    x_flux: np.ndarray,
    x_mask: np.ndarray,
    redshift: np.ndarray,
    preprocessing: dict[str, Any],
    cfg: dict[str, Any],
    device: torch.device,
    log=print,
) -> dict[str, np.ndarray]:
    """Reconstruct the SED (and parameters) of every galaxy by GLO inversion.

    Galaxies are processed in chunks of ``cfg["chunk_size"]``, each chunk is
    inverted from ``cfg["restarts"]`` random starts (seeds ``100 * seed + r``)
    and the restarts are averaged. The spread over restarts is returned as a
    diagnostic of optimisation stability, not as a calibrated uncertainty.
    """
    seed_everything(cfg["seed"])
    generator.eval()
    for parameter in generator.parameters():
        parameter.requires_grad_(False)
    wave = np.asarray(preprocessing["wave_micron"], dtype=np.float32)
    ascending = np.argsort(wave, kind="mergesort")
    bands = torch.tensor(
        preprocessing["band_wavelengths_micron"], dtype=torch.float32, device=device
    )
    common = {
        "wave_asc": torch.tensor(wave[ascending], device=device),
        "ascending": torch.tensor(ascending, device=device),
        "sed_mean": torch.tensor(
            preprocessing["sed_log_mean"], dtype=torch.float32, device=device
        ),
        "sed_std": torch.tensor(
            preprocessing["sed_log_std"], dtype=torch.float32, device=device
        ),
        "flux_mean": torch.tensor(
            preprocessing["flux_log_mean"], dtype=torch.float32, device=device
        ),
        "flux_std": torch.tensor(
            preprocessing["flux_log_std"], dtype=torch.float32, device=device
        ),
    }
    latent_dim = generator.trunk[0].in_channels
    n = len(x_flux)
    outputs = {"sed": [], "params": [], "sed_spread": [], "params_spread": []}
    for start in range(0, n, cfg["chunk_size"]):
        stop = min(start + cfg["chunk_size"], n)
        z = torch.tensor(redshift[start:stop], dtype=torch.float32, device=device)
        tensors = {
            **common,
            "lam_rest": bands.unsqueeze(0) / (1.0 + z.unsqueeze(1)),
            "X_flux": torch.tensor(
                x_flux[start:stop], dtype=torch.float32, device=device
            ),
            "X_mask": torch.tensor(
                x_mask[start:stop], dtype=torch.float32, device=device
            ),
        }
        seds, params = [], []
        for restart in range(cfg["restarts"]):
            sed, param, loss = _invert_once(
                generator,
                tensors,
                latent_dim,
                cfg["seed"] * 100 + restart,
                cfg["steps"],
                cfg["learning_rate"],
            )
            seds.append(sed)
            params.append(param)
        log(
            f"  galaxies {start}-{stop}: final photometric loss of last restart {loss:.5f}"
        )
        outputs["sed"].append(np.mean(seds, axis=0))
        outputs["params"].append(np.mean(params, axis=0))
        outputs["sed_spread"].append(np.std(seds, axis=0))
        outputs["params_spread"].append(np.std(params, axis=0))
    return {
        key: np.concatenate(value).astype(np.float32) for key, value in outputs.items()
    }


# --------------------------------------------------------------------------- #
# Per-parameter heads
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_parameter_heads(
    model: ParameterHeads, inputs: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray | None, dict[str, np.ndarray]]:
    """Point estimates, error bars and class probabilities.

    A classification head returns the grid value it rates most likely, its
    error bar is the standard deviation of its probabilities over the grid,
    so a hesitant head reports a wide error bar, like a regression head.
    Error bars are in standardised units (``None`` without uncertainty heads).
    """
    model.eval()
    x = torch.tensor(inputs, dtype=torch.float32, device=device)
    outputs = model(x)
    n_params = max(spec["index"] for spec in model.specs) + 1
    values = np.zeros((len(inputs), n_params), dtype=np.float32)
    sigma = np.zeros_like(values) if model.uncertainty else None
    probabilities: dict[str, np.ndarray] = {}
    for k, spec in enumerate(model.specs):
        j = spec["index"]
        if spec["type"] == "cls":
            probs = torch.softmax(outputs[k], dim=1)
            levels = torch.tensor(
                spec["levels_std"], dtype=probs.dtype, device=probs.device
            )
            values[:, j] = levels[torch.argmax(probs, dim=1)].cpu().numpy()
            probabilities[spec["name"]] = probs.cpu().numpy().astype(np.float32)
            if sigma is not None:
                mean = (probs * levels).sum(dim=1)
                variance = (probs * torch.square(levels - mean.unsqueeze(1))).sum(dim=1)
                sigma[:, j] = torch.sqrt(variance.clamp_min(0.0)).cpu().numpy()
        elif sigma is not None:
            values[:, j] = outputs[k][:, 0].cpu().numpy()
            sigma[:, j] = (
                torch.exp(0.5 * outputs[k][:, 1].clamp(*LOG_VAR_RANGE)).cpu().numpy()
            )
        else:
            values[:, j] = outputs[k].squeeze(1).cpu().numpy()
    return values, sigma, probabilities


# --------------------------------------------------------------------------- #
# Conditional WGAN-GP
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sample_cwgan(
    generator: CWGANGenerator,
    conditions: np.ndarray,
    cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Average SED and parameters over ``cfg["draws"]`` noise draws per galaxy.

    The spread over draws is returned as a diagnostic, it is not a calibrated
    uncertainty.
    """
    seed_everything(cfg["seed"])
    generator.eval()
    latent_dim = generator.trunk[0].in_channels - conditions.shape[1]
    outputs = {"sed": [], "params": [], "sed_spread": [], "params_spread": []}
    for start in range(0, len(conditions), cfg["chunk_size"]):
        cond = torch.tensor(
            conditions[start : start + cfg["chunk_size"]],
            dtype=torch.float32,
            device=device,
        )
        seds, params = [], []
        for _ in range(cfg["draws"]):
            sed, param = generator(
                torch.randn(cond.shape[0], latent_dim, device=device), cond
            )
            seds.append(sed)
            params.append(param)
        sed_stack, param_stack = torch.stack(seds), torch.stack(params)
        outputs["sed"].append(sed_stack.mean(0).cpu().numpy())
        outputs["params"].append(param_stack.mean(0).cpu().numpy())
        outputs["sed_spread"].append(sed_stack.std(0, unbiased=False).cpu().numpy())
        outputs["params_spread"].append(
            param_stack.std(0, unbiased=False).cpu().numpy()
        )
    return {
        key: np.concatenate(value).astype(np.float32) for key, value in outputs.items()
    }
