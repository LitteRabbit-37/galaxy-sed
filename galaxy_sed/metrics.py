"""Evaluation metrics.

* ``sed_scores``: R^2 over all wavelengths of the standardised log SED, and
  the chi-square of Rissaki et al. (2024, Eq. 6) on the rest-frame 5 to 35
  micron interval, with a 10 per cent uncertainty on both the reconstruction
  and the truth. A galaxy counts as well reconstructed when chi^2 < 5.
* ``parameter_scores``: per parameter, R^2 in the modelled coordinates (log10
  for scale parameters) and, for parameters sampled on a small grid, the
  balanced accuracy of the nearest grid value.
* ``bolometric_luminosity``: integral of the SED over frequency.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, r2_score

from galaxy_sed.data import destandardise_params, destandardise_sed

SPEED_OF_LIGHT_MICRON_PER_S = 2.99792458e14


def rissaki_chi2(
    true_flux: np.ndarray, pred_flux: np.ndarray, wave_micron: np.ndarray
) -> np.ndarray:
    """Per-galaxy chi-square on the rest-frame 5 to 35 micron interval."""
    wave = np.asarray(wave_micron, dtype=np.float64)
    window = (wave >= 5.0) & (wave <= 35.0)
    truth = np.asarray(true_flux, dtype=np.float64)[:, window]
    pred = np.asarray(pred_flux, dtype=np.float64)[:, window]
    variance = np.maximum(
        np.square(0.1 * pred) + np.square(0.1 * truth), np.finfo(np.float64).tiny
    )
    return np.mean(np.square(pred - truth) / variance, axis=1)


def sed_scores(
    true_std: np.ndarray, pred_std: np.ndarray, preprocessing: dict[str, Any]
) -> dict[str, float]:
    chi2 = rissaki_chi2(
        destandardise_sed(true_std, preprocessing),
        destandardise_sed(pred_std, preprocessing),
        preprocessing["wave_micron"],
    )
    truth = np.asarray(true_std, dtype=np.float64).ravel()
    pred = np.asarray(pred_std, dtype=np.float64).ravel()
    return {
        "r2": float(r2_score(truth, pred)),
        "chi2_median": float(np.median(chi2)),
        "fraction_chi2_below_5": float(np.mean(chi2 < 5.0)),
    }


def _balanced_accuracy(
    true_values: np.ndarray, pred_values: np.ndarray, grid: np.ndarray
) -> float:
    """Balanced accuracy after snapping both values to the nearest grid value."""
    true_class = np.argmin(np.abs(true_values[:, None] - grid[None, :]), axis=1)
    pred_class = np.argmin(np.abs(pred_values[:, None] - grid[None, :]), axis=1)
    cm = confusion_matrix(true_class, pred_class, labels=np.arange(grid.size))
    present = cm.sum(axis=1) > 0
    return float(np.mean(np.diag(cm)[present] / cm.sum(axis=1)[present]))


def parameter_scores(
    true_std: np.ndarray,
    pred_std: np.ndarray,
    preprocessing: dict[str, Any],
    polar_flag: np.ndarray,
    train_std: np.ndarray,
    max_grid_classes: int = 20,
) -> dict[str, dict[str, Any]]:
    """Per-parameter scores, polar-dust parameters only where polar dust is present."""
    names = preprocessing["param_names"]
    truth = destandardise_params(true_std, preprocessing)
    pred = destandardise_params(pred_std, preprocessing)
    train = destandardise_params(train_std, preprocessing)
    scores = {}
    for j, name in enumerate(names):
        valid = (
            np.asarray(polar_flag, dtype=bool)
            if name in preprocessing["polar_dependent"]
            else np.ones(len(truth), dtype=bool)
        )
        yt, yp = (
            np.asarray(true_std, dtype=np.float64)[valid, j],
            np.asarray(pred_std, dtype=np.float64)[valid, j],
        )
        record: dict[str, Any] = {
            "n": int(valid.sum()),
            "r2": float(r2_score(yt, yp)) if yt.size >= 2 and np.std(yt) > 0 else None,
        }
        grid = np.unique(np.round(train[:, j], decimals=8))
        if 2 <= grid.size <= max_grid_classes and valid.any():
            record["n_grid_values"] = int(grid.size)
            record["balanced_accuracy"] = _balanced_accuracy(
                truth[valid, j], pred[valid, j], grid
            )
        scores[name] = record
    return scores


def bolometric_luminosity(flux_snu: np.ndarray, wave_micron: np.ndarray) -> np.ndarray:
    """Integral of S_nu over frequency, in the flux units times hertz.

    The simulated SEDs are flux densities per unit frequency, so the total is
    the integral over nu = c / lambda, not over lambda.
    """
    wave = np.asarray(wave_micron, dtype=np.float64)
    order = np.argsort(
        SPEED_OF_LIGHT_MICRON_PER_S / wave, kind="mergesort"
    )  # ascending frequency
    nu = SPEED_OF_LIGHT_MICRON_PER_S / wave[order]
    return np.trapezoid(np.asarray(flux_snu, dtype=np.float64)[..., order], nu, axis=-1)
