"""Data preparation: synthetic photometry, splits and normalisation.

The training data are simulated galaxies: a rest-frame spectral energy
distribution (SED) on a fixed wavelength grid, plus the physical parameters
used to generate it. This module turns them into the arrays the models use.

* ``X_flux``   20 photometric fluxes, log10 then standardised per band.
* ``X_mask``   1 where the band falls inside the rest-frame grid, else 0.
* ``Y_sed``    full SED, log10 then standardised per wavelength.
* ``Y_params`` physical parameters, log10 where configured, then
  standardised per parameter.

Every normalisation statistic is computed on the training split only and then
applied unchanged to the validation and test splits, so nothing about the
evaluation galaxies leaks into training.

Units. The SED values are flux densities per unit frequency (S_nu) in the
arbitrary units of the simulations. Photometry passed to the pretrained models
must be expressed in the same units (see ``synthetic_photometry``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# Columns with a spread below this value are constant: they keep a unit scale
# so that standardisation never divides by zero.
STD_FLOOR = 1e-12


# --------------------------------------------------------------------------- #
# Configuration helpers
# --------------------------------------------------------------------------- #
def load_config(path: str | Path) -> dict[str, Any]:
    """Read the JSON configuration file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def band_names(config: dict[str, Any]) -> list[str]:
    return [band["name"] for band in config["bands"]]


def band_wavelengths(config: dict[str, Any]) -> np.ndarray:
    """Observed-frame band wavelengths in micron, shape (n_bands,)."""
    return np.array(
        [band["wavelength_micron"] for band in config["bands"]], dtype=np.float64
    )


# --------------------------------------------------------------------------- #
# Photometry
# --------------------------------------------------------------------------- #
def synthetic_photometry(
    wave: np.ndarray,
    flux: np.ndarray,
    redshift: np.ndarray,
    bands: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample each rest-frame SED at the rest wavelengths of the observed bands.

    A band observed at ``lambda_obs`` sees the rest-frame wavelength
    ``lambda_obs / (1 + z)``. The SED is interpolated linearly there. Bands
    whose rest wavelength falls outside the SED grid get NaN and mask 0, at
    high redshift this happens to the ultraviolet bands.

    Parameters
    ----------
    wave : (n_wave,) rest-frame wavelength grid in micron, in any order.
    flux : (n_galaxies, n_wave) SED values on that grid.
    redshift : (n_galaxies,) redshift of each galaxy.
    bands : (n_bands,) observed-frame band wavelengths in micron.

    Returns
    -------
    photometry : (n_galaxies, n_bands) float64, NaN outside the grid.
    mask : (n_galaxies, n_bands) int8, 1 inside the grid.
    """
    wave = np.asarray(wave, dtype=np.float64)
    flux = np.atleast_2d(np.asarray(flux, dtype=np.float64))
    order = np.argsort(wave, kind="mergesort")
    wave_asc, flux_asc = wave[order], flux[:, order]

    z = np.asarray(redshift, dtype=np.float64).reshape(-1, 1)
    lam_rest = np.asarray(bands, dtype=np.float64)[None, :] / (1.0 + z)
    inside = (lam_rest >= wave_asc[0]) & (lam_rest <= wave_asc[-1])

    # Index of the grid interval containing each query point, then the usual
    # linear interpolation y_lo + slope * (x - x_lo).
    upper = np.clip(
        np.searchsorted(wave_asc, lam_rest, side="left"), 1, wave_asc.size - 1
    )
    lower = upper - 1
    rows = np.arange(flux_asc.shape[0])[:, None]
    slope = (flux_asc[rows, upper] - flux_asc[rows, lower]) / (
        wave_asc[upper] - wave_asc[lower]
    )
    photometry = slope * (lam_rest - wave_asc[lower]) + flux_asc[rows, lower]
    photometry[~inside] = np.nan
    return photometry, inside.astype(np.int8)


# --------------------------------------------------------------------------- #
# Splits and standardisation
# --------------------------------------------------------------------------- #
def split_indices(
    n: int, seed: int, fractions: tuple[float, float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random train/validation/test partition, each part sorted."""
    permutation = np.random.default_rng(seed).permutation(n)
    n_train = int(fractions[0] * n)
    n_val = int(fractions[1] * n)
    return (
        np.sort(permutation[:n_train]),
        np.sort(permutation[n_train : n_train + n_val]),
        np.sort(permutation[n_train + n_val :]),
    )


def mean_std(
    values: np.ndarray, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean and population standard deviation.

    With ``mask``, only entries where the mask is 1 contribute. Constant
    columns get a unit standard deviation.
    """
    if mask is None:
        mean = values.mean(axis=0)
        std = values.std(axis=0)
    else:
        valid = mask.astype(bool)
        counts = np.maximum(valid.sum(axis=0).astype(np.float64), 1.0)
        mean = np.where(valid, values, 0.0).sum(axis=0) / counts
        std = np.sqrt(np.where(valid, (values - mean) ** 2, 0.0).sum(axis=0) / counts)
    std = np.where(std < STD_FLOOR, 1.0, std)
    return mean, std


def target_column(raw: Any, name: str) -> np.ndarray:
    """Read one target from a raw file. ``AGNpar_<k>`` is column k of ``AGNpars``."""
    if name.startswith("AGNpar_"):
        return np.asarray(raw["AGNpars"], dtype=np.float64)[:, int(name.split("_")[1])]
    return np.asarray(raw[name], dtype=np.float64)


def log_photometry(photometry: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """log10 of the valid fluxes, NaN elsewhere."""
    out = np.full(photometry.shape, np.nan, dtype=np.float64)
    valid = mask.astype(bool)
    out[valid] = np.log10(photometry[valid])
    return out


# --------------------------------------------------------------------------- #
# Dataset preparation
# --------------------------------------------------------------------------- #
def prepare_dataset(
    raw_path: str | Path, config: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Build the training arrays from a raw simulation file.

    The raw ``.npz`` file must contain ``wave`` (n_wave,), ``flux``
    (n_galaxies, n_wave), ``redshift`` (n_galaxies,) and one array per target
    listed in the configuration. Optional arrays: ``sb_on`` (starburst flag,
    used for selection), ``polar_on`` (polar dust flag, used to mask the
    polar-dependent targets) and ``AGNpars`` (n_galaxies, 4).
    """
    data_cfg = config["data"]
    raw = np.load(raw_path, allow_pickle=False)
    wave = np.asarray(raw["wave"], dtype=np.float64)
    flux = np.asarray(raw["flux"], dtype=np.float64)
    redshift = np.asarray(raw["redshift"], dtype=np.float64)

    # Galaxies without a starburst are excluded, as in the reference study.
    keep = np.ones(redshift.size, dtype=bool)
    if data_cfg.get("require_starburst", False) and "sb_on" in raw.files:
        keep &= np.asarray(raw["sb_on"]) == 1
    flux, redshift = flux[keep], redshift[keep]
    n = int(keep.sum())

    idx_train, idx_val, idx_test = split_indices(
        n, int(data_cfg["split_seed"]), tuple(data_cfg["split_fractions"])
    )

    # Inputs: photometry, log10, standardised with training statistics only.
    bands = band_wavelengths(config)
    photometry, mask = synthetic_photometry(wave, flux, redshift, bands)
    log_phot = log_photometry(photometry, mask)
    flux_mean, flux_std = mean_std(log_phot[idx_train], mask[idx_train])
    x_flux = np.where(mask == 1, (log_phot - flux_mean) / flux_std, 0.0)

    # Target 1: the full SED.
    sed_log = np.log10(np.clip(flux, 1e-30, None))
    sed_mean, sed_std = mean_std(sed_log[idx_train])
    y_sed = (sed_log - sed_mean) / sed_std

    # Target 2: the physical parameters. Scale-like parameters spanning
    # several decades are modelled in log10, zeros (absent components) are
    # floored at 1 before the logarithm.
    names, use_log, columns = [], [], []
    for target in data_cfg["targets"]:
        column = target_column(raw, target["name"])[keep]
        if target["log10"]:
            column = np.log10(np.clip(column, 1.0, None))
        names.append(target["name"])
        use_log.append(bool(target["log10"]))
        columns.append(column)
    params_raw = np.stack(columns, axis=1)
    params_mean, params_std = mean_std(params_raw[idx_train])
    y_params = (params_raw - params_mean) / params_std

    if "polar_on" in raw.files:
        flag_pol = np.asarray(raw["polar_on"])[keep].astype(np.int8)
    else:
        flag_pol = np.ones(n, dtype=np.int8)

    return {
        "X_flux": x_flux.astype(np.float32),
        "X_mask": mask.astype(np.int8),
        "Y_sed": y_sed.astype(np.float32),
        "Y_params": y_params.astype(np.float32),
        "Y_flag_pol": flag_pol,
        "flux_input_mean": flux_mean.astype(np.float32),
        "flux_input_std": flux_std.astype(np.float32),
        "sed_mean": sed_mean.astype(np.float32),
        "sed_std": sed_std.astype(np.float32),
        "params_mean": params_mean.astype(np.float32),
        "params_std": params_std.astype(np.float32),
        "param_names": np.array(names),
        "param_use_log": np.array(use_log, dtype=bool),
        "polar_dependent": np.array(data_cfg.get("polar_dependent_targets", [])),
        "idx_train": idx_train.astype(np.int64),
        "idx_val": idx_val.astype(np.int64),
        "idx_test": idx_test.astype(np.int64),
        "wave": wave.astype(np.float32),
        "band_wavelengths": bands.astype(np.float32),
        "band_names": np.array(band_names(config)),
        "redshift": redshift.astype(np.float32),
    }


def save_prepared(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_prepared(path: str | Path) -> dict[str, np.ndarray]:
    """Load a prepared dataset fully into memory."""
    with np.load(path, allow_pickle=False) as npz:
        return {key: npz[key] for key in npz.files}


# --------------------------------------------------------------------------- #
# Model inputs
# --------------------------------------------------------------------------- #
def photometry_inputs(data: dict[str, np.ndarray], idx: np.ndarray) -> np.ndarray:
    """The 40 model inputs: 20 standardised fluxes followed by 20 mask bits."""
    return np.concatenate(
        [
            data["X_flux"][idx].astype(np.float32),
            data["X_mask"][idx].astype(np.float32),
        ],
        axis=1,
    )


def parameter_mask(data: dict[str, np.ndarray], idx: np.ndarray) -> np.ndarray:
    """1 where a target is meaningful, polar-dust targets are 0 without polar dust."""
    names = [str(name) for name in data["param_names"]]
    mask = np.ones((len(idx), len(names)), dtype=np.float32)
    flag = data["Y_flag_pol"][idx].astype(np.float32)
    for name in data["polar_dependent"]:
        if str(name) in names:
            mask[:, names.index(str(name))] = flag
    return mask


# --------------------------------------------------------------------------- #
# Preprocessing statistics shipped with the models
# --------------------------------------------------------------------------- #
def preprocessing_from_prepared(data: dict[str, np.ndarray]) -> dict[str, Any]:
    """Everything needed to prepare new photometry and read model outputs."""
    train_z = data["redshift"][data["idx_train"]]
    return {
        "band_names": [str(name) for name in data["band_names"]],
        "band_wavelengths_micron": data["band_wavelengths"].astype(float).tolist(),
        "wave_micron": data["wave"].astype(float).tolist(),
        "flux_log_mean": data["flux_input_mean"].astype(float).tolist(),
        "flux_log_std": data["flux_input_std"].astype(float).tolist(),
        "sed_log_mean": data["sed_mean"].astype(float).tolist(),
        "sed_log_std": data["sed_std"].astype(float).tolist(),
        "param_names": [str(name) for name in data["param_names"]],
        "param_log10": data["param_use_log"].astype(bool).tolist(),
        "param_mean": data["params_mean"].astype(float).tolist(),
        "param_std": data["params_std"].astype(float).tolist(),
        "polar_dependent": [str(name) for name in data["polar_dependent"]],
        "redshift_range": [float(train_z.min()), float(train_z.max())],
    }


def save_preprocessing(path: str | Path, preprocessing: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(preprocessing, indent=2) + "\n", encoding="utf-8")


def load_preprocessing(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def standardise_photometry(
    photometry: np.ndarray, preprocessing: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Turn raw band fluxes into model inputs.

    Missing or non-positive fluxes (NaN, zero or negative) are masked.
    Returns the standardised fluxes (0 where masked) and the mask.
    """
    photometry = np.atleast_2d(np.asarray(photometry, dtype=np.float64))
    mask = (np.isfinite(photometry) & (photometry > 0)).astype(np.int8)
    log_phot = log_photometry(np.where(mask == 1, photometry, 1.0), mask)
    mean = np.asarray(preprocessing["flux_log_mean"])
    std = np.asarray(preprocessing["flux_log_std"])
    x_flux = np.where(mask == 1, (log_phot - mean) / std, 0.0)
    return x_flux.astype(np.float32), mask


def destandardise_sed(sed_std: np.ndarray, preprocessing: dict[str, Any]) -> np.ndarray:
    """Standardised SED to physical flux on the wavelength grid."""
    log_flux = np.asarray(sed_std, dtype=np.float64) * np.asarray(
        preprocessing["sed_log_std"]
    ) + np.asarray(preprocessing["sed_log_mean"])
    return np.power(10.0, log_flux)


def destandardise_params(
    params_std: np.ndarray, preprocessing: dict[str, Any]
) -> np.ndarray:
    """Standardised parameters to physical values (undoing the log10 transforms)."""
    raw = np.asarray(params_std, dtype=np.float64) * np.asarray(
        preprocessing["param_std"]
    ) + np.asarray(preprocessing["param_mean"])
    use_log = np.asarray(preprocessing["param_log10"], dtype=bool)
    out = raw.copy()
    out[..., use_log] = np.power(10.0, raw[..., use_log])
    return out
