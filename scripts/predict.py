"""Predict SEDs and physical parameters from 20-band photometry.

Uses the models in ``--models`` (the published ``weights/`` by default, or a
working directory produced by steps 1 to 6).

Input, one row per galaxy:
* a CSV file with one column per band, named as in ``preprocessing.json``
* or a ``.npz`` file with an array ``photometry`` of shape (n_galaxies, 20).
Missing bands are empty cells, NaN or non-positive values. Fluxes must be in
the units of the training simulations (see the README).

Output:
* ``<output>.npz``: redshift, SEDs (GLO and cWGAN-GP) on the wavelength grid,
  parameters from every route (direct MLP and heads, each with the photometry
  alone and with the reconstructed SED, and cWGAN-GP), error bars of the heads,
  bolometric luminosities
* ``<output>.csv``: redshift and parameters of the photometry heads with
  their error bars, one row per galaxy.

    uv run python scripts/predict.py --input my_photometry.csv --output predictions
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from galaxy_sed.data import (
    destandardise_params,
    destandardise_sed,
    load_config,
    load_preprocessing,
    standardise_photometry,
)
from galaxy_sed.inference import (
    invert_glo,
    predict_batched,
    predict_parameter_heads,
    predict_redshift,
    sample_cwgan,
)
from galaxy_sed.io import (
    CWGAN,
    DIRECT_MLP,
    GLO,
    PREPROCESSING,
    default_device,
    direct_mlp_file,
    load_model,
    parameter_heads_file,
)
from galaxy_sed.metrics import bolometric_luminosity


def read_photometry(path: Path, band_names: list[str]) -> np.ndarray:
    if path.suffix == ".npz":
        return np.load(path)["photometry"].astype(np.float64)
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    missing = [name for name in band_names if name not in rows[0]]
    if missing:
        raise SystemExit(f"Missing band columns in {path}: {', '.join(missing)}")
    return np.array(
        [
            [float(row[name]) if row[name].strip() else np.nan for name in band_names]
            for row in rows
        ]
    )


def physical_sigma(sigma_std: np.ndarray, preprocessing: dict) -> np.ndarray:
    """Error bars in physical units for linear parameters, in dex for log10 parameters."""
    return sigma_std * np.asarray(preprocessing["param_std"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input", type=Path, required=True, help="CSV or .npz photometry file"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="output path without extension"
    )
    parser.add_argument(
        "--models",
        type=Path,
        default=Path("weights"),
        help="directory with the model files",
    )
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    preprocessing = load_preprocessing(args.models / PREPROCESSING)
    names = preprocessing["param_names"]
    device = default_device()
    photometry = read_photometry(args.input, preprocessing["band_names"])
    x_flux, mask = standardise_photometry(photometry, preprocessing)
    inputs = np.concatenate([x_flux, mask.astype(np.float32)], axis=1)
    print(f"{len(inputs)} galaxies, {int(mask.sum())} valid fluxes, device {device}")

    out = {
        "wave_micron": np.asarray(preprocessing["wave_micron"]),
        "param_names": np.array(names),
    }

    mlp, _ = load_model(args.models / DIRECT_MLP, device)
    mlp_std = predict_batched(mlp, inputs, device)
    out["redshift"] = predict_redshift(mlp_std, names, preprocessing)
    out["params_direct_mlp"] = destandardise_params(mlp_std, preprocessing)

    glo_sed_std = None
    if (args.models / GLO).exists():
        glo, _ = load_model(args.models / GLO, device)
        print("GLO inversion...")
        glo_result = invert_glo(
            glo,
            x_flux,
            mask.astype(np.float32),
            out["redshift"],
            preprocessing,
            config["inversion"],
            device,
        )
        glo_sed_std = glo_result["sed"]
        out["sed_glo"] = destandardise_sed(glo_sed_std, preprocessing)
        out["luminosity_glo"] = bolometric_luminosity(
            out["sed_glo"], out["wave_micron"]
        )
        if (args.models / direct_mlp_file("hybrid")).exists():
            hybrid_mlp, _ = load_model(args.models / direct_mlp_file("hybrid"), device)
            hybrid_std = predict_batched(
                hybrid_mlp, np.concatenate([inputs, glo_sed_std], axis=1), device
            )
            out["params_direct_mlp_hybrid"] = destandardise_params(
                hybrid_std, preprocessing
            )

    for mode in ("photometry", "hybrid"):
        path = args.models / parameter_heads_file(mode)
        if not path.exists() or (mode == "hybrid" and glo_sed_std is None):
            continue
        heads, _ = load_model(path, device)
        head_inputs = (
            inputs
            if mode == "photometry"
            else np.concatenate([inputs, glo_sed_std], axis=1)
        )
        values, sigma, _ = predict_parameter_heads(heads, head_inputs, device)
        out[f"params_heads_{mode}"] = destandardise_params(values, preprocessing)
        if sigma is not None:
            out[f"sigma_heads_{mode}"] = physical_sigma(sigma, preprocessing)
    out["sigma_units"] = np.array(
        ["dex" if log else "linear" for log in preprocessing["param_log10"]]
    )

    if (args.models / CWGAN).exists():
        generator, _ = load_model(args.models / CWGAN, device)
        cwgan = sample_cwgan(generator, inputs, config["cwgan"], device)
        out["sed_cwgan"] = destandardise_sed(cwgan["sed"], preprocessing)
        out["params_cwgan"] = destandardise_params(cwgan["params"], preprocessing)
        out["luminosity_cwgan"] = bolometric_luminosity(
            out["sed_cwgan"], out["wave_micron"]
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output.with_suffix(".npz"), **out)
    if "params_heads_photometry" in out:
        with open(
            args.output.with_suffix(".csv"), "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            sigma = out.get("sigma_heads_photometry")
            header = ["galaxy", "redshift_direct_mlp"]
            for name in names:
                header += [name] + ([f"{name}_sigma"] if sigma is not None else [])
            writer.writerow(header)
            for i in range(len(inputs)):
                row = [i, f"{out['redshift'][i]:.6g}"]
                for j in range(len(names)):
                    row.append(f"{out['params_heads_photometry'][i, j]:.6g}")
                    if sigma is not None:
                        row.append(f"{sigma[i, j]:.6g}")
                writer.writerow(row)
    print(
        f"Saved {args.output.with_suffix('.npz')} and {args.output.with_suffix('.csv')}"
    )


if __name__ == "__main__":
    main()
