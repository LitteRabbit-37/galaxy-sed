"""Step 8: evaluate every route on one split and write the results tables.

Reads the predictions of steps 2, 4, 5 and 7 that are present in the working
directory and writes ``results_<split>.json`` and ``results_<split>.md``.
With ``--figure``, also draws the reconstruction of a representative galaxy,
the one whose GLO reconstruction has the median chi-square.

    uv run python scripts/08_evaluate.py --split test --figure docs/example_reconstruction.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import r2_score

from galaxy_sed.data import destandardise_sed, load_prepared, load_preprocessing
from galaxy_sed.io import (
    PREPARED,
    PREPROCESSING,
    RECONSTRUCTIONS,
    common_arguments,
    direct_mlp_predictions_file,
    parameter_heads_file,
    write_json,
)
from galaxy_sed.metrics import bolometric_luminosity, parameter_scores, rissaki_chi2, sed_scores


def load_routes(workdir: Path, split: str) -> tuple[dict, dict, dict]:
    """Collect the available SED and parameter predictions for one split."""
    seds, params, timing = {}, {}, {}
    glo_path = workdir / RECONSTRUCTIONS / f"glo_{split}.npz"
    if glo_path.exists():
        glo = np.load(glo_path)
        seds["GLO (predicted redshift)"] = glo["sed"]
        timing["GLO (predicted redshift)"] = float(glo["seconds_per_galaxy"])
    cwgan_path = workdir / f"cwgan_{split}.npz"
    if cwgan_path.exists():
        cwgan = np.load(cwgan_path)
        seds["cWGAN-GP"] = cwgan["sed"]
        timing["cWGAN-GP"] = float(cwgan["seconds_per_galaxy"])
    for mode, label in (("photometry", "Direct MLP, photometry"), ("hybrid", "Direct MLP, photometry + SED")):
        path = workdir / direct_mlp_predictions_file(mode)
        if path.exists():
            params[label] = np.load(path)[split]
    for mode, label in (("photometry", "Heads, photometry"), ("hybrid", "Heads, photometry + SED")):
        path = workdir / f"parameter_heads_{mode}_predictions.npz"
        if path.exists():
            params[label] = np.load(path)[f"{split}_values"]
    if cwgan_path.exists():
        params["cWGAN-GP"] = cwgan["params"]
    return seds, params, timing


def uncertainty_coverage(workdir: Path, split: str, truth_std: np.ndarray, flag: np.ndarray,
                         preprocessing: dict) -> dict:
    """Fraction of galaxies whose truth lies within 1 and 2 error bars.

    Only regression heads are scored: a classification head expresses its
    uncertainty as probabilities over a grid, not as a Gaussian error bar.
    """
    out = {}
    for mode in ("photometry", "hybrid"):
        path = workdir / f"parameter_heads_{mode}_predictions.npz"
        if not path.exists() or f"{split}_sigma" not in np.load(path).files:
            continue
        predictions = np.load(path)
        values, sigma = predictions[f"{split}_values"], predictions[f"{split}_sigma"]
        specs = torch.load(workdir / parameter_heads_file(mode), weights_only=True)["init_kwargs"]["specs"]
        within_1, within_2 = [], []
        for spec in specs:
            if spec["type"] != "reg":
                continue
            j, name = spec["index"], spec["name"]
            valid = flag.astype(bool) if name in preprocessing["polar_dependent"] else np.ones(len(flag), bool)
            error = np.abs(values[valid, j] - truth_std[valid, j])
            within_1.append(float(np.mean(error <= sigma[valid, j])))
            within_2.append(float(np.mean(error <= 2 * sigma[valid, j])))
        if within_1:
            out[mode] = {"within_1_sigma": float(np.mean(within_1)), "within_2_sigma": float(np.mean(within_2)),
                         "n_parameters": len(within_1)}
    return out


def markdown(results: dict, preprocessing: dict) -> str:
    lines = [f"## Results on the {results['split']} split ({results['n_galaxies']} galaxies)", ""]
    lines += ["### Full SED", "", "| Route | R^2 | chi^2 < 5 | median chi^2 | time per galaxy |",
              "|---|---:|---:|---:|---:|"]
    for route, s in results["sed"].items():
        seconds = results["seconds_per_galaxy"].get(route)
        timing = f"{seconds * 1e3:.2f} ms" if seconds else ""
        lines.append(f"| {route} | {s['r2']:.4f} | {100 * s['fraction_chi2_below_5']:.1f} % | "
                     f"{s['chi2_median']:.2f} | {timing} |")
    lines += ["", "### Bolometric luminosity (integral over frequency)", "", "| Route | R^2 on log10 L |", "|---|---:|"]
    lines += [f"| {route} | {r2:.4f} |" for route, r2 in results["luminosity_r2"].items()]
    routes = list(results["parameters"])
    lines += ["", "### Physical parameters", "",
              "R^2 for continuous parameters. For a parameter sampled on K values, balanced accuracy (acc), "
              "whose chance level is 1/K.", "",
              "| Parameter | " + " | ".join(routes) + " |", "|---|" + "---:|" * len(routes)]
    for name in preprocessing["param_names"]:
        cells, n_values = [], None
        for route in routes:
            record = results["parameters"][route][name]
            if "balanced_accuracy" in record:
                n_values = record["n_grid_values"]
                cells.append(f"acc {record['balanced_accuracy']:.2f}")
            else:
                cells.append(f"{record['r2']:.2f}" if record["r2"] is not None else "n/a")
        label = f"{name} (K = {n_values})" if n_values else name
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    if results["uncertainty_coverage"]:
        lines += ["", "### Error bars of the parameter heads (continuous parameters)", "",
                  "| Input | truth within 1 sigma | truth within 2 sigma |", "|---|---:|---:|"]
        for mode, c in results["uncertainty_coverage"].items():
            lines.append(f"| {mode} | {100 * c['within_1_sigma']:.1f} % | {100 * c['within_2_sigma']:.1f} % |")
        lines.append("")
        lines.append("A calibrated Gaussian error bar contains the truth 68.3 % and 95.4 % of the time.")
    return "\n".join(lines) + "\n"


def draw_example(path: Path, data: dict, preprocessing: dict, split: str, seds: dict) -> None:
    """Truth, reconstructions and photometry for the galaxy with the median GLO chi-square."""
    idx = data[f"idx_{split}"]
    wave = np.asarray(preprocessing["wave_micron"])
    truth = destandardise_sed(data["Y_sed"][idx], preprocessing)
    glo_label = "GLO (predicted redshift)"
    chi2 = rissaki_chi2(truth, destandardise_sed(seds[glo_label], preprocessing), wave)
    k = int(np.argsort(chi2)[len(chi2) // 2])
    galaxy = idx[k]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    order = np.argsort(wave)
    ax.plot(wave[order], truth[k][order], color="black", lw=1.6, label="True SED")
    colours = {"GLO (predicted redshift)": "#8E44AD", "cWGAN-GP": "#E67E22"}
    for label, sed in seds.items():
        ax.plot(wave[order], destandardise_sed(sed[k], preprocessing)[order], lw=1.2, ls="--",
                color=colours.get(label), label=label)
    mask = data["X_mask"][galaxy].astype(bool)
    bands = np.asarray(preprocessing["band_wavelengths_micron"]) / (1.0 + data["redshift"][galaxy])
    fluxes = 10 ** (data["X_flux"][galaxy] * np.asarray(preprocessing["flux_log_std"])
                    + np.asarray(preprocessing["flux_log_mean"]))
    ax.plot(bands[mask], fluxes[mask], "o", color="#2471A3", ms=6,
            label=f"Input photometry ({int(mask.sum())} of {mask.size} bands inside the grid)")
    ax.set(xscale="log", yscale="log", xlabel="Rest-frame wavelength (micron)",
           ylabel="Flux density S_nu (arbitrary units)",
           title=f"Test galaxy at z = {data['redshift'][galaxy]:.2f} (median GLO chi-square {chi2[k]:.2f})")
    ax.grid(alpha=0.3, which="both", lw=0.4)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = common_arguments(__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--figure", type=Path, help="path of an example reconstruction figure")
    args = parser.parse_args()
    data = load_prepared(args.workdir / PREPARED)
    preprocessing = load_preprocessing(args.workdir / PREPROCESSING)
    idx = data[f"idx_{args.split}"]
    truth_sed, truth_params = data["Y_sed"][idx], data["Y_params"][idx]
    flag = data["Y_flag_pol"][idx]
    train_params = data["Y_params"][data["idx_train"]]
    seds, params, timing = load_routes(args.workdir, args.split)

    wave = preprocessing["wave_micron"]
    true_lum = np.log10(bolometric_luminosity(destandardise_sed(truth_sed, preprocessing), wave))
    results = {
        "split": args.split,
        "n_galaxies": int(idx.size),
        "sed": {route: sed_scores(truth_sed, sed, preprocessing) for route, sed in seds.items()},
        "luminosity_r2": {
            route: float(r2_score(true_lum, np.log10(bolometric_luminosity(destandardise_sed(sed, preprocessing), wave))))
            for route, sed in seds.items()
        },
        "parameters": {route: parameter_scores(truth_params, pred, preprocessing, flag, train_params)
                       for route, pred in params.items()},
        "uncertainty_coverage": uncertainty_coverage(args.workdir, args.split, truth_params, flag, preprocessing),
        "seconds_per_galaxy": timing,
    }
    write_json(args.workdir / f"results_{args.split}.json", results)
    table = markdown(results, preprocessing)
    (args.workdir / f"results_{args.split}.md").write_text(table, encoding="utf-8")
    print(table)
    if args.figure and "GLO (predicted redshift)" in seds:
        draw_example(args.figure, data, preprocessing, args.split, seds)
        print(f"Figure saved to {args.figure}")


if __name__ == "__main__":
    main()
