"""Step 4: reconstruct full SEDs by GLO inversion.

For each galaxy, the redshift comes from the direct MLP (step 2), never from
the truth, so the reconstruction faces the same conditions as a real survey.
Reconstructions of the training and validation sets are inputs of the hybrid
parameter heads (step 5), the test set is used for evaluation (step 8).

Writes ``reconstructions/glo_<split>.npz`` with the standardised SED, the
generator's parameters, the spread over restarts and the redshift used.

    uv run python scripts/04_reconstruct_sed.py
    uv run python scripts/04_reconstruct_sed.py --splits test
"""

import time

import numpy as np

from galaxy_sed.data import load_config, load_prepared, load_preprocessing
from galaxy_sed.inference import invert_glo, predict_redshift
from galaxy_sed.io import (
    DIRECT_MLP_PREDICTIONS,
    GLO,
    PREPARED,
    PREPROCESSING,
    RECONSTRUCTIONS,
    common_arguments,
    default_device,
    load_model,
)


def main() -> None:
    parser = common_arguments(__doc__.splitlines()[0])
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--redshift",
        choices=["predicted", "true"],
        default="predicted",
        help="predicted: direct MLP estimate (deployment), true: catalogue value",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)["inversion"]
    data = load_prepared(args.workdir / PREPARED)
    preprocessing = load_preprocessing(args.workdir / PREPROCESSING)
    names = preprocessing["param_names"]
    device = default_device()
    generator, _ = load_model(args.workdir / GLO, device)
    mlp_predictions = (
        np.load(args.workdir / DIRECT_MLP_PREDICTIONS)
        if args.redshift == "predicted"
        else None
    )

    for split in args.splits:
        idx = data[f"idx_{split}"]
        if args.redshift == "predicted":
            redshift = predict_redshift(mlp_predictions[split], names, preprocessing)
        else:
            redshift = data["redshift"][idx].astype(np.float32)
        print(f"Inverting {split} ({idx.size} galaxies, {args.redshift} redshift)")
        started = time.perf_counter()
        result = invert_glo(
            generator,
            data["X_flux"][idx],
            data["X_mask"][idx].astype(np.float32),
            redshift,
            preprocessing,
            cfg,
            device,
        )
        elapsed = time.perf_counter() - started
        if args.redshift == "predicted":
            # The rest-frame SED does not encode the redshift, report the estimate actually used.
            column = names.index("redshift")
            result["params"][:, column] = mlp_predictions[split][:, column]
        path = args.workdir / RECONSTRUCTIONS / f"glo_{split}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path, **result, redshift=redshift, seconds_per_galaxy=elapsed / idx.size
        )
        print(f"  saved {path} ({elapsed / idx.size * 1e3:.2f} ms per galaxy)")


if __name__ == "__main__":
    main()
