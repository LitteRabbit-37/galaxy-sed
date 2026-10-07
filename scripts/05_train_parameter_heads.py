"""Step 5: train one network per physical parameter.

Two inputs are available:

* ``photometry``: the 20 fluxes and their masks (40 numbers)
* ``hybrid``: the same, plus the GLO-reconstructed SED (223 numbers). The
  heads are trained on reconstructed SEDs (step 4), not on the true ones, so
  they learn how far an imperfect reconstruction can be trusted.

With ``uncertainty`` enabled in the configuration, every parameter also gets a
per-galaxy error bar.

    uv run python scripts/05_train_parameter_heads.py --input photometry
    uv run python scripts/05_train_parameter_heads.py --input hybrid
"""

import numpy as np

from galaxy_sed.data import (
    destandardise_params,
    load_config,
    load_prepared,
    load_preprocessing,
)
from galaxy_sed.inference import predict_parameter_heads
from galaxy_sed.io import (
    PREPARED,
    PREPROCESSING,
    common_arguments,
    default_device,
    model_inputs,
    parameter_heads_file,
    save_model,
    write_json,
)
from galaxy_sed.models import parameter_specs
from galaxy_sed.training import train_parameter_heads


def main() -> None:
    parser = common_arguments(__doc__.splitlines()[0])
    parser.add_argument("--input", choices=["photometry", "hybrid"], required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)["parameter_heads"]
    data = load_prepared(args.workdir / PREPARED)
    preprocessing = load_preprocessing(args.workdir / PREPROCESSING)
    device = default_device()
    print(f"Device: {device}, input: {args.input}")

    train_physical = destandardise_params(
        data["Y_params"][data["idx_train"]], preprocessing
    )
    specs = parameter_specs(
        train_physical,
        preprocessing["param_names"],
        np.asarray(preprocessing["param_mean"]),
        np.asarray(preprocessing["param_std"]),
        np.asarray(preprocessing["param_log10"]),
        cfg["max_classes"],
    )
    print(
        "Classification heads: "
        + ", ".join(s["name"] for s in specs if s["type"] == "cls")
    )

    inputs = {
        split: model_inputs(data, args.workdir, args.input, split)
        for split in ("train", "val", "test")
    }
    model, history = train_parameter_heads(
        data, inputs["train"], inputs["val"], specs, cfg, device
    )
    init_kwargs = {
        "n_in": inputs["train"].shape[1],
        "specs": specs,
        "uncertainty": cfg["uncertainty"],
        "dropout": cfg["dropout"],
    }
    path = args.workdir / parameter_heads_file(args.input)
    save_model(path, model, init_kwargs, {"input": args.input, "config": cfg})
    write_json(args.workdir / f"parameter_heads_{args.input}_history.json", history)

    predictions = {}
    for split in ("val", "test"):
        values, sigma, _ = predict_parameter_heads(model, inputs[split], device)
        predictions[f"{split}_values"] = values
        if sigma is not None:
            predictions[f"{split}_sigma"] = sigma
    np.savez(
        args.workdir / f"parameter_heads_{args.input}_predictions.npz", **predictions
    )
    print(f"Saved {path} and predictions for val and test")


if __name__ == "__main__":
    main()
