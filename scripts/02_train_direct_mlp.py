"""Step 2: train the direct MLP (one network for all parameters).

Two inputs are available, as for the parameter heads of step 5:

* ``photometry``: the 20 fluxes and their masks. Its redshift estimate is what
  the GLO inversion uses (step 4), so this network is trained first.
* ``hybrid``: the same, plus the GLO-reconstructed SED. It needs the
  reconstructions of step 4, so it is trained after them.

Predictions for every split are saved so the following steps never need the
true redshift.

    uv run python scripts/02_train_direct_mlp.py
    uv run python scripts/02_train_direct_mlp.py --input hybrid
"""

import numpy as np

from galaxy_sed.data import load_config, load_prepared
from galaxy_sed.inference import predict_batched
from galaxy_sed.io import (
    PREPARED,
    common_arguments,
    default_device,
    direct_mlp_file,
    direct_mlp_predictions_file,
    model_inputs,
    save_model,
    write_json,
)
from galaxy_sed.training import train_direct_mlp


def main() -> None:
    parser = common_arguments(__doc__.splitlines()[0])
    parser.add_argument("--input", choices=["photometry", "hybrid"], default="photometry")
    args = parser.parse_args()
    cfg = load_config(args.config)["direct_mlp"]
    data = load_prepared(args.workdir / PREPARED)
    device = default_device()
    print(f"Device: {device}, input: {args.input}")

    inputs = {
        split: model_inputs(data, args.workdir, args.input, split)
        for split in ("train", "val", "test")
    }
    model, history = train_direct_mlp(
        data, cfg, device, inputs_train=inputs["train"], inputs_val=inputs["val"]
    )
    init_kwargs = {
        "n_in": model.net[0].in_features,
        "n_out": model.net[-1].out_features,
        "dropout": cfg["dropout"],
    }
    path = args.workdir / direct_mlp_file(args.input)
    metadata = {
        "param_names": [str(name) for name in data["param_names"]],
        "input": args.input,
        "config": cfg,
    }
    save_model(path, model, init_kwargs, metadata)
    write_json(args.workdir / path.name.replace(".pt", "_history.json"), history)

    predictions = {
        split: predict_batched(model, split_inputs, device)
        for split, split_inputs in inputs.items()
    }
    np.savez(args.workdir / direct_mlp_predictions_file(args.input), **predictions)
    print(f"Saved {path} and predictions for train, val and test")


if __name__ == "__main__":
    main()
