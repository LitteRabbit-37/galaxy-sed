"""Step 3: train the GLO generator on the training SEDs and parameters.

    uv run python scripts/03_train_glo.py
"""

from galaxy_sed.data import load_config, load_prepared
from galaxy_sed.io import GLO, PREPARED, common_arguments, default_device, save_model, write_json
from galaxy_sed.training import train_glo


def main() -> None:
    args = common_arguments(__doc__.splitlines()[0]).parse_args()
    cfg = load_config(args.config)["glo"]
    data = load_prepared(args.workdir / PREPARED)
    device = default_device()
    print(f"Device: {device}")

    model, history = train_glo(data, cfg, device)
    init_kwargs = {"latent_dim": cfg["latent_dim"], "sed_len": model.sed_len,
                   "n_params": model.params_head[-1].out_features}
    save_model(args.workdir / GLO, model, init_kwargs,
               {"param_names": [str(name) for name in data["param_names"]], "config": cfg})
    write_json(args.workdir / "glo_history.json", history)
    print(f"Saved {args.workdir / GLO}")


if __name__ == "__main__":
    main()
