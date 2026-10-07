"""Step 6: train the conditional WGAN-GP.

Saves the generator, used for prediction, and the critic, needed only to
resume training.

    uv run python scripts/06_train_cwgan.py
"""

from galaxy_sed.data import load_config, load_prepared
from galaxy_sed.io import CWGAN, CWGAN_CRITIC, PREPARED, common_arguments, default_device, save_model, write_json
from galaxy_sed.training import train_cwgan


def main() -> None:
    args = common_arguments(__doc__.splitlines()[0]).parse_args()
    cfg = load_config(args.config)["cwgan"]
    data = load_prepared(args.workdir / PREPARED)
    device = default_device()
    print(f"Device: {device}")

    generator, critic, history = train_cwgan(data, cfg, device)
    sizes = {"sed_len": generator.sed_len, "n_params": generator.params_head[-1].out_features}
    cond_dim = generator.trunk[0].in_channels - cfg["latent_dim"]
    meta = {"param_names": [str(name) for name in data["param_names"]], "config": cfg}
    save_model(args.workdir / CWGAN, generator, {"latent_dim": cfg["latent_dim"], "cond_dim": cond_dim, **sizes}, meta)
    save_model(args.workdir / CWGAN_CRITIC, critic, {"cond_dim": cond_dim, **sizes}, meta)
    write_json(args.workdir / "cwgan_history.json", history)
    print(f"Saved {args.workdir / CWGAN}")


if __name__ == "__main__":
    main()
