"""Step 7: SEDs and parameters from the conditional WGAN-GP.

Averages ``draws`` generated samples per galaxy. Writes ``cwgan_<split>.npz``.

    uv run python scripts/07_predict_cwgan.py --split test
"""

import time

import numpy as np

from galaxy_sed.data import load_config, load_prepared, photometry_inputs
from galaxy_sed.inference import sample_cwgan
from galaxy_sed.io import CWGAN, PREPARED, common_arguments, default_device, load_model


def main() -> None:
    parser = common_arguments(__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["val", "test"], default="test")
    args = parser.parse_args()
    cfg = load_config(args.config)["cwgan"]
    data = load_prepared(args.workdir / PREPARED)
    device = default_device()
    generator, _ = load_model(args.workdir / CWGAN, device)

    conditions = photometry_inputs(data, data[f"idx_{args.split}"])
    started = time.perf_counter()
    result = sample_cwgan(generator, conditions, cfg, device)
    elapsed = time.perf_counter() - started
    path = args.workdir / f"cwgan_{args.split}.npz"
    np.savez(path, **result, seconds_per_galaxy=elapsed / len(conditions))
    print(f"Saved {path} ({elapsed / len(conditions) * 1e3:.3f} ms per galaxy, {cfg['draws']} draws)")


if __name__ == "__main__":
    main()
