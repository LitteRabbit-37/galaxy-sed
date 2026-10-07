"""Step 1: build the training arrays from a raw simulation file.

Computes the synthetic 20-band photometry of every galaxy, splits the sample
into training (80 %), validation (10 %) and test (10 %) sets, and standardises
inputs and targets with statistics measured on the training set only.

Writes ``prepared.npz`` and ``preprocessing.json`` to the working directory.

    uv run python scripts/01_prepare_data.py --raw data/random_sampling_galaxies_AGN1_HOST2.npz
"""

from pathlib import Path

import numpy as np

from galaxy_sed.data import load_config, prepare_dataset, preprocessing_from_prepared, save_prepared, save_preprocessing
from galaxy_sed.io import PREPARED, PREPROCESSING, common_arguments


def main() -> None:
    parser = common_arguments(__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, required=True, help="raw .npz simulation file")
    args = parser.parse_args()

    config = load_config(args.config)
    arrays = prepare_dataset(args.raw, config)
    save_prepared(args.workdir / PREPARED, arrays)
    save_preprocessing(args.workdir / PREPROCESSING, preprocessing_from_prepared(arrays))

    n = len(arrays["redshift"])
    print(f"Galaxies kept: {n} (train {arrays['idx_train'].size}, "
          f"validation {arrays['idx_val'].size}, test {arrays['idx_test'].size})")
    coverage = arrays["X_mask"].mean(axis=0)
    for name, fraction in zip(arrays["band_names"], coverage):
        print(f"  {name:<18} inside the rest-frame grid for {100 * fraction:5.1f} % of galaxies")
    print(f"Targets: {', '.join(str(name) for name in arrays['param_names'])}")
    print(f"Saved {args.workdir / PREPARED} and {args.workdir / PREPROCESSING}")
    assert np.isfinite(arrays["X_flux"]).all() and np.isfinite(arrays["Y_sed"]).all()


if __name__ == "__main__":
    main()
