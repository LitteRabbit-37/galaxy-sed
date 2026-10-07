"""End-to-end run of the whole pipeline on a tiny random sample, on CPU."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from galaxy_sed.data import (
    destandardise_params,
    load_config,
    photometry_inputs,
    prepare_dataset,
    preprocessing_from_prepared,
)
from galaxy_sed.inference import invert_glo, predict_batched, predict_parameter_heads, predict_redshift, sample_cwgan
from galaxy_sed.models import parameter_specs
from galaxy_sed.training import train_cwgan, train_direct_mlp, train_glo, train_parameter_heads

CONFIG = Path(__file__).resolve().parents[1] / "config.json"


def tiny_raw_file(path: Path, n: int = 64) -> None:
    """Smooth random SEDs on the 223-point grid, with a few parameters."""
    rng = np.random.default_rng(0)
    wave = np.geomspace(0.0303, 12340.0, 223)[::-1]
    log_wave = np.log(wave)[None, :]
    flux = np.exp(rng.normal(size=(n, 1)) + np.sin(log_wave * rng.uniform(0.5, 1.5, size=(n, 1))))
    np.savez(
        path,
        wave=wave,
        flux=flux,
        redshift=rng.uniform(0.1, 3.0, n),
        metallicity=rng.choice([0.001, 0.008, 0.02], n),
        f_host=10 ** rng.uniform(8, 12, n),
        polar_T=rng.uniform(800, 1200, n),
        sb_on=np.ones(n, dtype=int),
        polar_on=rng.integers(0, 2, n),
        AGNpars=rng.uniform(1, 10, size=(n, 4)),
    )


class PipelineTest(unittest.TestCase):
    def test_every_step_runs_and_returns_finite_arrays(self):
        device = torch.device("cpu")
        config = load_config(CONFIG)
        config["data"]["targets"] = [
            {"name": "redshift", "log10": False},
            {"name": "metallicity", "log10": False},
            {"name": "f_host", "log10": True},
            {"name": "polar_T", "log10": False},
            {"name": "AGNpar_1", "log10": False},
        ]
        quick = {"epochs": 1, "patience": 1, "batch_size": 8, "seed": 0}
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw.npz"
            tiny_raw_file(raw)
            data = prepare_dataset(raw, config)
        preprocessing = preprocessing_from_prepared(data)
        names = preprocessing["param_names"]
        test = data["idx_test"]

        mlp, _ = train_direct_mlp(data, {**config["direct_mlp"], **quick}, device, log=lambda *_: None)
        redshift = predict_redshift(predict_batched(mlp, photometry_inputs(data, test), device), names, preprocessing)

        glo, _ = train_glo(data, {**config["glo"], **quick}, device, log=lambda *_: None)
        inversion = {"seed": 0, "steps": 3, "restarts": 2, "learning_rate": 0.05, "chunk_size": 4}
        reconstruction = invert_glo(glo, data["X_flux"][test], data["X_mask"][test].astype(np.float32), redshift,
                                    preprocessing, inversion, device, log=lambda *_: None)
        self.assertEqual(reconstruction["sed"].shape, (test.size, 223))

        # Hybrid direct MLP: photometry followed by the reconstructed SED.
        def hybrid_inputs(split):
            idx = data[f"idx_{split}"]
            sed = invert_glo(glo, data["X_flux"][idx], data["X_mask"][idx].astype(np.float32), data["redshift"][idx],
                             preprocessing, inversion, device, log=lambda *_: None)["sed"]
            return np.concatenate([photometry_inputs(data, idx), sed], axis=1)

        hybrid_mlp, _ = train_direct_mlp(data, {**config["direct_mlp"], **quick}, device, log=lambda *_: None,
                                         inputs_train=hybrid_inputs("train"), inputs_val=hybrid_inputs("val"))
        self.assertEqual(hybrid_mlp.net[0].in_features, 40 + 223)
        hybrid_values = predict_batched(hybrid_mlp, hybrid_inputs("test"), device)
        self.assertEqual(hybrid_values.shape, (test.size, len(names)))

        train_physical = destandardise_params(data["Y_params"][data["idx_train"]], preprocessing)
        specs = parameter_specs(train_physical, names, np.asarray(preprocessing["param_mean"]),
                                np.asarray(preprocessing["param_std"]), np.asarray(preprocessing["param_log10"]), 12)
        self.assertEqual(specs[1]["type"], "cls")  # metallicity takes three values
        heads, _ = train_parameter_heads(
            data, photometry_inputs(data, data["idx_train"]), photometry_inputs(data, data["idx_val"]), specs,
            {**config["parameter_heads"], **quick}, device, log=lambda *_: None)
        values, sigma, _ = predict_parameter_heads(heads, photometry_inputs(data, test), device)
        self.assertEqual(values.shape, (test.size, len(names)))

        generator, _, _ = train_cwgan(data, {**config["cwgan"], **quick}, device, log=lambda *_: None)
        sampled = sample_cwgan(generator, photometry_inputs(data, test), {"seed": 0, "draws": 2, "chunk_size": 4}, device)

        for array in (redshift, reconstruction["sed"], hybrid_values, values, sigma, sampled["sed"], sampled["params"]):
            self.assertTrue(np.isfinite(array).all())


if __name__ == "__main__":
    unittest.main()
