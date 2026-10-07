"""Tests for model shapes, model files and metrics."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from galaxy_sed.io import load_model, save_model
from galaxy_sed.metrics import bolometric_luminosity, rissaki_chi2
from galaxy_sed.models import CWGANCritic, CWGANGenerator, DirectMLP, GLOGenerator, ParameterHeads, parameter_specs


class ModelShapeTest(unittest.TestCase):
    def test_output_shapes(self):
        x = torch.randn(5, 40)
        self.assertEqual(tuple(DirectMLP().eval()(x).shape), (5, 18))
        sed, params = GLOGenerator().eval()(torch.randn(5, 32))
        self.assertEqual((tuple(sed.shape), tuple(params.shape)), ((5, 223), (5, 18)))
        sed, params = CWGANGenerator().eval()(torch.randn(5, 32), x)
        self.assertEqual((tuple(sed.shape), tuple(params.shape)), ((5, 223), (5, 18)))
        self.assertEqual(tuple(CWGANCritic()(sed, params, x).shape), (5,))

    def test_specs_choose_classification_for_small_grids(self):
        train = np.column_stack([np.repeat([1.0, 2.0, 3.0], 10), np.linspace(0.0, 1.0, 30)])
        specs = parameter_specs(train, ["grid", "continuous"], np.zeros(2), np.ones(2), np.array([False, False]), 12)
        self.assertEqual([s["type"] for s in specs], ["cls", "reg"])
        self.assertEqual(specs[0]["levels_std"], [1.0, 2.0, 3.0])
        heads = ParameterHeads(40, specs, uncertainty=True).eval()
        outputs = heads(torch.randn(4, 40))
        self.assertEqual([tuple(o.shape) for o in outputs], [(4, 3), (4, 2)])


class ModelFileTest(unittest.TestCase):
    def test_round_trip_with_safe_loading(self):
        model = DirectMLP().eval()
        x = torch.randn(3, 40)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mlp.pt"
            save_model(path, model, {"n_in": 40, "n_out": 18, "dropout": 0.1}, {"note": "test"})
            loaded, metadata = load_model(path)
        self.assertEqual(metadata, {"note": "test"})
        torch.testing.assert_close(loaded(x), model(x))


class MetricTest(unittest.TestCase):
    def test_chi2_is_zero_for_a_perfect_reconstruction(self):
        wave = np.geomspace(1.0, 100.0, 40)
        flux = np.ones((2, 40))
        np.testing.assert_allclose(rissaki_chi2(flux, flux, wave), 0.0)

    def test_chi2_of_a_ten_percent_error(self):
        wave = np.geomspace(1.0, 100.0, 40)
        truth = np.ones((1, 40))
        # 10 % error on every sample: 0.1^2 / (0.011 + 0.01) per sample.
        np.testing.assert_allclose(rissaki_chi2(truth, 1.1 * truth, wave), [0.01 / 0.0221], rtol=1e-12)

    def test_luminosity_is_an_integral_over_frequency(self):
        c = 2.99792458e14
        wave = np.geomspace(1.0, 10.0, 2000)
        flux = np.ones((1, wave.size))  # constant S_nu: integral = nu_max - nu_min
        expected = c / 1.0 - c / 10.0
        np.testing.assert_allclose(bolometric_luminosity(flux, wave), [expected], rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
