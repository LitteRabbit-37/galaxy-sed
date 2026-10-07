"""Tests for the data preparation functions."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from galaxy_sed.data import (
    destandardise_params,
    mean_std,
    split_indices,
    standardise_photometry,
    synthetic_photometry,
)
from galaxy_sed.io import RECONSTRUCTIONS, model_inputs


class SyntheticPhotometryTest(unittest.TestCase):
    def test_matches_linear_interpolation(self):
        rng = np.random.default_rng(0)
        wave = np.geomspace(0.03, 1000.0, 50)[::-1]  # descending, as in the simulation files
        flux = rng.uniform(1.0, 10.0, size=(4, wave.size))
        redshift = np.array([0.1, 1.0, 2.5, 6.0])
        bands = np.array([0.15, 1.2, 24.0, 500.0])
        photometry, mask = synthetic_photometry(wave, flux, redshift, bands)
        order = np.argsort(wave)
        for i, z in enumerate(redshift):
            expected = np.interp(bands / (1 + z), wave[order], flux[i, order], left=np.nan, right=np.nan)
            np.testing.assert_allclose(photometry[i], expected, rtol=1e-12, equal_nan=True)
            np.testing.assert_array_equal(mask[i], np.isfinite(expected).astype(np.int8))

    def test_bands_outside_the_grid_are_masked(self):
        wave = np.array([1.0, 2.0, 3.0])
        photometry, mask = synthetic_photometry(wave, np.ones((1, 3)), np.array([0.0]), np.array([0.5, 2.0, 4.0]))
        np.testing.assert_array_equal(mask[0], [0, 1, 0])
        self.assertTrue(np.isnan(photometry[0, 0]) and np.isnan(photometry[0, 2]))


class SplitAndStandardiseTest(unittest.TestCase):
    def test_split_is_a_reproducible_partition(self):
        first = split_indices(1000, 42, (0.8, 0.1, 0.1))
        second = split_indices(1000, 42, (0.8, 0.1, 0.1))
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        self.assertEqual([part.size for part in first], [800, 100, 100])
        self.assertEqual(np.unique(np.concatenate(first)).size, 1000)

    def test_masked_statistics_ignore_masked_entries(self):
        values = np.array([[1.0, np.nan], [3.0, 5.0]])
        mask = np.array([[1, 0], [1, 1]])
        mean, std = mean_std(values, mask)
        np.testing.assert_allclose(mean, [2.0, 5.0])
        np.testing.assert_allclose(std, [1.0, 1.0])  # the constant column gets a unit scale

    def test_photometry_round_trip(self):
        preprocessing = {"flux_log_mean": [0.0, 1.0], "flux_log_std": [1.0, 2.0]}
        x_flux, mask = standardise_photometry(np.array([[10.0, np.nan], [100.0, 1000.0]]), preprocessing)
        np.testing.assert_array_equal(mask, [[1, 0], [1, 1]])
        np.testing.assert_allclose(x_flux, [[1.0, 0.0], [2.0, 1.0]])

    def test_parameters_are_destandardised_with_log10(self):
        preprocessing = {"param_mean": [1.0, 0.0], "param_std": [2.0, 1.0], "param_log10": [True, False]}
        values = destandardise_params(np.array([[0.5, 3.0]]), preprocessing)
        np.testing.assert_allclose(values, [[100.0, 3.0]])


class ModelInputsTest(unittest.TestCase):
    def test_hybrid_inputs_append_the_reconstructed_sed(self):
        data = {
            "X_flux": np.arange(12, dtype=np.float32).reshape(3, 4),
            "X_mask": np.ones((3, 4), dtype=np.int8),
            "idx_val": np.array([0, 2]),
        }
        sed = np.full((2, 5), 7.0, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / RECONSTRUCTIONS).mkdir()
            np.savez(Path(tmp) / RECONSTRUCTIONS / "glo_val.npz", sed=sed)
            photometry = model_inputs(data, Path(tmp), "photometry", "val")
            hybrid = model_inputs(data, Path(tmp), "hybrid", "val")
        self.assertEqual(photometry.shape, (2, 8))  # 4 fluxes and 4 mask bits
        np.testing.assert_array_equal(hybrid[:, :8], photometry)
        np.testing.assert_array_equal(hybrid[:, 8:], sed)


if __name__ == "__main__":
    unittest.main()
