"""Tests for regime model fitting utilities.

These tests validate cluster/state outputs, model-selection behavior, and
transition matrix probability constraints.
"""

from __future__ import annotations

import numpy as np


class TestFitGMM:
    """Tests for GMM fitting and BIC model selection."""

    def test_output_shape_and_labels(self) -> None:
        """Ensure labels and BIC outputs have expected dimensions."""
        from volatility_regimes.regimes.latent_models import fit_gmm

        rng = np.random.default_rng(42)
        cluster_1 = rng.normal(loc=0.0, scale=0.1, size=(100, 3))
        cluster_2 = rng.normal(loc=3.0, scale=0.1, size=(100, 3))
        cluster_3 = rng.normal(loc=6.0, scale=0.1, size=(100, 3))
        features = np.vstack([cluster_1, cluster_2, cluster_3])

        labels, bic_scores, best_k, _ = fit_gmm(features, min_k=2, max_k=5)

        assert labels.shape == (300,)
        assert set(labels).issubset({0, 1, 2, 3, 4})
        assert len(bic_scores) == 4
        assert best_k == 3
        assert len(np.unique(labels)) == best_k

    def test_bic_selects_correct_k(self) -> None:
        """Ensure BIC selects two clusters for two-cluster synthetic data."""
        from volatility_regimes.regimes.latent_models import fit_gmm

        rng = np.random.default_rng(42)
        features = np.vstack(
            [
                rng.normal(loc=0.0, scale=0.3, size=(200, 4)),
                rng.normal(loc=5.0, scale=0.3, size=(200, 4)),
            ]
        )

        _, _, best_k, _ = fit_gmm(features, min_k=2, max_k=5)
        assert best_k == 2


class TestFitHMM:
    """Tests for HMM fitting and transition matrix constraints."""

    def test_transition_matrix_rows_sum_to_one(self) -> None:
        """Check each transition matrix row sums to one."""
        from volatility_regimes.regimes.latent_models import fit_hmm

        rng = np.random.default_rng(42)
        features = np.vstack(
            [
                rng.normal(loc=0.0, scale=0.3, size=(200, 3)),
                rng.normal(loc=5.0, scale=0.3, size=(200, 3)),
            ]
        )

        labels, transition_matrix, _ = fit_hmm(
            features,
            n_states=2,
            n_iter=50,
            n_restarts=3,
        )

        assert labels.shape == (400,)
        assert transition_matrix.shape == (2, 2)
        np.testing.assert_allclose(transition_matrix.sum(axis=1), 1.0, atol=1e-6)
