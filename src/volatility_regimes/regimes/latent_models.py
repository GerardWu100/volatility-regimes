"""Regime modeling using Gaussian Mixture Model (GMM) and Hidden Markov Model (HMM).

This module fits two complementary unsupervised models:
- GMM (Gaussian Mixture Model): clusters daily observations independently.
- HMM (Hidden Markov Model): models latent states with temporal persistence.

Workflow
--------
1. Standardize feature matrix using z-score normalization.
2. Fit GMM across candidate state counts K and select K by minimum BIC.
3. Fit HMM with selected K and choose best restart by log-likelihood.
4. Reorder resulting labels by volatility level for interpretability.
"""

from __future__ import annotations

import logging

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def fit_gmm(
    feature_matrix: np.ndarray,
    min_k: int = 2,
    max_k: int = 6,
) -> tuple[np.ndarray, dict[int, float], int, GaussianMixture]:
    """Fit GMM for each K in range and select best K by BIC.

    Parameters
    ----------
    feature_matrix : np.ndarray
        Matrix with shape `(n_samples, n_features)`.
    min_k : int, default=2
        Minimum mixture components to evaluate.
    max_k : int, default=6
        Maximum mixture components to evaluate.

    Returns
    -------
    tuple[np.ndarray, dict[int, float], int, GaussianMixture]
        - labels : predicted cluster labels for best model.
        - bic_scores : mapping from K to BIC value.
        - best_k : K minimizing BIC.
        - best_model : fitted GaussianMixture instance for best_k.
    """
    bic_scores: dict[int, float] = {}
    candidate_models: dict[int, GaussianMixture] = {}

    for n_components in range(min_k, max_k + 1):
        candidate_model = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            n_init=5,
            random_state=42,
        )

        candidate_model.fit(feature_matrix)

        bic_value = float(candidate_model.bic(feature_matrix))
        bic_scores[n_components] = bic_value
        candidate_models[n_components] = candidate_model

        logger.info("GMM K=%s BIC=%.1f", n_components, bic_value)

    best_k = min(bic_scores, key=bic_scores.get)
    best_model = candidate_models[best_k]
    labels = best_model.predict(feature_matrix)

    logger.info("GMM selected K=%s with BIC=%.1f", best_k, bic_scores[best_k])
    return labels, bic_scores, best_k, best_model


def fit_hmm(
    feature_matrix: np.ndarray,
    n_states: int,
    n_iter: int = 200,
    n_restarts: int = 10,
) -> tuple[np.ndarray, np.ndarray, GaussianHMM]:
    """Fit Gaussian HMM with multiple random restarts.

    Parameters
    ----------
    feature_matrix : np.ndarray
        Matrix with shape `(n_samples, n_features)`.
    n_states : int
        Number of hidden states.
    n_iter : int, default=200
        Maximum expectation-maximization iterations per restart.
    n_restarts : int, default=10
        Number of random initializations.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, GaussianHMM]
        - labels : most likely state sequence from Viterbi decoding.
        - transition_matrix : matrix `P(state_t+1 | state_t)`.
        - best_model : fitted HMM with highest in-sample log-likelihood.

    Raises
    ------
    RuntimeError
        If all restart attempts fail.
    """
    best_score = -np.inf
    best_model: GaussianHMM | None = None

    for restart_index in range(n_restarts):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=restart_index,
        )

        try:
            model.fit(feature_matrix)
            score = float(model.score(feature_matrix))
        except Exception as error:  # noqa: BLE001
            logger.warning("HMM restart %s failed: %s", restart_index, error)
            continue

        if score > best_score:
            best_score = score
            best_model = model

    if best_model is None:
        raise RuntimeError("All HMM restarts failed to converge")

    labels = best_model.predict(feature_matrix)
    transition_matrix = best_model.transmat_

    logger.info("HMM fitted with %s states, log-likelihood=%.1f", n_states, best_score)
    return labels, transition_matrix, best_model


def order_regimes_by_volatility(
    labels: np.ndarray,
    standardized_features: np.ndarray,
    atm_iv_col_idx: int = 0,
) -> np.ndarray:
    """Relabel regimes from low-vol to high-vol ordering.

    Parameters
    ----------
    labels : np.ndarray
        Original model labels with shape `(n_samples,)`.
    standardized_features : np.ndarray
        Standardized feature matrix used during model fitting.
    atm_iv_col_idx : int, default=0
        Index of the ATM IV feature column in `standardized_features`.

    Returns
    -------
    np.ndarray
        Relabeled regime array where label 0 is the lowest-vol regime.
    """
    unique_labels = np.unique(labels)
    regime_mean_atm = {
        int(regime_label): float(
            standardized_features[labels == regime_label, atm_iv_col_idx].mean()
        )
        for regime_label in unique_labels
    }

    # Relabel so 0 is the lowest ATM-IV regime and higher ids rise with vol.
    low_to_high_labels = sorted(regime_mean_atm, key=regime_mean_atm.get)
    old_to_new = {
        original_label: new_label
        for new_label, original_label in enumerate(low_to_high_labels)
    }
    return np.array([old_to_new[int(label)] for label in labels], dtype=int)


def standardize_features(
    feature_matrix: np.ndarray,
) -> tuple[np.ndarray, StandardScaler]:
    """Apply z-score standardization to each feature column.

    Parameters
    ----------
    feature_matrix : np.ndarray
        Matrix with shape `(n_samples, n_features)`.

    Returns
    -------
    tuple[np.ndarray, StandardScaler]
        Standardized feature matrix and fitted scaler object.
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_matrix)
    return scaled, scaler
