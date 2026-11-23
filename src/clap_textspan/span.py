"""Signed TextSpan (Gandelsman et al., ICLR 2024), raw residual-STD selection.

Uncentered support SVD, centered fitting targets, twice reorthogonalized directions.
Gaussian magnitudes are deliberately preserved to reproduce the paper run.
"""

import numpy as np


def support(values, rank=80):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite [fit clips, dimension] contributions")
    _, singular, vt = np.linalg.svd(values, full_matrices=False)
    numerical = int((singular > singular[0] * 1e-7).sum()) if singular[0] else 0
    return vt[: min(rank, numerical)].T.copy()


def fit(values, candidates, k=32, subspace=None):
    values = np.asarray(values, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64)
    if (
        candidates.ndim != 2
        or values.shape[1] != candidates.shape[1]
        or not np.isfinite(candidates).all()
    ):
        raise ValueError("Invalid dictionary embeddings")
    subspace = support(values) if subspace is None else subspace
    mean = values.mean(0)
    centered = values - mean
    residual = centered @ subspace
    covariance = np.cov(residual, rowvar=False, ddof=1).reshape(subspace.shape[1], -1)
    original = np.linalg.norm(candidates, axis=1)
    projected = candidates @ subspace
    used = np.zeros(len(candidates), dtype=bool)
    directions = []
    indices = []
    explained = []
    energy = np.sum(centered**2)
    for step in range(min(k, subspace.shape[1])):
        vectors = projected.copy()
        if directions:
            prior = np.column_stack(directions)
            for _ in range(2):
                vectors -= (vectors @ prior) @ prior.T
        norms = np.linalg.norm(vectors, axis=1)
        valid = (~used) & (original > 0) & (norms > 1e-10 * original)
        scores = np.sqrt(
            np.maximum(np.einsum("ij,ij->i", vectors @ covariance, vectors), 0)
        )
        winner = int(np.argmax(np.where(valid, scores, -np.inf)))
        if (
            not valid[winner]
            or scores[winner]
            <= 1e-10
            * np.linalg.norm(centered)
            / np.sqrt(len(values) - 1)
            * original[winner]
        ):
            break
        q = vectors[winner] / norms[winner]
        directions.append(q)
        indices.append(winner)
        used[winner] = True
        residual -= np.outer(residual @ q, q)
        basis = subspace @ np.column_stack(directions)
        explained.append(
            float(np.sum((centered @ basis) ** 2) / energy) if energy else 0.0
        )
    basis = (
        subspace @ np.column_stack(directions)
        if directions
        else np.empty((values.shape[1], 0))
    )
    np.testing.assert_allclose(
        basis.T @ basis, np.eye(len(indices)), atol=1e-10, rtol=0
    )
    return dict(
        mean=mean,
        basis=basis,
        indices=np.array(indices, dtype=int),
        explained=np.array(explained),
    )


def projection_scores(centered, basis, texts):
    coefficients = np.asarray(centered, dtype=np.float64) @ basis
    increments = coefficients.T[:, :, None] * (basis.T @ texts.T)[:, None, :]
    return np.concatenate(
        [np.zeros((1, len(centered), len(texts))), np.cumsum(increments, axis=0)]
    )
