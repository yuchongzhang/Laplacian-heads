"""Uncentered spectra and streaming, observation-weighted next-token ANOVA."""

import numpy as np
import torch

ALPHAS = (0.90, 0.95, 0.99, 0.999)


def uncentered_spectrum(tokens):
    """All d eigenvalues of X.T @ X in float64, without centering or rescaling.

    Use the smaller Gram matrix; fall back to direct CPU SVD if the Gram
    eigensolver loses a nonnegative tail. No ridge or eigenvalue floor is used.
    """
    x = tokens.detach().double()
    if x.ndim != 2 or min(x.shape) == 0 or not torch.isfinite(x).all():
        raise ValueError("Expected a finite nonempty (tokens, width) matrix")
    gram = x.T @ x if x.shape[1] <= x.shape[0] else x @ x.T
    values = torch.linalg.eigvalsh(gram).flip(0).cpu()
    if not torch.isfinite(values).all() or (values <= 0).any():
        values = torch.linalg.svdvals(x.cpu()).square()
    values = torch.nn.functional.pad(values, (0, x.shape[1] - len(values)))
    if not np.isclose(values.sum().item(), x.square().sum().item(), rtol=1e-10, atol=1e-12):
        raise ValueError("Spectrum does not preserve the squared Frobenius norm")
    return values.numpy()


def component_counts(eigenvalues, alphas=ALPHAS):
    """Smallest k reaching each energy fraction, independently per sequence."""
    values = np.asarray(eigenvalues, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Expected finite nonnegative spectra (sequences, width)")
    if not alphas or any(not 0 < alpha < 1 for alpha in alphas):
        raise ValueError("Energy fractions must lie strictly between zero and one")
    cumulative = values.cumsum(axis=1)
    total = cumulative[:, -1:]
    counts = np.stack([(cumulative < alpha * total).sum(axis=1) + 1 for alpha in alphas], axis=1)
    counts[total[:, 0] == 0] = 0
    return counts


def spectrum_summary(eigenvalues):
    values = np.asarray(eigenvalues)
    counts = component_counts(values)
    return {
        "num_sequences": len(values), "alphas": list(ALPHAS),
        "mean_eigenvalues": values.mean(axis=0).tolist(),
        "std_eigenvalues": values.std(axis=0, ddof=0).tolist(),
        "mean_components": counts.mean(axis=0).tolist(),
        "std_components": counts.std(axis=0, ddof=0).tolist(),
        "zero_energy_sequences": int((values.sum(axis=1) == 0).sum()),
    }


class TokenClassMoments:
    """Float64 sufficient statistics; memory O(vocabulary * width)."""

    def __init__(self, vocabulary_size, width, device="cpu"):
        self.counts = torch.zeros(vocabulary_size, dtype=torch.int64, device=device)
        self.sums = torch.zeros(vocabulary_size, width, dtype=torch.float64, device=device)
        self.squared_norms = torch.zeros(vocabulary_size, dtype=torch.float64, device=device)

    def update(self, features, labels):
        if features.shape[:-1] != labels.shape:
            raise ValueError("Each representation must have one next-token label")
        x = features.detach().reshape(-1, features.shape[-1]).to(self.sums.device, torch.float64)
        y = labels.reshape(-1).to(self.sums.device, torch.int64)
        if not torch.isfinite(x).all():
            raise ValueError("Non-finite token representations")
        self.counts += torch.bincount(y, minlength=len(self.counts))
        self.sums.index_add_(0, y, x)
        self.squared_norms.index_add_(0, y, x.square().sum(1))

    def summary(self):
        counts = self.counts.cpu().numpy()
        keep = counts > 0
        if not keep.any():
            raise ValueError("No next-token observations accumulated")
        n = counts[keep]
        sums = self.sums.cpu().numpy()[keep]
        squared_norms = self.squared_norms.cpu().numpy()[keep]
        population = int(n.sum())
        means = sums / n[:, None]
        mean = sums.sum(axis=0) / population
        residual = squared_norms - np.einsum("ij,ij->i", sums, means)
        tolerance = 1e-10 * max(float(squared_norms.max()), 1.0)
        if residual.min() < -tolerance:
            raise ValueError("Numerically invalid negative within-class variance")
        # Only remove negative roundoff from differences of positive moments.
        within = max(float(residual.sum() / population), 0.0)
        between = float(np.einsum("i,ij,ij->", n, means - mean, means - mean) / population)
        total = float(squared_norms.sum() / population - mean @ mean)
        if not np.isclose(within + between, total, rtol=1e-9, atol=1e-9):
            raise ValueError("Within-class and between-class variances do not sum to total variance")
        total = max(total, 0.0)
        return {
            "num_tokens": population, "num_classes": int(keep.sum()),
            "class_ids": np.flatnonzero(keep).tolist(), "class_counts": n.tolist(),
            "class_weighting": "observation_count", "within_class": within,
            "between_class": between, "total": total,
            "within_class_fraction": within / total if total else None,
            "between_class_fraction": between / total if total else None,
        }
