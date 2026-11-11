"""Numerical definitions of metrics: cossim, SNR, PCA, Simplex projection, NC metrics."""
import torch
import torch.nn.functional as F


def cosine_similarity(x):
    """Off-diagonal pairwise cosine per image without a quadratic token matrix."""
    n = x.shape[1]
    if n < 2:
        raise ValueError('Cosine similarity requires at least two tokens.')
    unit = F.normalize(x, p=2, dim=-1)
    # Subtract actual squared norms, rather than n, to also handle zero tokens.
    return (unit.sum(1).square().sum(-1) - unit.square().sum((1, 2))) / (n * (n - 1))


def signal_to_noise(x):
    mean = x.mean(dim=1)
    norm = mean.norm(p=2, dim=1)
    std = ((x - mean[:, None]).norm(p=2, dim=-1).square().mean(dim=1)).sqrt()
    return norm / (std + 1e-9)


def pca_projection(x):
    """Global token centering and the top two right singular vectors."""
    flat = x.reshape(-1, x.shape[-1])
    centered = flat - flat.mean(dim=0)
    # Reduced SVD preserves the projection and avoids an enormous square U.
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    return (centered @ vh[:2].T).reshape(*x.shape[:2], 2)


def simplex_projection(x, weights):
    if weights.shape[0] != 3:
        raise ValueError('The paper simplex projection uses exactly three classes.')
    mean = weights.mean(dim=0, keepdim=True)
    u, s, vh = torch.linalg.svd(weights - mean, full_matrices=False)
    keep = s > 1e-5
    q = u[:, keep] @ vh[keep]
    a = 2**0.5 * weights.new_tensor([[.5, -.5, 0], [0, 0, (3/4)**.5]])
    a = a @ (torch.eye(3, device=weights.device) - weights.new_ones(3, 3) / 3)
    flat = x.reshape(-1, x.shape[-1])
    return ((a @ q @ (flat - mean).T).T).reshape(*x.shape[:2], 2)


def nc_geometry(class_means, weights):
    """Historical NC2/NC3: center BOTH class means and classifier weights."""
    m = class_means - class_means.mean(0, keepdim=True)
    w = weights - weights.mean(0, keepdim=True)
    c = len(m)
    if c < 2:
        raise ValueError('NC metrics require at least two represented classes.')
    mn, wn = m.norm(dim=1), w.norm(dim=1)
    if (mn == 0).any() or (wn == 0).any():
        raise ValueError('NC geometry is undefined for zero centered class/weight vectors.')
    def equiangular(v):
        gram = v @ v.T + 1 / (c - 1)
        gram.fill_diagonal_(0)
        return (gram.abs().sum() / (c * (c - 1))).item()
    return {
        'NC2_equinorm_M': (mn.std() / mn.mean()).item(),
        'NC2_equinorm_W': (wn.std() / wn.mean()).item(),
        'NC2_equiangular_M': equiangular(m / mn[:, None]),
        'NC2_equiangular_W': equiangular(w / wn[:, None]),
        'NC3_self_duality': (m / m.norm() - w / w.norm()).square().sum().item(),
    }


def ncc_predictions(tokens, class_means, chunk_size=8192):
    """Bound token-by-class working memory, retaining original squared distances."""
    result = []
    for x in tokens.split(chunk_size):
        distances = x.square().sum(1, keepdim=True) + class_means.square().sum(1)[None]
        distances = distances - 2 * (x @ class_means.T)
        result.append(distances.argmin(1))
    return torch.cat(result)
