import torch
import torch.nn as nn
from torch.special import psi


def EE(x: torch.Tensor, y: torch.Tensor, p: int, k: int, Thei):
    """
    Calculate embedded entropy from https://royalsocietypublishing.org/doi/10.1098/rsif.2021.0766
    Use the kNN method to estimate the MI(X, YpNN), where X = x_*, YpNN = (dy*(p+1)+1) NN around [y_t, y_*], *means t-1, t-2, ... t-p.
    kNN formula MI(X, YpNN) = psi(k) - <psi(nX+1) + psi(nYpNN+1) - psi(nN+1)>.
    For details of the formula, see https://journals.aps.org/pre/abstract/10.1103/PhysRevE.69.066138

    Parameters
    ----------
    x : (Dimx, T) torch.Tensor
        The time series X
    y : (Dimy, T) torch.Tensor
        The time series Y
    p : int
        The order of model to estimate causality
    k : int
        The k-th nearest number to use in calculating entropy, at least 2.
    Thei : int
        Half the length of Theiler correction window. [-Thei, Thei] around point i. Thei>=p.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    (dy, T) = y.shape
    dx = x.shape[0]

    # Calculate embedding of X
    indices = torch.arange(T - p, device=device).unsqueeze(1) + torch.arange(
        p, device=device
    ).flip(0).unsqueeze(0)
    X = x[:, indices].permute(1, 2, 0).reshape(T - p, dx * p)

    # Calculate embedding of Y
    indices = torch.arange(T - p, device=device).unsqueeze(1) + torch.arange(
        p + 1, device=device
    ).flip(0).unsqueeze(0)
    Y = y[:, indices].permute(1, 2, 0).reshape(T - p, dy * (p + 1))

    # Calculate the Theiler correction
    mask = torch.ones(size=(T - p, T - p), dtype=torch.bool, device=device)
    for i in range(T - p):
        ids = max(0, i - Thei)
        idf = min(i + Thei + 1, T - p)
        mask[i, ids:idf] = False

    # Calculate dy*(p+1)+1 NN in embedded Y space
    distances = torch.cdist(Y.unsqueeze(0), Y.unsqueeze(0)).squeeze(0)
    distances[~mask] = torch.inf
    _, pnnidx = torch.topk(distances, dy * (p + 1) + 1, dim=1, largest=False)
    neighbors = Y[pnnidx]
    YpNN = neighbors.reshape(T - p, -1)

    # Calculate mutual information using kNN method
    combined = torch.cat((X, YpNN), dim=1)
    distances_combined = torch.cdist(
        combined.unsqueeze(0), combined.unsqueeze(0), p=float("inf")
    ).squeeze(0)
    distances_combined[~mask] = torch.inf
    cheby_distance, _ = torch.topk(distances_combined, k, largest=False, dim=1)
    half_epsilon = cheby_distance[:, k - 1].unsqueeze(1)

    temp_dist = torch.cdist(X.unsqueeze(0), X.unsqueeze(0), p=float("inf")).squeeze(0)
    temp_dist[~mask] = torch.inf
    nX = torch.sum(
        temp_dist < half_epsilon,
        dim=1,
    )

    temp_dist = torch.cdist(
        YpNN.unsqueeze(0), YpNN.unsqueeze(0), p=float("inf")
    ).squeeze(0)
    temp_dist[~mask] = torch.inf
    nYpNN = torch.sum(
        temp_dist < half_epsilon,
        dim=1,
    )
    nN = torch.sum(mask, dim=1)

    # return mi
    valid = (nX != 0) & (nYpNN != 0)
    mi = torch.abs(
        psi(torch.tensor(k, dtype=torch.float, device=device))
        - torch.mean(psi(nX[valid] + 1))
        - torch.mean(psi(nYpNN[valid] + 1))
        + torch.mean(psi(nN[valid] + 1))
    )
    return mi


def cEE(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, p: int, k: int, Thei):
    """
    Calculate conditional/direct embedding entropy causality from x to y condition on z, with order p.
    From https://royalsocietypublishing.org/doi/10.1098/rsif.2021.0766
    For details of mutual information formula, see https://journals.aps.org/pre/abstract/10.1103/PhysRevE.69.066138

    Parameters
    ----------
    x : (Dimx, T) torch.Tensor
        The time series X
    y : (Dimy, T) torch.Tensor
        The time series Y
    z : (Dimz, T) torch.Tensor
        The time series Z
    p : int
        The order of model to estimate causality
    k : int
        The k-th nearest number to use in calculating entropy, at least 2.
    Thei : int
        Half the length of Theiler correction window. [-Thei, Thei] around point i. Thei>=p.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    (dy, T) = y.shape
    dx = x.shape[0]
    dz = z.shape[0]

    # Calculate embedding of X
    indices = torch.arange(T - p, device=device).unsqueeze(1) + torch.arange(
        p, device=device
    ).flip(0).unsqueeze(0)
    X = x[:, indices].permute(1, 2, 0).reshape(T - p, dx * p)

    # Calculate embedding of Y
    indices = torch.arange(T - p, device=device).unsqueeze(1) + torch.arange(
        p + 1, device=device
    ).flip(0).unsqueeze(0)
    Y = y[:, indices].permute(1, 2, 0).reshape(T - p, dy * (p + 1))
    # Calculate embedding of Z
    Z = z[:, indices].permute(1, 2, 0).reshape(T - p, dy * (p + 1))

    # Calculate the Theiler correction
    mask = torch.ones(size=(T - p, T - p), dtype=torch.bool, device=device)
    for i in range(T - p):
        ids = max(0, i - Thei)
        idf = min(i + Thei + 1, T - p)
        mask[i, ids:idf] = False

    # Calculate dy*(p+1)+1 NN in embedded Y,Z space
    distances = torch.cdist(Y.unsqueeze(0), Y.unsqueeze(0)).squeeze(0)
    distances[~mask] = torch.inf
    _, pnnidx = torch.topk(distances, dy * (p + 1) + 1, dim=1, largest=False)
    neighbors = Y[pnnidx]
    YpNN = neighbors.reshape(T - p, -1)

    distances = torch.cdist(Z.unsqueeze(0), Z.unsqueeze(0)).squeeze(0)
    distances[~mask] = torch.inf
    _, pnnidx = torch.topk(distances, dz * (p + 1) + 1, dim=1, largest=False)
    neighbors = Z[pnnidx]
    ZpNN = neighbors.reshape(T - p, -1)

    # Calculate mutual information using kNN method
    combined = torch.cat((X, YpNN, ZpNN), dim=1)
    distances_combined = torch.cdist(
        combined.unsqueeze(0), combined.unsqueeze(0), p=float("inf")
    ).squeeze(0)
    distances_combined[~mask] = torch.inf
    cheby_distance, _ = torch.topk(distances_combined, k, largest=False, dim=1)
    half_epsilon = cheby_distance[:, k - 1].unsqueeze(1)

    temp_dist = torch.cdist(
        torch.cat((X, ZpNN), dim=1).unsqueeze(0),
        torch.cat((X, ZpNN), dim=1).unsqueeze(0),
        p=float("inf"),
    ).squeeze(0)
    temp_dist[~mask] = torch.inf
    nXZ = torch.sum(temp_dist < half_epsilon, dim=1)

    temp_dist = torch.cdist(
        torch.cat((YpNN, ZpNN), dim=1).unsqueeze(0),
        torch.cat((YpNN, ZpNN), dim=1).unsqueeze(0),
        p=float("inf"),
    ).squeeze(0)
    temp_dist[~mask] = torch.inf
    nYZ = torch.sum(temp_dist < half_epsilon, dim=1)

    temp_dist = torch.cdist(ZpNN.unsqueeze(0), ZpNN.unsqueeze(0)).squeeze(0)
    temp_dist[~mask] = torch.inf
    nZ = torch.sum(temp_dist < half_epsilon, dim=1)

    valid = (nXZ != 0) & (nYZ != 0) & (nZ != 0)
    mi = torch.abs(
        psi(torch.tensor(k, dtype=torch.float, device=device))
        - torch.mean(psi(nXZ[valid] + 1))
        - torch.mean(psi(nYZ[valid] + 1))
        + torch.mean(psi(nZ[valid] + 1))
    )
    return mi
