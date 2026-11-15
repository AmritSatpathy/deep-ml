import numpy as np

def partitioned_inverse(M, k):
    n = M.shape[0]

    E = M[:k, :k]
    F = M[:k, k:]
    G = M[k:, :k]
    H = M[k:, k:]

    H_inv = np.linalg.inv(H)

    S = E - F @ H_inv @ G
    S_inv = np.linalg.inv(S)

    top_left = S_inv
    top_right = -S_inv @ F @ H_inv
    bottom_left = -H_inv @ G @ S_inv
    bottom_right = H_inv + H_inv @ G @ S_inv @ F @ H_inv

    return np.block([
        [top_left, top_right],
        [bottom_left, bottom_right]
    ])


def block_slogdet(M, k):
    E = M[:k, :k]
    F = M[:k, k:]
    G = M[k:, :k]
    H = M[k:, k:]

    S = E - F @ np.linalg.solve(H, G)

    sign_S, logdet_S = np.linalg.slogdet(S)
    sign_H, logdet_H = np.linalg.slogdet(H)

    sign = sign_S * sign_H
    logabsdet = logdet_S + logdet_H

    return sign, logabsdet


def bordered_inverse(A_inv, b, c):
    b = np.asarray(b).reshape(-1)
    n = len(b)

    # Schur complement
    v = A_inv @ b
    s = c - b @ v

    if abs(s) < 1e-12:
        raise ValueError("Schur complement is too close to zero.")

    # Construct inverse blocks with consistent dimensions
    top_left = A_inv + np.outer(v, v) / s
    top_right = (-v / s).reshape(n, 1)
    bottom_left = (-v / s).reshape(1, n)
    bottom_right = np.array([[1.0 / s]])

    return np.block([
        [top_left, top_right],
        [bottom_left, bottom_right]
    ])
