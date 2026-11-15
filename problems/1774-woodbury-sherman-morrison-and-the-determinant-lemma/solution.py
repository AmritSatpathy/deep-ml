import numpy as np

def woodbury_solve(sigma_diag, X, B):
    sigma_diag = np.asarray(sigma_diag)
    X = np.asarray(X)
    B = np.asarray(B)

    # Σ^(-1) is diagonal, so use element-wise division.
    sigma_inv_B = B / sigma_diag if B.ndim == 1 else B / sigma_diag[:, None]
    sigma_inv_X = X / sigma_diag[:, None]

    # Compute the D x D system.
    D = X.shape[1]
    C = np.eye(D) + X.T @ sigma_inv_X

    # Solve C Z = X.T Σ^(-1) B.
    rhs = X.T @ sigma_inv_B
    Z = np.linalg.solve(C, rhs)

    # Apply the Woodbury identity.
    correction = sigma_inv_X @ Z
    return sigma_inv_B - correction


def sherman_morrison(A_inv, u, v):
    A_inv = np.asarray(A_inv)
    u = np.asarray(u).reshape(-1)
    v = np.asarray(v).reshape(-1)

    # Compute A^(-1)u and v^T A^(-1).
    A_inv_u = A_inv @ u
    v_A_inv = v @ A_inv

    # Scalar denominator.
    denom = 1.0 + v @ A_inv_u

    if denom == 0:
        raise ValueError("Sherman-Morrison update is singular.")

    # Sherman-Morrison formula.
    return A_inv - np.outer(A_inv_u, v_A_inv) / denom


def logdet_rank_one_update(logdet_A, A_inv, u, v):
    A_inv = np.asarray(A_inv)
    u = np.asarray(u).reshape(-1)
    v = np.asarray(v).reshape(-1)

    # Matrix determinant lemma.
    factor = 1.0 + v @ (A_inv @ u)

    if factor <= 0:
        raise ValueError("Updated determinant is not positive.")

    return float(logdet_A + np.log(factor))