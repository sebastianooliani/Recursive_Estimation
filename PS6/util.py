import numpy as np
import scipy
from scipy.signal import place_poles


def obsv(A, H):
    """Compute the observability matrix of a linear system

    Args:
        A (np.ndarray): system matrix A
        H (np.ndarray): system matrix H

    Returns:
        (np.ndarray): observation matrix
    """
    n = A.shape[0]
    obsv = np.vstack([H] + [H @ np.linalg.matrix_power(A, i) for i in range(1, n)])
    return obsv


def ctrb(A, B):
    """Compute the controllability matrix of a linear system

    Args:
        A (np.ndarray): system matrix A
        B (np.ndarray): system matrix B

    Returns:
        (np.ndarray): controllability matrix
    """
    n = A.shape[0]
    ctrb = np.vstack([B] + [np.linalg.matrix_power(A, i) @ B for i in range(1, n)])
    return ctrb


def place(A, B, p):
    """Closed-loop pole assignment using state feedback

    Args:
        A (np.ndarray): system matrix A
        B (np.ndarray): system matrix B
        p (np.ndarray): desired poles

    Returns:
        np.ndarray: state-feedback matrix K
    """

    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")

    if A.shape[0] != B.shape[0]:
        raise ValueError("The number of rows of A must equal the number of rows in B")

    result = place_poles(A, B, p, method="YT")
    K = result.gain_matrix

    return K


def obsvf(A, B, C):
    # Observability staircase form
    aa, bb, cc, t = ctrbf(A.T, C.T, B.T)
    Abar = aa.T
    Bbar = cc.T
    Cbar = bb.T
    return Abar, Bbar, Cbar, t


def ctrbf(A, B, C):
    # This function implements the Staircase Algorithm of Rosenbrock, 1968.
    ra = A.shape[0]
    cb = B.shape[1]

    ptjn1 = np.eye(ra)
    ajn1 = A
    bjn1 = B
    rojn1 = cb
    deltajn1 = 0
    sigmajn1 = ra

    for jj in range(ra):
        uj, sj, vj = np.linalg.svd(bjn1)
        rsj = uj.shape[0]
        p = np.rot90(np.eye(rsj), 1)
        uj = uj @ p
        bb = uj.T @ bjn1
        roj = np.linalg.matrix_rank(bb)
        rbb = bb.shape[0]
        sigmaj = rbb - roj
        sigmajn1 = sigmaj
        if roj == 0:
            break
        if sigmaj == 0:
            break
        abxy = uj.T @ ajn1 @ uj
        aj = abxy[0:sigmaj, 0:sigmaj]
        bj = abxy[0:sigmaj, sigmaj : sigmaj + roj]
        ajn1 = aj
        bjn1 = bj
        ruj = uj.shape[0]
        cuj = uj.shape[1]
        ptj = ptjn1 @ np.vstack(
            (
                np.hstack((uj, np.zeros((ruj, deltajn1)))),
                np.hstack((np.zeros((deltajn1, cuj)), np.eye(deltajn1))),
            )
        )
        ptjn1 = ptj
        deltaj = deltajn1 + roj
        deltajn1 = deltaj

    t = ptjn1.T
    Abar = t @ A @ t.T
    Bbar = t @ B
    Cbar = C @ t.T

    return Abar, Bbar, Cbar, t


def dlqr(A, B, Q, R):
    """Linear-quadratic regulator design for discrete-time systems

    Args:
        A (np.ndarray): system matrix A
        B (np.ndarray): system matrix B
        Q (np.ndarray): prcoess noise covariance matrix Q
        R (np.ndarray): measurement noise covariance matrix R

    Returns:
        np.ndarray: optimal gain matrix
    """
    X = scipy.linalg.solve_discrete_are(A, B, Q, R)
    return scipy.linalg.solve(B.T @ X @ B + R, B.T @ X @ A)
