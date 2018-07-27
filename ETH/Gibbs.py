import scipy.optimize as opt
import scipy.linalg as la
import numpy as np
from .basic import *


def rho2beta(H, rho):
    """Find beta from energy of rho

    Args
    ----
        H: np.ndarray
            Hamiltonian
        rho: np.ndarray
            Density matrix
    Returns:
        beta: float
    """
    E = trace2(H, rho).real
    return energy2beta(H, E)


def energy2beta(H, E, m=10):
    """Find beta=1/kT

    Args
    ----
        H: np.ndarray
            Hamiltonian
        E: float
            Energy
    Returns:
        beta: float
    """
    if beta2energy(H, m) < E < beta2energy(H, -m):
        b = opt.newton(lambda x: beta2energy(H, x) - E,
                       0, tol=1e-6, maxiter=100).real
        return b
    else:
        return m if E < 0 else -m


def beta2rho(H, beta):
    """rho for Gibbs ensemble with given beta

    Args
    ----
        H: np.ndarray
            Hamiltonian
        beta: float
    Returns
    ----
        rho: np.ndarray
            Normalized :math:`e^{-\\beta H}`
    """
    # Rescale the Hamiltonian to give converging expm result
    H_less = H + np.eye(*H.shape) * np.sqrt(la.norm(H)) * np.sign(beta)
    R = la.expm(-beta * H_less)
    return R / np.trace(R)


def beta2entropy(H, b):
    """Entropy S for Gibbs ensemble with given beta

    Args
    ----
        H: np.ndarray
            Hamiltonian
        beta: float
    Returns
    ----
        S: float
            Entropy for :math:`e^{-\\beta H}`
    """
    R = beta2rho(H, b)
    return entropy(la.eigvalsh(R))


def beta2energy(H, beta):
    """Convert beta to energy

    Args
    ----
        H: np.ndarray
            Hamiltonian
        beta: float
    Returns
    ----
        E: np.ndarray
            Energy for :math:`e^{-\\beta H}`
    """
    return trace2(H, beta2rho(H, beta)).real


def beta2var(H, beta):
    """rho for Gibbs ensemble with given beta

    Args
    ----
        H: np.ndarray
            Hamiltonian
        beta: float
    Returns
    ----
        variance: float
            Energy variance for :math:`e^{-\\beta H}`
    """
    rho = beta2rho(H, beta)
    return (trace2(H@H, rho) - trace2(H, rho)**2).real
