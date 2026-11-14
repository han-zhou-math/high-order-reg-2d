"""Shared spectral differentiation for manufactured boundary densities."""
from dataclasses import dataclass
import numpy as np


def spectral_derivative(values, order=1):
    values = np.asarray(values)
    n = values.shape[0]
    modes = np.fft.fftfreq(n, d=1 / n)
    shape = (n,) + (1,) * (values.ndim - 1)
    return np.fft.ifft(np.fft.fft(values, axis=0)
                       * ((1j * modes)**order).reshape(shape), axis=0).real


@dataclass(frozen=True)
class PressureBoundaryData:
    phi: np.ndarray
    psi: np.ndarray
    q_s: np.ndarray
    q_ss: np.ndarray


def pressure_densities(geometry, q, f, mu=1.0):
    """Construct phi_p and psi_p by the numerical method's spectral path."""
    n = len(q)
    t = 2 * np.pi * np.arange(n) / n
    _, speed, tau, normal, _, _ = geometry(t)
    q_s = spectral_derivative(q) / speed[:, None]
    q_ss = spectral_derivative(q_s) / speed[:, None]
    f_tau_s = spectral_derivative(np.sum(f * tau, axis=-1)) / speed
    q_s_normal_s = spectral_derivative(np.sum(q_s * normal, axis=-1)) / speed
    phi = f_tau_s - 2 * mu * q_s_normal_s
    psi = np.sum(f * normal, axis=-1) + 2 * mu * np.sum(q_s * tau, axis=-1)
    return PressureBoundaryData(phi, psi, q_s, q_ss)

