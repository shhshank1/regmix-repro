"""Transfer-matrix reflectance of a SiN/SiO2/SiN/SiO2 stack on silicon (normal incidence).

Refractive-index models are approximate, textbook-style fits (not DACON's proprietary
ones): SiO2 uses the Malitson Sellmeier fit, Si3N4 a simple Cauchy model with k = 0,
and silicon a table of approximate (n, k) values at 50 nm steps, linearly interpolated.
Convention: complex index N = n - i*k (Born & Wolf / Macleod).
"""
import numpy as np

WL_MIN, WL_MAX, N_WL = 400.0, 800.0, 226
WAVELENGTHS = np.linspace(WL_MIN, WL_MAX, N_WL)  # nm

_SI_TABLE = np.array([  # wavelength nm, n, k  (approximate crystalline-Si values)
    [400, 5.57, 0.387], [450, 4.67, 0.144], [500, 4.30, 0.073],
    [550, 4.08, 0.047], [600, 3.94, 0.030], [650, 3.84, 0.019],
    [700, 3.78, 0.013], [750, 3.73, 0.0095], [800, 3.69, 0.0065],
])


def n_sio2(wl):
    """Malitson Sellmeier fit for fused silica; wl in nm."""
    l2 = (np.asarray(wl) / 1000.0) ** 2
    n2 = 1 + 0.6961663 * l2 / (l2 - 0.0684043**2) \
           + 0.4079426 * l2 / (l2 - 0.1162414**2) \
           + 0.8974794 * l2 / (l2 - 9.896161**2)
    return np.sqrt(n2)


def n_sin(wl):
    """Cauchy model for Si3N4 (lossless); wl in nm. About 1.95 at 550 nm."""
    return 1.90 + 1.5e4 / np.asarray(wl) ** 2


def n_si(wl):
    """Complex index N = n - i*k of silicon, interpolated from the table."""
    n = np.interp(wl, _SI_TABLE[:, 0], _SI_TABLE[:, 1])
    k = np.interp(wl, _SI_TABLE[:, 0], _SI_TABLE[:, 2])
    return n - 1j * k


def reflectance(thickness, wl=WAVELENGTHS):
    """Reflectance for a batch of stacks.

    thickness: array (N, 4) in nm, order layer_1 (top, SiN) .. layer_4 (SiO2).
    Returns array (N, len(wl)) with values in [0, 1].
    """
    d = np.atleast_2d(np.asarray(thickness, dtype=float))
    wl = np.asarray(wl, dtype=float)
    layer_n = [n_sin(wl), n_sio2(wl), n_sin(wl), n_sio2(wl)]
    eta0 = 1.0  # air
    eta_s = n_si(wl)

    N = d.shape[0]
    # Characteristic matrix M = M1 @ M2 @ M3 @ M4, entries shaped (N, L)
    m11 = np.ones((N, wl.size), dtype=complex)
    m12 = np.zeros_like(m11)
    m21 = np.zeros_like(m11)
    m22 = np.ones_like(m11)
    for j, nj in enumerate(layer_n):
        delta = 2 * np.pi * nj[None, :] * d[:, j:j + 1] / wl[None, :]
        c, s = np.cos(delta), np.sin(delta)
        a11, a12 = c, 1j * s / nj[None, :]
        a21, a22 = 1j * nj[None, :] * s, c
        m11, m12, m21, m22 = (m11 * a11 + m12 * a21, m11 * a12 + m12 * a22,
                              m21 * a11 + m22 * a21, m21 * a12 + m22 * a22)
    B = m11 + m12 * eta_s[None, :]
    C = m21 + m22 * eta_s[None, :]
    r = (eta0 * B - C) / (eta0 * B + C)
    return np.clip(np.abs(r) ** 2, 0.0, 1.0)
