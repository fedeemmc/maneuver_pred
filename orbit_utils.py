import numpy as np

MU = 398600.4418  # km^3/s^2


def orbital_elements(r, v):
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)

    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)

    e_vec = (np.cross(v, h) / MU) - r / r_norm
    e = np.linalg.norm(e_vec)

    energy = v_norm**2 / 2 - MU / r_norm
    a = -MU / (2 * energy)

    return a, e