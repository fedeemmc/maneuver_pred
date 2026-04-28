from datetime import datetime, timedelta
import numpy as np

from sgp4.api import Satrec

TLE_LINE1 = "1 46265U 20059A   26111.45612292 +.00000000 +00000+0 +14128-4 6 10875"
TLE_LINE2 = "2 46265  97.8889 298.1184 0001527  91.2546 256.6925 14.82141163305070"
MU = 398600.4418  # km^3/s^2

def datetime_to_jd(dt):
    ref = datetime(2000, 1, 1, 12, 0, 0)
    delta = dt - ref
    jd = 2451545.0 + delta.total_seconds() / 86400.0
    jd_int = int(jd)
    fr = jd - jd_int
    return jd_int, fr

def jd_to_datetime(jd, fr):
    ref = datetime(2000, 1, 1, 12, 0, 0)
    days = (jd - 2451545.0) + fr
    return ref + timedelta(days=days)


def calc_e_vec(r_vec, v_vec):
    r_norm = np.linalg.norm(r_vec)
    return ((np.dot(v_vec, v_vec) * r_vec -
             np.dot(r_vec, v_vec) * v_vec) / MU) - (r_vec / r_norm)

def get_state(sat, dt):
    jd, fr = datetime_to_jd(dt)
    error, r, v = sat.sgp4(jd, fr)
    if error != 0:
        return None, None
    return np.array(r), np.array(v)

def compute_reference_e_vector(sat, t0):
    e_vecs = []
    for dt in np.arange(0, 3*86400, 30):  # 3 días
        t = t0 - timedelta(seconds=float(dt))
        r, v = get_state(sat, t)
        e_vecs.append(calc_e_vec(r, v))

    e_ref = np.mean(e_vecs, axis=0)
    return e_ref 


sat = Satrec.twoline2rv(TLE_LINE1, TLE_LINE2)
t0 = jd_to_datetime(sat.jdsatepoch, sat.jdsatepochF)
e_ref = compute_reference_e_vector(sat, t0)
e_ref_mag = np.linalg.norm(e_ref)
print(e_ref)
print(e_ref_mag)