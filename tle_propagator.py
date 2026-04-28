import numpy as np
from sgp4.api import Satrec
from sgp4.api import jday


def propagate_tle(tle_line1, tle_line2, dt_utc):
    """
    Propaga TLE a datetime UTC
    devuelve r [km], v [km/s] en TEME
    """

    sat = Satrec.twoline2rv(tle_line1, tle_line2)

    jd, fr = jday(
        dt_utc.year,
        dt_utc.month,
        dt_utc.day,
        dt_utc.hour,
        dt_utc.minute,
        dt_utc.second + dt_utc.microsecond * 1e-6
    )

    error_code, r, v = sat.sgp4(jd, fr)

    if error_code != 0:
        raise RuntimeError(f"SGP4 error: {error_code}")

    return np.array(r), np.array(v)