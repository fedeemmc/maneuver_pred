import numpy as np
from sgp4.api import Satrec

MU = 398600.4418  # km^3/s^2

tle1 = "1 46265U 20059A   26112.46510440 +.00000000 +00000+0 +13507-4 6 10914"
tle2 = "2 46265  97.8897 299.1125 0001450  93.7773 234.6140 14.82188142305229"

def semi_major_axis_mean_precise(tle1, tle2, samples=8640):
    sat = Satrec.twoline2rv(tle1, tle2)
    MU = 398600.4418

    a_vals = []

    for k in range(samples):
        dt = k * 10  # cada minuto
        
        jd = sat.jdsatepoch
        fr = sat.jdsatepochF + dt/86400.0
        
        e, r, v = sat.sgp4(jd, fr)
        
        r = np.linalg.norm(r)
        v = np.linalg.norm(v)
        
        energy = v**2/2 - MU/r
        a = -MU/(2*energy)
        
        a_vals.append(a)

    return np.mean(a_vals)

print(semi_major_axis_mean_precise(tle1, tle2))