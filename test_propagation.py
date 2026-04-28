from datetime import datetime
from tle_propagator import propagate_tle
from orbit_utils import orbital_elements
import numpy as np
import math
tle1 = "1 46265U 20059A   26008.82344931 +.00000000 +00000+0 +13466-4 6 10316"
tle2 = "2 46265  97.8846 196.9514 0001422  85.9815 173.5050 14.82174169289875"

t = datetime(2026, 4, 22, 15, 0, 0)

r, v = propagate_tle(tle1, tle2, t)

a, e = orbital_elements(r, v)

print("r =", r)
print("v =", v)
print("SMA =", a)
print("ecc =", e)