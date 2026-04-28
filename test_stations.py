from stations_reader import read_stations
from tle_propagator import propagate_tle
from orbit_utils import orbital_elements

tle1 = "1 00000U 00000A   26085.12345678  .00000023  00000-0  00000-0 0  9990"
tle2 = "2 00000  97.8906 123.4567 0012000  90.0000 270.0000 14.12345678    00"

stations = read_stations(xml_file="S1B_OPER_ODF_EVENTS_CODS_20260329T115051_EVENTS_O_1.xml")

for st in stations:

    aos_time = st["aos"]

    r, v = propagate_tle(tle1, tle2, aos_time)

    elems =  orbital_elements(r, v)
    
    print(elems)
    # simular burn