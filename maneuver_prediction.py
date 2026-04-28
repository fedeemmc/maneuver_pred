#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SAOCOM-1B - OPTIMIZACIÓN DE MANIOBRA (VERSIÓN FDS CONSISTENTE)

- Usa SGP4 (TLE)
- Convierte a elementos medios mediante promedio orbital (J2 filtering)
- Aplica Δv tangencial correctamente
- Optimiza minimizando Δe y error respecto a target
"""

import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import numpy as np

from sgp4.api import Satrec

# ========================= CONFIGURACIÓN =========================

CURRENT_DATE = datetime(2026, 4, 22)

MSMA_CURRENT = 6997.59150  # km
TARGET_MSMA = 6997.74       # km
E_TARGET = 0.0012           # objetivo (MEAN, como reporte)

EXIT_WINDOW = datetime(2026, 4, 24)

THRUST_N = 14.0
MASA_KG = 2809.961098
EFICIENCIA = 0.92

MU = 398600.4418  # km^3/s^2

XML_FILE = "S1B_OPER_ODF_EVENTS_CODS_20260422T114403_EVENTS_O_1.XML"

# ==================== TLE ====================

TLE_LINE1 = "1 46265U 20059A   26112.46510440 +.00000000 +00000+0 +13507-4 6 10914"
TLE_LINE2 = "2 46265  97.8897 299.1125 0001450  93.7773 234.6140 14.82188142305229"

# ========================= UTILIDADES =========================

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

def get_state(sat, dt):
    jd, fr = datetime_to_jd(dt)
    error, r, v = sat.sgp4(jd, fr)
    if error != 0:
        return None, None
    return np.array(r), np.array(v)

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

def calc_e_vec(r_vec, v_vec):
    r_norm = np.linalg.norm(r_vec)
    return ((np.dot(v_vec, v_vec) * r_vec -
             np.dot(r_vec, v_vec) * v_vec) / MU) - (r_vec / r_norm)


def tangential_unit(r_vec, v_vec):
    r_norm = np.linalg.norm(r_vec)
    v_tang = v_vec - np.dot(v_vec, r_vec) / r_norm**2 * r_vec
    return v_tang / np.linalg.norm(v_tang)


# ========================= MEDIA ORBITAL =========================

def calcular_e_media(sat, t0, window_s=86400, step_s=10):
    e_vecs = []

    for dt in np.arange(0, window_s, step_s):
        t = t0 + timedelta(seconds=float(dt))
        r, v = get_state(sat, t)
        if r is None:
            continue

        e_vecs.append(calc_e_vec(r, v))

    return np.linalg.norm(np.mean(e_vecs, axis=0))


def calcular_e_media_post_burn(sat, t0, delta_v_ms,
                              window_s=17500, step_s=10):

    jd0, fr0 = datetime_to_jd(t0)
    err, r0, v0 = sat.sgp4(jd0, fr0)
    if err != 0:
        return None

    r0 = np.array(r0)
    v0 = np.array(v0)

    v_hat = tangential_unit(r0, v0)
    dv_vec = (delta_v_ms / 1000.0) * v_hat

    e_pre_list = []
    e_post_list = []

    # Get excentricity from t_burn - 24h
    for dt in np.arange(0, window_s, step_s):

        t = t0 - timedelta(seconds=float(dt))
        r, v = get_state(sat, t)
        if r is None:
            continue

        # PRE
        e_pre_list.append(calc_e_vec(r, v))

    # Get excentricity from t_burn + 24h
    for dt in np.arange(0, window_s, step_s):

        t = t0 + timedelta(seconds=float(dt))
        r, v = get_state(sat, t)
        if r is None:
            continue

        # POST (primer orden)
        if dt == 0:
            v = v + dv_vec

        e_post_list.append(calc_e_vec(r, v))

    e_pre = np.mean(e_pre_list, axis=0)
    e_post = np.mean(e_post_list, axis=0)

    return e_pre, e_post

def compute_reference_e_vector(sat, t0):
    e_vecs = []
    for dt in np.arange(0, 3*86400, 30):  # 3 días
        t = t0 - timedelta(seconds=float(dt))
        r, v = get_state(sat, t)
        e_vecs.append(calc_e_vec(r, v))

    e_ref = np.mean(e_vecs, axis=0)
    return e_ref / np.linalg.norm(e_ref)

# ========================= Δv =========================

print("\n=== SAOCOM MANEUVER PLANIFICATION ===\n")

delta_a = TARGET_MSMA - MSMA_CURRENT

v_orb = np.sqrt(MU / MSMA_CURRENT)
delta_v_ms = (v_orb * delta_a / (2 * MSMA_CURRENT)) * 1000

duracion = (MASA_KG * delta_v_ms) / (THRUST_N * EFICIENCIA)

print(f"Δa required   : {delta_a:.5f} km")
print(f"Δv required   : {delta_v_ms:.5f} m/s")
print(f"Burn duration : {duracion:.2f} s\n")

# ========================= SAT =========================

sat = Satrec.twoline2rv(TLE_LINE1, TLE_LINE2)
t0 = jd_to_datetime(sat.jdsatepoch, sat.jdsatepochF)
e_ref = compute_reference_e_vector(sat, t0)
e_target_vec = E_TARGET * e_ref
#e_target_vec =  np.array([-1.56526246e-04, -6.02857652e-05,  1.20560387e-03])
# ========================= EVENTOS =========================

tree = ET.parse(XML_FILE)
root = tree.getroot()

candidatos = []

for event in root.findall('.//Event'):

    event_id = event.findtext('eventID', 'N/A')
    start_str = event.findtext('start')
    stop_str = event.findtext('stop')
    el_max = float(event.findtext('elMax', 0))
    rev_num = event.findtext('revNum', 'N/A')

    try:
        start = datetime.strptime(start_str, "%Y/%m/%dT%H:%M:%S.%f")
        stop = datetime.strptime(stop_str, "%Y/%m/%dT%H:%M:%S.%f")
    except:
        continue

    if start > EXIT_WINDOW:
        continue

    # Filtro de estaciones no deseadas 
    if event_id in ['IML', 'HTN', 'LIM', 'IVK', 'KRU', 'SAA', 'TSS', 'IMT']: 
        continue

    dur_min = (stop - start).total_seconds() / 60
    if dur_min < 10:
        continue

    if el_max < 10:
        continue

    best_local = None

    # ================= SAMPLEO FINO =================
    for dt_sec in np.arange(0, (stop - start).total_seconds(), 60):

        t = start + timedelta(seconds=float(dt_sec))

        res = calcular_e_media_post_burn(sat, t, delta_v_ms)
        if res is None:
            continue

        e_pre_vec, e_post_vec = res
        e_pre = np.linalg.norm(e_pre_vec)
        e_post = np.linalg.norm(e_post_vec)

        delta_e = np.linalg.norm(e_post_vec - e_pre_vec)
        error_target = np.linalg.norm(e_post_vec - e_target_vec)

        windows_margin = (EXIT_WINDOW - t).total_seconds()

        # ================= FUNCIÓN OBJETIVO =================
        score = (
            #-1e-2 * windows_margin
            #-1e6 * delta_e
            -1e8 * error_target
            #+ dur_min * 2
            #+ el_max * 1
        )

        if best_local is None or score > best_local["score"]:
            best_local = {
                "eventID": event_id,
                "start": start,
                "stop": stop,
                "duration": dur_min,
                "revNum": rev_num,
                "time": t,
                "el_max": el_max,
                "score": score,
                "e_pre": e_pre,
                "e_post": e_post,
                "delta_e": delta_e
            }

    if best_local:
        candidatos.append(best_local)

# ========================= RESULTADOS =========================

if not candidatos:
    print("No valid contacts found.")
    exit()

candidatos.sort(key=lambda x: x["score"], reverse=True)
best = candidatos[0]

print("\n=== BEST SOLUTION ===")
print(f"Station              : {best['eventID']}")
print(f"AOS                  : {best['start'].strftime('%Y-%m-%d %H:%M:%S')}")
print(f"LOS                  : {best['stop'].strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Duration             : {best['duration']:.1f} min")
print(f"Max. Elevation       : {best['el_max']}")
print(f"Revolution           : {best['revNum']}")
print(f"Burn time            : {best['time']}")
print(f"Burn duration        : {int(np.round(duracion))} s")
print(f"Elevation            : {best['el_max']:.1f}°")

print("\n--- Excentricity (MEAN-like) ---")
print(f"e initial     : {best['e_pre']:.8f}")
print(f"e final       : {best['e_post']:.8f}")
print(f"Δe            : {best['delta_e']:.2e}")

print(f" =============================================================================================================================")
print(f"|                                                   TOP 100 CONTACTS                                                          |")
print(f"|=============================================================================================================================|")
print(f"| STATION |           AOS - LOS            |MAX. ELEVATION| DURATION |BURN TIME |BURN DURATION|     e      |    Δe    | Score |")
print(f"|-----------------------------------------------------------------------------------------------------------------------------|")
for i, c in enumerate(candidatos[:100], 1):
    print(f"| {i:2d}. {c['eventID']} | {(c['start']).strftime('%Y-%m-%d %H:%M:%S')} - {c['stop'].strftime('%H:%M:%S')} |     {c['el_max']}°    | {c['duration']} min | {c['time'].strftime('%H:%M:%S')} |     {int(np.round(duracion))} s     | {c['e_post']:.4e} | {c['delta_e']:.2e} | {c['score']:.2f} |")