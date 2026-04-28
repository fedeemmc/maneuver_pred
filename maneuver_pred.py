
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAOCOM-1B — Orbit Maintenance Maneuver Optimizer
=================================================
Estrategia:
  - SGP4 (TLE) para propagación
  - Promedio orbital J2-filtered para elementos medios
  - Burn tangencial instantáneo (primer orden)
  - Función objetivo multi-criterio:
      1. Error respecto al e-vector target (frozen condition)
      2. Penalización por latitud del punto de quema (debe ser ~ecuador)
      3. Bonus por nodo correcto (Norte: ascendente / Sur: descendente)
      4. Minimizar Δe adicional introducido por la quema

Convención de hemisferio:
  - Quema NORTE → punto de quema en nodo ascendente (lat ≈ 0°, vz > 0)
  - Quema SUR   → punto de quema en nodo descendente (lat ≈ 0°, vz < 0)
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from sgp4.api import Satrec

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN — editar aquí
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # ── Misión ──────────────────────────────────────────────────────────────
    current_date:   datetime = datetime(2026, 4, 23)
    exit_window:    datetime = datetime(2026, 4, 24 )

    # ── Elementos objetivo (del status report) ───────────────────────────────
    msma_current:   float = 6997.58890      # km  — SMA media actual
    msma_target:    float = 6997.74      # km  — SMA media objetivo
    e_target:       float = 0.0012       # excentricidad media objetivo

    # ── Propulsión ───────────────────────────────────────────────────────────
    thrust_n:       float = 14.0         # N
    mass_kg:        float = 2809.961098  # kg
    efficiency:     float = 0.92

    # ── Maniobra anterior ────────────────────────────────────────────────────
    # Para SSO frozen (ω ≈ 90°):
    #   'north' → quemar cerca del POLO NORTE (perigeo, u≈90°)
    #             → sube 'e' levemente
    #   'south' → quemar cerca del POLO SUR   (apogeo,  u≈270°)
    #             → baja 'e' levemente
    # Se alternan para mantener |e| estable a largo plazo.
    # Si la última fue norte retrógrada y hay que corregir: también 'north'
    target_hemisphere: str = 'north'

    # ── TLE ──────────────────────────────────────────────────────────────────
    tle_file: str = "S1B_OPER_ODF_ORBDET_CODS_20260422T114403_TLE_O_1.TLE"

    # ── XML de eventos ───────────────────────────────────────────────────────
    xml_file: str = "S1B_OPER_ODF_EVENTS_CODS_20260422T114403_EVENTS_O_1.XML"

    # ── Filtros de contacto ──────────────────────────────────────────────────
    excluded_stations: list = field(default_factory=lambda: [
        'IML', 'HTN', 'LIM', 'IVK', 'KRU', 'SAA', 'IMT',  'TSS'
    ])
    min_duration_min: float = 10.0
    min_elevation_deg: float = 10.0

    # ── Sampleo dentro del contacto ──────────────────────────────────────────
    burn_sample_step_s: float = 10.0     # paso de búsqueda dentro del contacto

    # ── Ventana de promedio orbital ───────────────────────────────────────────
    avg_window_s:  float = 86400.0       # ~media órbita cada lado del burn
    avg_step_s:    float = 10.0

    # ── Pesos de la función objetivo ─────────────────────────────────────────
    w_error_target: float = 1e8   # error vs e_target_vec  (dominante)
    w_lat_penalty:  float = 1e6   # penalización latitud burn
    w_delta_e:      float = 1e4   # Δe adicional introducido

    # ── Output ───────────────────────────────────────────────────────────────
    top_n: int = 20

    # ── Constantes ───────────────────────────────────────────────────────────
    mu: float = 398600.4418   # km³/s²


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITARIOS ORBITALES
# ══════════════════════════════════════════════════════════════════════════════
# Load TLE from text function
def load_tle_from_file(path):
    # Get TLE lines 1 and 2 from file
    with open(path, "r") as f:
        lines = f.readlines()
    return lines[1], lines[2]


def datetime_to_jd(dt: datetime) -> tuple[int, float]:
    ref   = datetime(2000, 1, 1, 12, 0, 0)
    days  = (dt - ref).total_seconds() / 86400.0
    jd    = 2451545.0 + days
    jd_i  = int(jd)
    return jd_i, jd - jd_i


def jd_to_datetime(jd: float, fr: float) -> datetime:
    ref  = datetime(2000, 1, 1, 12, 0, 0)
    return ref + timedelta(days=(jd - 2451545.0) + fr)


def get_state(sat: Satrec, dt: datetime) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    jd, fr = datetime_to_jd(dt)
    err, r, v = sat.sgp4(jd, fr)
    if err != 0:
        return None, None
    return np.array(r), np.array(v)


def eccentricity_vector(r: np.ndarray, v: np.ndarray, mu: float) -> np.ndarray:
    r_n = np.linalg.norm(r)
    return (np.dot(v, v) * r - np.dot(r, v) * v) / mu - r / r_n


def tangential_unit(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Dirección tangencial (perpendicular a r, en el plano orbital)."""
    r_n  = np.linalg.norm(r)
    v_t  = v - np.dot(v, r) / r_n**2 * r
    return v_t / np.linalg.norm(v_t)


def geocentric_latitude_rad(r: np.ndarray) -> float:
    return float(np.arcsin(r[2] / np.linalg.norm(r)))


def rv2sma(r: np.ndarray, v: np.ndarray, mu: float) -> float:
    """ Obtener SMA a partir de los vectores r, v """
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)

    return (1 / (2 / r_mag - v_mag ** 2 / mu))

# ══════════════════════════════════════════════════════════════════════════════
#  PROMEDIO ORBITAL (e-vector medio)
# ══════════════════════════════════════════════════════════════════════════════

def mean_params_pre(sat, t_burn, window_s, step_s, mu):
    e_list = []
    a_list = []
    for dt_s in np.arange(0, window_s + step_s, step_s):
        t    = t_burn - timedelta(seconds=float(dt_s))
        r, v = get_state(sat, t)
        if r is None:
            continue
        e_list.append(eccentricity_vector(r, v, mu))
        a_list.append(rv2sma(r, v, mu))

    a_arr = np.array(a_list)
    print(f"  [PRE]  a_osc min={a_arr.min():.4f}  max={a_arr.max():.4f}  "
          f"mean={a_arr.mean():.4f}  std={a_arr.std():.4f}  n={len(a_arr)}")

    return np.mean(e_list, axis=0), np.mean(a_list)

"""
def mean_params_post(sat, t_burn, dv_vec, window_s, step_s, mu):
    r0, v0  = get_state(sat, t_burn)
    v0_post = v0 + dv_vec

    states  = propagate_rk4(r0, v0_post, window_s, step_s, mu)

    e_list  = [eccentricity_vector(r, v, mu) for _, r, v in states]
    a_list  = [rv2sma(r, v, mu)              for _, r, v in states]

    return np.mean(e_list, axis=0), np.mean(a_list)
"""
def mean_params_post(sat, t_burn, dv_vec, window_s, step_s, mu):
    """
    Post-burn consistente: propaga con RK4 solo para calcular la
    DIFERENCIA respecto al TLE, luego la suma al SGP4.
    
    e_post(t) ≈ e_SGP4(t) + [e_RK4_postburn(t) - e_RK4_preburn(t)]
                              └──────── Δe por el burn ────────────┘
    """
    r0, v0 = get_state(sat, t_burn)
    if r0 is None:
        return np.zeros(3), 0.0

    # Propagar SIN burn (referencia RK4)
    states_pre  = propagate_rk4(r0, v0,          window_s, step_s, mu)
    # Propagar CON burn
    states_post = propagate_rk4(r0, v0 + dv_vec, window_s, step_s, mu)

    e_list = []
    a_list = []

    for (_, r_pre, v_pre), (_, r_post, v_post) in zip(states_pre, states_post):
        # Δ introducido por el burn (RK4 se cancela parcialmente)
        delta_e = eccentricity_vector(r_post, v_post, mu) - \
                  eccentricity_vector(r_pre,  v_pre,  mu)
        delta_a = rv2sma(r_post, v_post, mu) - rv2sma(r_pre, v_pre, mu)

        # Aplicar Δ sobre el SGP4 (que tiene drag y correcciones empíricas)
        t = t_burn + timedelta(seconds=float(_))
        r_sgp4, v_sgp4 = get_state(sat, t)
        if r_sgp4 is None:
            continue

        e_list.append(eccentricity_vector(r_sgp4, v_sgp4, mu) + delta_e)
        a_list.append(rv2sma(r_sgp4, v_sgp4, mu) + delta_a)

    return np.mean(e_list, axis=0), np.mean(a_list)


def reference_e_direction(sat: Satrec, t0: datetime, mu: float) -> np.ndarray:
    """Dirección del e-vector media sobre 7 días anteriores (condición frozen actual)."""
    e_list = []
    for dt_s in np.arange(0, 7 * 86400, 60):
        t    = t0 - timedelta(seconds=float(dt_s))
        r, v = get_state(sat, t)
        if r is None:
            continue
        e_list.append(eccentricity_vector(r, v, mu))

    e_mean = np.mean(e_list, axis=0)
    norm   = np.linalg.norm(e_mean)
    return e_mean / norm if norm > 1e-12 else e_mean

def propagate_rk4(r0, v0, duration_s, step_s, mu,
                  J2=1.08263e-3, J3=-2.54e-6, Re=6378.137,
                  B=0.0):
    """
    Integrador RK4 con J2 + J3 (Montenbruck & Gill, Cap 3.2).
    
    Aceleraciones en ECI (km/s²), r en km.
    B = Cd * A / m [km²/kg] — coeficiente de drag (0 = sin drag).
    Retorna lista de (t_s, r, v).
    """

    def accel(r, v):
        x, y, z = r
        r_n  = np.linalg.norm(r)
        r2   = r_n * r_n
        r3   = r2  * r_n
        r4   = r2  * r2
        r5   = r4  * r_n

        # ── Gravedad central ─────────────────────────────────────────────
        a = -mu / r3 * r

        # ── J2 (Montenbruck & Gill 3.75) ─────────────────────────────────
        # a_J2 = -3/2 · J2 · μ · Re² / r⁵ · [x(1-5z²/r²), y(1-5z²/r²), z(3-5z²/r²)]
        c2   = -1.5 * J2 * mu * Re**2 / r5
        z2r2 = z * z / r2
        a   += c2 * np.array([
            x * (1.0 - 5.0 * z2r2),
            y * (1.0 - 5.0 * z2r2),
            z * (3.0 - 5.0 * z2r2),
        ])

        # ── J3 (Montenbruck & Gill 3.76) ─────────────────────────────────
        # a_J3 = 5/2 · J3 · μ · Re³ / r⁷ · [x·z·(3-7z²/r²),
        #                                      y·z·(3-7z²/r²),
        #                                      3r²/5 - 6z² + 7z⁴/r²]  × (−1)
        # Signo: el término J3 tiene signo negativo en la expansión del potencial
        c3   = 0.5 * J3 * mu * Re**3 / (r5 * r2)   # ×5 ya incluido abajo
        z2   = z * z
        z4   = z2 * z2
        a   += c3 * np.array([
            -7.5 * x * z * (1.0 - 7.0 * z2 / (3.0 * r2)),
            -7.5 * y * z * (1.0 - 7.0 * z2 / (3.0 * r2)),
             3.0 * (1.0 - 10.0 * z2 / r2 + 35.0 * z4 / (3.0 * r4)),
        ])

        # ── Drag atmosférico (modelo exponencial muy simplificado) ────────
        # Solo activo si B > 0
        if B > 0.0:
            v_n  = np.linalg.norm(v)
            a   += -B * v_n * v       # F_drag = -B·|v|·v  [km/s²]

        return a

    def rk4_step(r, v, h):
        k1r = v
        k1v = accel(r, v)

        k2r = v + 0.5 * h * k1v
        k2v = accel(r + 0.5 * h * k1r, v + 0.5 * h * k1v)

        k3r = v + 0.5 * h * k2v
        k3v = accel(r + 0.5 * h * k2r, v + 0.5 * h * k2v)

        k4r = v + h * k3v
        k4v = accel(r + h * k3r, v + h * k3v)

        r_new = r + h / 6.0 * (k1r + 2*k2r + 2*k3r + k4r)
        v_new = v + h / 6.0 * (k1v + 2*k2v + 2*k3v + k4v)
        return r_new, v_new

    r, v   = r0.copy(), v0.copy()
    states = [(0.0, r.copy(), v.copy())]

    for t in np.arange(step_s, duration_s + step_s, step_s):
        r, v = rk4_step(r, v, step_s)
        states.append((float(t), r.copy(), v.copy()))

    return states
# ══════════════════════════════════════════════════════════════════════════════
#  SCORE DEL PUNTO DE QUEMA
# ══════════════════════════════════════════════════════════════════════════════

def burn_point_penalty(r: np.ndarray, v: np.ndarray, hemisphere: str) -> float:
    """
    Penalización por punto de quema.

    Para una SSO frozen (ω ≈ 90°):
      - Perigeo en polo NORTE (u ≈ 90°,  lat ≈ +97.9° - 90° ≈ polo N)
      - Apogeo  en polo SUR   (u ≈ 270°, lat ≈ polo S)

    Una quema tangencial en el perigeo/apogeo cambia 'a' sin rotar el
    e-vector. Una quema en el ecuador SÍ rota el e-vector, rompiendo
    la condición frozen.

    Alternancia:
      'north' → quemar cerca del polo norte (perigeo) → sube e levemente
      'south' → quemar cerca del polo sur   (apogeo)  → baja e levemente

    Retorna desviación angular en rad respecto al polo objetivo [0, π].
    """
    lat = geocentric_latitude_rad(r)   # rad, en [-π/2, +π/2]

    if hemisphere == 'north':
        optimal_lat = np.pi / 2        # polo norte
    else:
        optimal_lat = -np.pi / 2       # polo sur

    return abs(lat - optimal_lat)      # 0 = óptimo, π = polo opuesto


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUACIÓN DE UN CANDIDATO DE QUEMA
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BurnCandidate:
    station:    str
    aos:        datetime
    los:        datetime
    duration_min: float
    rev_num:    str
    el_max:     float
    burn_time:  datetime
    burn_lat_deg: float
    burn_vz_dir:  str          # '↑N' o '↓S'
    e_pre:      float
    e_post:     float
    delta_e:    float
    error_target: float
    a_pre:       float
    a_post:      float
    delta_a:      float
    lat_penalty:  float
    score:      float


def evaluate_burn(
    sat:           Satrec,
    t_burn:        datetime,
    dv_vec:        np.ndarray,
    e_pre_vec:     np.ndarray,
    e_target_vec:  np.ndarray,
    a_pre:         float,  
    cfg:           Config,
) -> Optional[dict]:
    """Evalúa un instante de quema y retorna métricas + score."""
    
    r0, v0 = get_state(sat, t_burn)
    lat_penalty  = burn_point_penalty(r0, v0, cfg.target_hemisphere)

    # e-vector pre y post quema (promedio orbital centrado en t_burn)
    e_post_vec, a_post = mean_params_post(sat, t_burn, dv_vec, cfg.avg_window_s, cfg.avg_step_s, cfg.mu)

    e_pre        = float(np.linalg.norm(e_pre_vec))
    e_post       = float(np.linalg.norm(e_post_vec))
    delta_e      = float(np.linalg.norm(e_post_vec - e_pre_vec))
    delta_a      = a_post - a_pre
    error_target = float(np.linalg.norm(e_post_vec - e_target_vec))

    score = (
        - cfg.w_error_target * error_target
        - cfg.w_lat_penalty  * lat_penalty
        - cfg.w_delta_e      * delta_e
    )

    lat_deg  = np.degrees(geocentric_latitude_rad(r0))
    # Indicador de proximidad al polo objetivo
    if cfg.target_hemisphere == 'north':
        pole_dist = abs(lat_deg - 90.0)
    else:
        pole_dist = abs(lat_deg + 90.0)
    vz_dir   = f"Δpole={pole_dist:.1f}°"

    return dict(
        e_pre=e_pre, e_post=e_post, delta_e=delta_e, a_pre=a_pre, a_post=a_post, 
        delta_a=delta_a, error_target=error_target, lat_penalty=lat_penalty,
        score=score, lat_deg=lat_deg, vz_dir=vz_dir,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  LECTURA DEL XML DE EVENTOS
# ══════════════════════════════════════════════════════════════════════════════

def load_contacts(cfg: Config) -> list[dict]:
    tree = ET.parse(cfg.xml_file)
    root = tree.getroot()

    contacts = []
    for event in root.findall('.//Event'):
        station  = event.findtext('eventID', 'N/A')
        start_s  = event.findtext('start')
        stop_s   = event.findtext('stop')
        el_max   = float(event.findtext('elMax', 0))
        rev_num  = event.findtext('revNum', 'N/A')

        try:
            start = datetime.strptime(start_s, "%Y/%m/%dT%H:%M:%S.%f")
            stop  = datetime.strptime(stop_s,  "%Y/%m/%dT%H:%M:%S.%f")
        except Exception:
            continue

        if (cfg.current_date > start) or (start > cfg.exit_window):
            continue
        if station in cfg.excluded_stations:
            continue

        dur_min = (stop - start).total_seconds() / 60
        if dur_min < cfg.min_duration_min:
            continue
        if el_max < cfg.min_elevation_deg:
            continue

        contacts.append(dict(
            station=station, start=start, stop=stop,
            duration_min=dur_min, rev_num=rev_num, el_max=el_max,
        ))

    return contacts


# ══════════════════════════════════════════════════════════════════════════════
#  OPTIMIZACIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def optimize(cfg: Config) -> list[BurnCandidate]:

    # ── Δv ──────────────────────────────────────────────────────────────────
    delta_a    = cfg.msma_target - cfg.msma_current
    v_orb      = np.sqrt(cfg.mu / cfg.msma_current)
    delta_v_ms = (v_orb * delta_a / (2 * cfg.msma_current)) * 1000.0  # m/s
    burn_dur_s = (cfg.mass_kg * delta_v_ms) / (cfg.thrust_n * cfg.efficiency)

    print("\n" + "═" * 60)
    print("  SAOCOM-1B — MANEUVER OPTIMIZER")
    print("═" * 60)
    print(f"  Δa required    : {delta_a:+.5f} km")
    print(f"  Δv required    : {delta_v_ms:+.5f} m/s")
    print(f"  Burn duration  : {burn_dur_s:.2f} s  ({int(round(burn_dur_s))} s)")
    print(f"  Target hemi    : {cfg.target_hemisphere.upper()}")
    print("═" * 60 + "\n")

    # ── Satélite ─────────────────────────────────────────────────────────────
    sat   = Satrec.twoline2rv(tle_line1, tle_line2)
    t_tle = jd_to_datetime(sat.jdsatepoch, sat.jdsatepochF)

    # e-vector de referencia (dirección frozen actual)
    print("  Computing reference e-vector (3-day average)...")
    e_ref_hat     = reference_e_direction(sat, t_tle, cfg.mu)
    e_target_vec  = cfg.e_target * e_ref_hat

    print(f"  e_target_vec   : {e_target_vec}")
    print()

    # ── Contactos ────────────────────────────────────────────────────────────
    contacts = load_contacts(cfg)
    print(f"  Contacts found : {len(contacts)}\n")

    # ── Búsqueda ─────────────────────────────────────────────────────────────
    candidates: list[BurnCandidate] = []

    for ct in contacts:
        best_local: Optional[dict] = None

        duration_s = (ct['stop'] - ct['start']).total_seconds()

        # Calculate e_vector in range [t_start - 24h, t_start]
        e_pre_vec, a_pre  = mean_params_pre(sat, ct['start'], cfg.avg_window_s, cfg.avg_step_s, cfg.mu)

        for dt_s in np.arange(0, duration_s, cfg.burn_sample_step_s):

            t_burn = ct['start'] + timedelta(seconds=float(dt_s))

            r, v = get_state(sat, t_burn)
            if r is None:
                continue

            # Δv en dirección tangencial
            dv_vec = (delta_v_ms / 1000.0) * tangential_unit(r, v)

            result = evaluate_burn(sat, t_burn, dv_vec, e_pre_vec, e_target_vec, a_pre, cfg)
            if result is None:
                continue

            if best_local is None or result['score'] > best_local['score']:
                best_local = {**result, 'burn_time': t_burn}

        if best_local:
            candidates.append(BurnCandidate(
                station     = ct['station'],
                aos         = ct['start'],
                los         = ct['stop'],
                duration_min= ct['duration_min'],
                rev_num     = ct['rev_num'],
                el_max      = ct['el_max'],
                burn_time   = best_local['burn_time'],
                burn_lat_deg= best_local['lat_deg'],
                burn_vz_dir = best_local['vz_dir'],
                e_pre       = best_local['e_pre'],
                e_post      = best_local['e_post'],
                delta_e     = best_local['delta_e'],
                error_target= best_local['error_target'],
                a_pre       = best_local['a_pre'],
                a_post      = best_local['a_post'],
                delta_a     = best_local['delta_a'],
                lat_penalty = best_local['lat_penalty'],
                score       = best_local['score'],
            ))
            print(
                f"  [{ct['station']:3s}] {ct['start'].strftime('%m-%d %H:%M')} "
                f"lat={best_local['lat_deg']:+6.1f}°  {best_local['vz_dir']}  "
                f"e_post={best_local['e_post']:.6f}  "
                f"a_post={best_local['a_post']:.6f}  "
                f"score={best_local['score']:.2f}"
            )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates, delta_v_ms, burn_dur_s


# ══════════════════════════════════════════════════════════════════════════════
#  OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def print_results(
    candidates:   list[BurnCandidate],
    delta_v_ms:   float,
    burn_dur_s:   float,
    cfg:          Config,
) -> None:

    if not candidates:
        print("\n  No valid candidates found.")
        return

    best = candidates[0]

    print("\n" + "═" * 70)
    print("  BEST SOLUTION")
    print("═" * 70)
    print(f"  Station        : {best.station}")
    print(f"  AOS            : {best.aos.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  LOS            : {best.los.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  Duration       : {best.duration_min:.1f} min")
    print(f"  Max Elevation  : {best.el_max:.1f}°")
    print(f"  Revolution     : {best.rev_num}")
    print(f"  Burn time      : {best.burn_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  Burn duration  : {int(round(burn_dur_s))} s")
    print(f"  Δv             : {delta_v_ms:.5f} m/s")
    print(f"  Burn latitude  : {best.burn_lat_deg:+.2f}° {best.burn_vz_dir}")
    print()
    print(f"  e initial      : {best.e_pre:.8f}")
    print(f"  e final        : {best.e_post:.8f}")
    print(f"  Δe             : {best.delta_e:.4e}")
    print(f"  error_target   : {best.error_target:.4e}")
    print(f"  a initial      : {best.a_pre:.6e}")
    print(f"  a final        : {best.a_post:.6e}")
    print(f"  Δa             : {best.delta_a:.6e}")
    print(f"  lat_penalty    : {best.lat_penalty:.4f} rad")
    print(f"  score          : {best.score:.4f}")

    # ── Tabla top-N ──────────────────────────────────────────────────────────
    N = min(cfg.top_n, len(candidates))

    pole_label = 'Δpole_N' if cfg.target_hemisphere == 'north' else 'Δpole_S'
    hdr = (
        f"\n{'':─<135}\n"
        f"  TOP {N} CANDIDATES  (target hemisphere: {cfg.target_hemisphere.upper()})\n"
        f"{'':─<135}\n"
        f"  {'#':>3}  {'STA':4}  {'AOS':19}  {'LOS':8}  {'EL':>5}  "
        f"{'DUR':>6}  {'BURN UTC':8}  {'LAT':>7}  {pole_label:>8}  "
        f"{'e_post':>10}  {'Δe':>8}  {'err_tgt':>9} {'a_post':>9} {'score':>10}\n"
        f"{'':─<135}"
    )
    print(hdr)

    for i, c in enumerate(candidates[:N], 1):
        # Extraer Δpole del string almacenado (e.g. "Δpole=12.3°")
        try:
            pole_val = c.burn_vz_dir.split('=')[1]   # "12.3°"
        except Exception:
            pole_val = c.burn_vz_dir
        print(
            f"  {i:>3}. {c.station:4s}  "
            f"{c.aos.strftime('%Y-%m-%d %H:%M:%S')}  "
            f"{c.los.strftime('%H:%M:%S')}  "
            f"{c.el_max:>5.1f}°  "
            f"{c.duration_min:>5.1f}m  "
            f"{c.burn_time.strftime('%H:%M:%S')}  "
            f"{c.burn_lat_deg:>+6.1f}°  "
            f"{pole_val:>8}  "
            f"{c.e_post:>10.4e}  "
            f"{c.delta_e:>8.4e}  "
            f"{c.error_target:>9.4e}  "
            f"{c.a_post:>9.6f}"
            f"{c.score:>10.2f}"
        )

    print(f"{'':─<135}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = Config()
    tle_line1, tle_line2 = load_tle_from_file(cfg.tle_file)
    candidates, dv, bd = optimize(cfg)  
    print_results(candidates, dv, bd, cfg)