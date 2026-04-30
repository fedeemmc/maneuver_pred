#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAOCOM-1B — Orbit Maintenance Maneuver Optimizer
=================================================
Estrategia:
  - SGP4 (TLE) para toda la propagación
  - Burn instantáneo de primer orden:
      · Δe_vec y Δa calculados exactamente en t_burn (una sola vez)
      · Sumados como offset constante sobre la ventana post-burn SGP4
  - El Δe reportado es el del burn en t_burn, NO la diferencia de
    promedios osculadores (que tiene bias y drift secular)
  - e_post_vec = e_pre_vec_medio + Δe_vec_burn
    donde e_pre_vec_medio es el promedio osculador pre-burn

Convención de hemisferio para SSO frozen (ω ≈ 90°):
  - Quema NORTE → polo norte (perigeo, u ≈ 90°)  → sube |e| levemente
  - Quema SUR   → polo sur   (apogeo,  u ≈ 270°) → baja |e| levemente
  Se alternan para mantener |e| estable a largo plazo.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from sgp4.api import Satrec

import argparse

from ftp_utils import (
    get_latest_file,
    get_tle_history,
    TLE_REGEX,
    EVENT_REGEX,
    get_file_closest_to_date,
    get_latest_tle,
    get_latest_xml
)

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # ── Misión ───────────────────────────────────────────────────────────────
    current_date:   datetime = datetime(2026, 1, 23)
    exit_window:    datetime = datetime(2026, 1, 24)

    # ── Elementos objetivo (del status report) ────────────────────────────────
    #msma_current:   float = 6997.64890   # km  — SMA media actual
    msma_target:    float = 6997.76      # km  — SMA media objetivo
    e_target:       float = 0.0012       # excentricidad media objetivo

    # ── Propulsión ────────────────────────────────────────────────────────────
    thrust_n:       float = 14.0         # N
    mass_kg:        float = 2809.961098  # kg
    efficiency:     float = 0.92

    # ── Hemisferio objetivo ───────────────────────────────────────────────────
    # 'north' → quemar cerca del POLO NORTE (perigeo, u ≈ 90°)
    # 'south' → quemar cerca del POLO SUR   (apogeo,  u ≈ 270°)
    target_hemisphere: str = 'south'

    # ── Archivos ──────────────────────────────────────────────────────────────
    tle_file: str = "S1B_OPER_ODF_ORBDET_CODS_20260122T202010_TLE_O_1.TLE"
    xml_file: str = "S1B_OPER_ODF_EVENTS_CODS_20260122T145528_EVENTS_O_1.XML"

    # ── Filtros de contacto ───────────────────────────────────────────────────
    excluded_stations: list = field(default_factory=lambda: [
        'IML', 'HTN', 'LIM', 'IVK', 'KRU', 'SAA', 'IMT', 'TSS'
    ])
    min_duration_min:  float = 10.0
    min_elevation_deg: float = 10.0

    # ── Sampleo dentro del contacto ───────────────────────────────────────────
    burn_sample_step_s: float = 10.0

    # ── Ventana de promedio orbital ───────────────────────────────────────────
    # Usada solo para estimar e_pre_medio (referencia pre-burn).
    # El Δe del burn se calcula directamente en t_burn, no de este promedio.
    # Recomendado: múltiplo entero del período orbital (~5957 s).
    avg_window_s: float = 58292.0     # ~14.5 órbitas
    avg_step_s:   float = 10.0

    # ── Pesos de la función objetivo ──────────────────────────────────────────
    w_e_error_target: float = 1e8   # |e_post_vec - e_target_vec|  (dominante)
    w_lat_penalty:  float = 1e6   # desviación del polo objetivo
    w_delta_e:      float = 1e4   # |Δe_vec| del burn

    # ── Output ────────────────────────────────────────────────────────────────
    top_n: int = 20

    # ── Debug ─────────────────────────────────────────────────────────────────
    debug: bool = False

    # ── Constantes ────────────────────────────────────────────────────────────
    mu: float = 398600.4418   # km³/s²
    earth_r: float = 6378.137 # km

    # ── FTP ───────────────────────────────────────────────────────────────────
    host = "192.168.86.186"
    port = 22
    username = "moc"
    password = "C0Ds.CM0c"

    satellite = "S1B"   # o "S1A"

    tle_timestamp: datetime = None
    xml_timestamp: datetime = None



# ══════════════════════════════════════════════════════════════════════════════
#  UTILITARIOS
# ══════════════════════════════════════════════════════════════════════════════

def load_tle_from_file(path: str) -> tuple[str, str]:
    with open(path) as f:
        lines = f.readlines()
    return lines[1].strip(), lines[2].strip()


def datetime_to_jd(dt: datetime) -> tuple[int, float]:
    ref  = datetime(2000, 1, 1, 12, 0, 0)
    days = (dt - ref).total_seconds() / 86400.0
    jd   = 2451545.0 + days
    jd_i = int(jd)
    return jd_i, jd - jd_i


def jd_to_datetime(jd: float, fr: float) -> datetime:
    return datetime(2000, 1, 1, 12) + timedelta(days=(jd - 2451545.0) + fr)


def get_state(sat: Satrec, dt: datetime):
    jd, fr = datetime_to_jd(dt)
    err, r, v = sat.sgp4(jd, fr)
    if err != 0:
        return None, None
    return np.array(r), np.array(v)


def eccentricity_vector(r: np.ndarray, v: np.ndarray, mu: float) -> np.ndarray:
    r_n = np.linalg.norm(r)
    return (np.dot(v, v) * r - np.dot(r, v) * v) / mu - r / r_n


def rv2sma(r: np.ndarray, v: np.ndarray, mu: float) -> float:
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)
    denom = 2.0 / r_mag - v_mag**2 / mu
    return 1.0 / denom if denom > 0 else float('inf')


def tangential_unit(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    r_n = np.linalg.norm(r)
    v_t = v - np.dot(v, r) / r_n**2 * r
    return v_t / np.linalg.norm(v_t)


def geocentric_latitude_rad(r: np.ndarray) -> float:
    return float(np.arcsin(np.clip(r[2] / np.linalg.norm(r), -1.0, 1.0)))


def compute_mean_eccentricity_vector(
    tle_files: list,
    sample_per_orbit: int = 10,
    mu: float = 398600.4418
):
    """
    Calcula e⃗ promedio robusto usando múltiples TLEs.
    """

    e_vectors = []

    for path, ts in tle_files:

        with open(path) as f:
            lines = f.readlines()

        if len(lines) < 3:
            continue

        l1 = lines[1].strip()
        l2 = lines[2].strip()

        sat = Satrec.twoline2rv(l1, l2)

        # samplear alrededor del epoch del TLE
        for dt_s in np.linspace(0, 6000, sample_per_orbit):
            t = ts + timedelta(seconds=float(dt_s))

            r, v = get_state(sat, t)
            if r is None:
                continue

            e_vec = eccentricity_vector(r, v, mu)
            e_vectors.append(e_vec)

    if not e_vectors:
        return np.zeros(3)

    e_mean = np.mean(e_vectors, axis=0)
    norm = np.linalg.norm(e_mean)

    return e_mean / norm, norm

def build_config_from_cli() -> Config:
    parser = argparse.ArgumentParser(
        description="SAOCOM Maneuver Optimizer"
    )

    parser.add_argument("--current_date", type=str)
    parser.add_argument("--exit_window", type=str)

    parser.add_argument("--target_hemisphere", type=str, required=True,
                        choices=["north", "south"])

    parser.add_argument("--tle_file", type=str)
    parser.add_argument("--xml_file", type=str)

    parser.add_argument("--top_n", type=int, default=20)

    parser.add_argument("--satellite", type=str, required=True,
                        choices=["S1A", "S1B"])

    args = parser.parse_args()

    # ── Defaults dinámicos ─────────────────────────────

    now = datetime.now()

    current_date = (
        parse_datetime(args.current_date)
        if args.current_date else now
    )

    exit_window = (
        parse_datetime(args.exit_window)
        if args.exit_window else current_date + timedelta(days=1)
    )

    # ── Construcción base ─────────────────────────────

    cfg = Config()

    cfg.current_date = current_date
    cfg.exit_window  = exit_window
    cfg.target_hemisphere = args.target_hemisphere
    cfg.top_n = args.top_n
    cfg.satellite = args.satellite

    # ── Archivos TLE / XML ────────────────────────────

    cfg.tle_file = args.tle_file
    cfg.xml_file = args.xml_file

    return cfg

def parse_datetime(dt_str: str) -> datetime:
    return datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S")


# ══════════════════════════════════════════════════════════════════════════════
#  PERTURBACIÓN INSTANTÁNEA DEL BURN
# ══════════════════════════════════════════════════════════════════════════════

def burn_delta(
    r0: np.ndarray, v0: np.ndarray, dv_vec: np.ndarray, mu: float
) -> tuple[np.ndarray, float]:
    """
    Variación exacta (primer orden) de e-vector y SMA por un burn.

    Calculada una sola vez en t_burn a partir del estado osculador SGP4.
    Para Δv = 0.08 m/s vs v_orb ≈ 7.55 km/s el error de 2º orden
    es ~(Δv/v)² ≈ 1e-10 — completamente despreciable.

    Returns
    -------
    de_vec : np.ndarray  — variación del e-vector  (adimensional)
    da     : float       — variación del SMA [km]
    """
    e_pre  = eccentricity_vector(r0, v0,           mu)
    e_post = eccentricity_vector(r0, v0 + dv_vec,  mu)
    a_pre  = rv2sma(r0, v0,           mu)
    a_post = rv2sma(r0, v0 + dv_vec,  mu)
    return e_post - e_pre, a_post - a_pre


# ══════════════════════════════════════════════════════════════════════════════
#  PROMEDIO ORBITAL  (SGP4 puro)
# ══════════════════════════════════════════════════════════════════════════════

def mean_evec_sgp4(
    sat:       Satrec,
    t_center:  datetime,
    window_s:  float,
    step_s:    float,
    mu:        float,
    direction: int = -1,
) -> np.ndarray:
    """
    Promedio del e-vector osculador SGP4 sobre una ventana.

    direction = -1 → [t_center - window_s, t_center]   (pre-burn)
    direction = +1 → [t_center,  t_center + window_s]  (post-burn ref)

    Nota: este promedio tiene un bias respecto al elemento medio real
    de SGP4 porque la transformación osculador→medio es no lineal.
    Se usa solo como referencia; el Δe del burn se calcula por separado
    con burn_delta() para evitar que ese bias contamine Δe.
    """
    e_list = []
    for dt_s in np.arange(0, window_s + step_s, step_s):
        t    = t_center + timedelta(seconds=direction * float(dt_s))
        r, v = get_state(sat, t)
        if r is None:
            continue
        e_list.append(eccentricity_vector(r, v, mu))
    return np.mean(e_list, axis=0) if e_list else np.zeros(3)


def mean_sma_sgp4(
    sat:       Satrec,
    t_center:  datetime,
    window_s:  float,
    step_s:    float,
    mu:        float,
    direction: int = -1,
    da:        float = 0.0,
) -> float:
    """
    Promedio del SMA osculador SGP4 sobre una ventana.
    Si da != 0, suma el offset del burn a cada punto.
    """
    a_list = []
    for dt_s in np.arange(0, window_s + step_s, step_s):
        t    = t_center + timedelta(seconds=direction * float(dt_s))
        r, v = get_state(sat, t)
        if r is None:
            continue
        a_list.append(rv2sma(r, v, mu) + da)
    return float(np.mean(a_list)) if a_list else 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  SCORE DEL PUNTO DE QUEMA
# ══════════════════════════════════════════════════════════════════════════════

def burn_point_penalty(r: np.ndarray, hemisphere: str) -> float:
    """
    Penalización por latitud del punto de quema.

    Para SSO frozen (ω ≈ 90°) una quema tangencial en perigeo/apogeo
    cambia 'a' sin rotar el e-vector.

      'north' → óptimo en polo norte (lat = +90°, perigeo)
      'south' → óptimo en polo sur   (lat = -90°, apogeo)

    Retorna desviación en rad [0, π].
    """
    lat         = geocentric_latitude_rad(r)
    optimal_lat = np.pi / 2 if hemisphere == 'north' else -np.pi / 2
    return abs(lat - optimal_lat)


# ══════════════════════════════════════════════════════════════════════════════
#  CANDIDATO DE QUEMA
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BurnCandidate:
    station:       str
    aos:           datetime
    los:           datetime
    duration_min:  float
    rev_num:       str
    el_max:        float
    burn_time:     datetime
    burn_lat_deg:  float
    pole_dist_deg: float
    # Excentricidad
    e_pre:         float   # |e_pre_vec_medio|  (promedio osculador pre-burn)
    e_post:        float   # |e_pre_vec_medio + Δe_vec_burn|
    delta_e_burn:  float   # |Δe_vec_burn|  (solo efecto del burn)
    e_error_target:  float   # |e_post_vec - e_target_vec|
    # SMA
    a_pre:         float   # promedio osculador pre-burn [km]
    a_post:        float   # a_pre + Δa_burn [km]
    delta_a_burn:  float   # Δa del burn [km]
    # Score
    lat_penalty:   float
    score:         float


def evaluate_burn(
    sat:          Satrec,
    t_burn:       datetime,
    dv_vec:       np.ndarray,
    e_target_vec: np.ndarray,
    cfg:          Config,
) -> Optional[dict]:
    """
    Evalúa un instante de quema.

    Flujo:
      1. Calcular estado osculador en t_burn  → (r0, v0)
      2. burn_delta()  → Δe_vec, Δa  (exacto, primer orden)
      3. e_pre_vec = promedio osculador SGP4 en [t_burn - W, t_burn]
      4. e_post_vec = e_pre_vec + Δe_vec  (suma directa, sin segundo promedio)
      5. a_pre = promedio osculador SMA pre-burn
      6. a_post = a_pre + Δa

    De esta forma Δe refleja solo el efecto del burn, sin el bias
    del promedio osculador ni el drift secular entre ventanas.
    """
    r0, v0 = get_state(sat, t_burn)
    if r0 is None:
        return None

    # ── 1. Perturbación del burn ─────────────────────────────────────────────
    de_vec, da = burn_delta(r0, v0, dv_vec, cfg.mu)

    # ── 2. Estado medio pre-burn (promedio osculador SGP4) ───────────────────
    e_pre_vec = mean_evec_sgp4(
        sat, t_burn, cfg.avg_window_s, cfg.avg_step_s, cfg.mu, direction=-1
    )
    a_pre = mean_sma_sgp4(
        sat, t_burn, cfg.avg_window_s, cfg.avg_step_s, cfg.mu, direction=-1
    )

    # ── 3. Estado post-burn: suma directa del delta ──────────────────────────
    #    NO se hace un segundo promedio osculador post-burn.
    #    Razón: el promedio osculador post-burn tendría el mismo bias que
    #    e_pre_vec, entonces e_post_vec - e_pre_vec ≠ Δe_burn.
    #    Con la suma directa: e_post_vec - e_pre_vec = Δe_vec exactamente.
    e_post_vec = e_pre_vec + de_vec
    a_post     = a_pre + da

    # ── 4. Métricas ──────────────────────────────────────────────────────────
    e_pre        = float(np.linalg.norm(e_pre_vec))
    e_post       = float(np.linalg.norm(e_post_vec))
    delta_e_burn = float(np.linalg.norm(de_vec))
    e_error_target = float(np.linalg.norm(e_post_vec - e_target_vec))
    lat_penalty  = burn_point_penalty(r0, cfg.target_hemisphere)

    score = (
        - cfg.w_e_error_target * e_error_target
        - cfg.w_lat_penalty  * lat_penalty
        - cfg.w_delta_e      * delta_e_burn
    )

    lat_deg   = np.degrees(geocentric_latitude_rad(r0))
    pole_dist = (abs(lat_deg - 90.0) if cfg.target_hemisphere == 'north'
                 else abs(lat_deg + 90.0))

    if cfg.debug:
        print(f"    t_burn={t_burn.strftime('%H:%M:%S')}  "
              f"e_pre={e_pre:.6f}  e_post={e_post:.6f}  "
              f"|Δe_burn|={delta_e_burn:.2e}  "
              f"|Δa_burn|={da:.5f}  err={e_error_target:.2e}")

    return dict(
        e_pre=e_pre, e_post=e_post, delta_e_burn=delta_e_burn,
        a_pre=a_pre, a_post=a_post, delta_a_burn=da,
        e_error_target=e_error_target, lat_penalty=lat_penalty,
        score=score, lat_deg=lat_deg, pole_dist_deg=pole_dist,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  LECTURA DEL XML DE EVENTOS
# ══════════════════════════════════════════════════════════════════════════════

def load_contacts(cfg: Config) -> list[dict]:
    tree = ET.parse(cfg.xml_file)
    root = tree.getroot()
    contacts = []

    for event in root.findall('.//Event'):
        station = event.findtext('eventID', 'N/A')
        start_s = event.findtext('start')
        stop_s  = event.findtext('stop')
        el_max  = float(event.findtext('elMax', 0))
        rev_num = event.findtext('revNum', 'N/A')

        try:
            start = datetime.strptime(start_s, "%Y/%m/%dT%H:%M:%S.%f")
            stop  = datetime.strptime(stop_s,  "%Y/%m/%dT%H:%M:%S.%f")
        except Exception:
            continue

        if stop < cfg.current_date or start > cfg.exit_window:
            continue
        if station in cfg.excluded_stations:
            continue

        dur_min = (stop - start).total_seconds() / 60
        if dur_min < cfg.min_duration_min or el_max < cfg.min_elevation_deg:
            continue

        contacts.append(dict(
            station=station, start=start, stop=stop,
            duration_min=dur_min, rev_num=rev_num, el_max=el_max,
        ))

    return contacts


# ══════════════════════════════════════════════════════════════════════════════
#  OPTIMIZACIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def optimize(cfg: Config, sat: Satrec) -> tuple[list[BurnCandidate], float, float]:
    msma_current = sat.am * cfg.earth_r
    delta_a    = cfg.msma_target - msma_current
    v_orb      = np.sqrt(cfg.mu / msma_current)
    delta_v_ms = (v_orb * delta_a / (2 * msma_current)) * 1000.0  # m/s

    burn_dur_s = (cfg.mass_kg * delta_v_ms) / (cfg.thrust_n * cfg.efficiency)

    print("\n" + "═" * 60)
    print("  SAOCOM-1B — MANEUVER OPTIMIZER")
    print("═" * 60)
    print(f"  Δa required    : {delta_a:+.5f} km")
    print(f"  Δv required    : {delta_v_ms:+.5f} m/s")
    print(f"  Burn duration  : {burn_dur_s:.2f} s  ({int(round(burn_dur_s))} s)")
    print(f"  Target hemi    : {cfg.target_hemisphere.upper()}")
    print("═" * 60 + "\n")

    t_tle = jd_to_datetime(sat.jdsatepoch, sat.jdsatepochF)

    print("  Computing reference e-vector (30 days average)...")
    tle_files = get_tle_history(
        host=cfg.host,
        port=cfg.port,
        username=cfg.username,
        password=cfg.password,
        remote_dir="/TLE",
        satellite=cfg.satellite,
        center_date=cfg.current_date,
        months_back=3,
    )
    e_hat, e_mag = compute_mean_eccentricity_vector(tle_files, 10, cfg.mu)
    e_target_vec = e_hat * e_mag
    print(f"  e_target_vec   : {e_target_vec}\n")
    
    contacts = load_contacts(cfg)
    print(f"  Contacts found : {len(contacts)}\n")

    candidates: list[BurnCandidate] = []

    for ct in contacts:
        best_local: Optional[dict] = None
        duration_s = (ct['stop'] - ct['start']).total_seconds()

        for dt_s in np.arange(0, duration_s, cfg.burn_sample_step_s):
            t_burn = ct['start'] + timedelta(seconds=float(dt_s))

            r, v = get_state(sat, t_burn)
            if r is None:
                continue

            dv_vec = (delta_v_ms / 1000.0) * tangential_unit(r, v)
            result = evaluate_burn(sat, t_burn, dv_vec, e_target_vec, cfg)
            if result is None:
                continue

            if best_local is None or result['score'] > best_local['score']:
                best_local = {**result, 'burn_time': t_burn}

        if best_local:
            candidates.append(BurnCandidate(
                station       = ct['station'],
                aos           = ct['start'],
                los           = ct['stop'],
                duration_min  = ct['duration_min'],
                rev_num       = ct['rev_num'],
                el_max        = ct['el_max'],
                burn_time     = best_local['burn_time'],
                burn_lat_deg  = best_local['lat_deg'],
                pole_dist_deg = best_local['pole_dist_deg'],
                e_pre         = best_local['e_pre'],
                e_post        = best_local['e_post'],
                delta_e_burn  = best_local['delta_e_burn'],
                e_error_target  = best_local['e_error_target'],
                a_pre         = best_local['a_pre'],
                a_post        = best_local['a_post'],
                delta_a_burn  = best_local['delta_a_burn'],
                lat_penalty   = best_local['lat_penalty'],
                score         = best_local['score'],
            ))
            print(
                f"  [{ct['station']:3s}] {ct['start'].strftime('%m-%d %H:%M')}  "
                f"lat={best_local['lat_deg']:+6.1f}°  "
                f"Δpole={best_local['pole_dist_deg']:5.1f}°  "
                f"e_pre={best_local['e_pre']:.6f}  "
                f"e_post={best_local['e_post']:.6f}  "
                f"|Δe|={best_local['delta_e_burn']:.2e}  "
                f"a_post={best_local['a_post']:.4f}  "
                f"score={best_local['score']:.2f}"
            )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates, delta_v_ms, burn_dur_s


# ══════════════════════════════════════════════════════════════════════════════
#  OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def print_results(
    candidates: list[BurnCandidate],
    delta_v_ms: float,
    burn_dur_s: float,
    cfg:        Config,
) -> None:

    if not candidates:
        print("\n  No valid candidates found.")
        return

    best       = candidates[0]
    pole_label = 'Δpole_N' if cfg.target_hemisphere == 'north' else 'Δpole_S'

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
    print(f"  Burn latitude  : {best.burn_lat_deg:+.2f}°  "
          f"{pole_label}={best.pole_dist_deg:.1f}°")
    print()
    print(f"  e pre  (mean)  : {best.e_pre:.8f}")
    print(f"  e post (mean)  : {best.e_post:.8f}")
    print(f"  |Δe| burn      : {best.delta_e_burn:.4e}  ← solo efecto del burn")
    print(f"  e_error_target   : {best.e_error_target:.4e}")
    print(f"  a pre  (mean)  : {best.a_pre:.6f} km")
    print(f"  a post (mean)  : {best.a_post:.6f} km")
    print(f"  Δa burn        : {best.delta_a_burn:.6f} km")
    print(f"  lat_penalty    : {best.lat_penalty:.4f} rad")
    print(f"  score          : {best.score:.4f}")

    N   = min(cfg.top_n, len(candidates))
    sep = "─" * 150

    print(f"\n{sep}")
    print(f"  TOP {N} CANDIDATES  (target hemisphere: {cfg.target_hemisphere.upper()})")
    print(sep)
    print(
        f"  {'#':>3}  {'STA':4}  {'AOS':19}  {'LOS':8}  {'EL':>5}  "
        f"{'DUR':>6}  {'BURN UTC':8}  {'LAT':>7}  {pole_label:>8}  "
        f"{'e_pre':>10}  {'e_post':>10}  {'|Δe|burn':>10}  "
        f"{'e_err_tgt':>9}  {'a_pre':>9}  {'a_post':>10}"
        f"{'Δa_burn':>15}  {'score':>10}"
    )
    print(sep)

    for i, c in enumerate(candidates[:N], 1):
        print(
            f"  {i:>3}. {c.station:4s}  "
            f"{c.aos.strftime('%Y-%m-%d %H:%M:%S')}  "
            f"{c.los.strftime('%H:%M:%S')}  "
            f"{c.el_max:>5.1f}°  "
            f"{c.duration_min:>5.1f}m  "
            f"{c.burn_time.strftime('%H:%M:%S')}  "
            f"{c.burn_lat_deg:>+6.1f}°  "
            f"{c.pole_dist_deg:>7.1f}°  "
            f"{c.e_pre:>10.4e}  "
            f"{c.e_post:>10.4e}  "
            f"{c.delta_e_burn:>10.4e}  "
            f"{c.e_error_target:>9.4e}  "
            f"{c.a_pre:>9.6e}  "
            f"{c.a_post:>9.6e}  "
            f"{c.delta_a_burn:>9.5f}  "
            f"{c.score:>10.2f}"
        )

    print(f"{sep}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    cfg = build_config_from_cli()

    # ── Resolver archivos automáticamente ─────────────

    if cfg.tle_file is None:
        cfg.tle_file, cfg.tle_timestamp = get_file_closest_to_date(
            cfg.host, cfg.port, cfg.username, cfg.password,
            "/TLE", cfg.satellite, TLE_REGEX,
            cfg.current_date
        )
        print(f"[AUTO] Using TLE: {cfg.tle_file} ({cfg.tle_timestamp})")

    if cfg.xml_file is None:
        cfg.xml_file, cfg.xml_timestamp = get_file_closest_to_date(
            cfg.host, cfg.port, cfg.username, cfg.password,
            "/EVENTS.XML", cfg.satellite, EVENT_REGEX,
            cfg.current_date
        )
        print(f"[AUTO] Using XML: {cfg.xml_file} ({cfg.xml_timestamp})")

    # ── Cargar TLE ────────────────────────────────────

    tle_line1, tle_line2 = load_tle_from_file(cfg.tle_file)
    sat = Satrec.twoline2rv(tle_line1, tle_line2)

    # Debug opcional
    print(sat.am * cfg.earth_r)
    print(sat.em)

    candidates, dv, bd = optimize(cfg, sat)
    print_results(candidates, dv, bd, cfg)