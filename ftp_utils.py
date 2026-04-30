#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import gzip
import tempfile
from datetime import datetime, timedelta

import paramiko


# ══════════════════════════════════════════════════════════════════════
# REGEX
# ══════════════════════════════════════════════════════════════════════

TLE_REGEX = re.compile(
    r"(S1A|S1B)_OPER_ODF_ORBDET_CODS_(\d{8}T\d{6})_TLE_.*\.TLE\.GZ"
)

EVENT_REGEX = re.compile(
    r"(S1A|S1B)_OPER_ODF_EVENTS_CODS_(\d{8}T\d{6})_EVENTS_.*\.XML\.GZ"
)


def parse_file(filename, regex):
    m = regex.match(filename.strip())
    if not m:
        return None

    sat = m.group(1)
    ts  = datetime.strptime(m.group(2), "%Y%m%dT%H%M%S")

    return sat, ts


# ══════════════════════════════════════════════════════════════════════
# CONEXIÓN
# ══════════════════════════════════════════════════════════════════════

def connect_sftp(host, port, username, password):
    transport = paramiko.Transport((host, port))
    transport.connect(username=username, password=password)
    return paramiko.SFTPClient.from_transport(transport), transport


# ══════════════════════════════════════════════════════════════════════
# LISTADO + FILTRADO
# ══════════════════════════════════════════════════════════════════════

def list_filtered_files(sftp, remote_dir, satellite, regex):
    files = sftp.listdir(remote_dir)

    parsed = []
    for f in files:
        p = parse_file(f, regex)
        if not p:
            continue

        sat, ts = p
        if sat != satellite:
            continue

        parsed.append((f.strip(), ts))

    return parsed


# ══════════════════════════════════════════════════════════════════════
# DESCARGA + GUNZIP
# ══════════════════════════════════════════════════════════════════════

def download_and_extract(sftp, remote_dir, fname, tmp_dir):
    remote_path = f"{remote_dir}/{fname}"
    local_gz    = os.path.join(tmp_dir, fname)
    local_file  = local_gz[:-3]

    try:
        sftp.stat(remote_path)
    except FileNotFoundError:
        print(f"[WARN] Not found: {remote_path}")
        return None

    #print(f"[FTP] Downloading: {fname}")

    try:
        sftp.get(remote_path, local_gz)
    except Exception as e:
        print(f"[ERROR] {fname}: {e}")
        return None

    with gzip.open(local_gz, "rb") as f_in:
        with open(local_file, "wb") as f_out:
            f_out.write(f_in.read())

    return local_file


# ══════════════════════════════════════════════════════════════════════
# API PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

def get_latest_file(
    host, port, username, password,
    remote_dir, satellite, regex
):
    sftp, transport = connect_sftp(host, port, username, password)

    files = list_filtered_files(sftp, remote_dir, satellite, regex)

    if not files:
        raise RuntimeError(f"No files found in {remote_dir}")

    # ordenar por fecha
    files.sort(key=lambda x: x[1], reverse=True)

    tmp_dir = tempfile.mkdtemp(prefix="ftp_")

    latest_fname, ts = files[0]
    local_path = download_and_extract(sftp, remote_dir, latest_fname, tmp_dir)

    sftp.close()
    transport.close()

    return local_path, ts


def get_tle_history(
    host, port, username, password,
    remote_dir, satellite,
    center_date: datetime,
    months_back=3
):
    sftp, transport = connect_sftp(host, port, username, password)

    files = list_filtered_files(sftp, remote_dir, satellite, TLE_REGEX)

    # ── Ventana temporal coherente ─────────────────────
    cutoff = center_date - timedelta(days=30 * months_back)

    files = [(f, ts) for f, ts in files if cutoff <= ts <= center_date]

    files.sort(key=lambda x: x[1])

    tmp_dir = tempfile.mkdtemp(prefix="tle_hist_")

    results = []

    for fname, ts in files:
        path = download_and_extract(sftp, remote_dir, fname, tmp_dir)
        if path:
            results.append((path, ts))

    sftp.close()
    transport.close()

    return results

def get_file_closest_to_date(
    host, port, username, password,
    remote_dir, satellite, regex,
    target_date: datetime
):
    sftp, transport = connect_sftp(host, port, username, password)

    files = list_filtered_files(sftp, remote_dir, satellite, regex)

    if not files:
        raise RuntimeError(f"No files found in {remote_dir}")

    # elegir el más cercano a target_date
    files.sort(key=lambda x: abs((x[1] - target_date).total_seconds()))

    tmp_dir = tempfile.mkdtemp(prefix="ftp_")

    fname, ts = files[0]
    local_path = download_and_extract(sftp, remote_dir, fname, tmp_dir)

    sftp.close()
    transport.close()

    return local_path, ts

def get_latest_tle(cfg):
    return get_latest_file(
        host=cfg.host,
        port=cfg.port,
        username=cfg.username,
        password=cfg.password,
        remote_dir="/TLE",
        satellite=cfg.satellite,
        regex=TLE_REGEX
    )


def get_latest_xml(cfg):
    return get_latest_file(
        host=cfg.host,
        port=cfg.port,
        username=cfg.username,
        password=cfg.password,
        remote_dir="/EVENTS.XML",
        satellite=cfg.satellite,
        regex=EVENT_REGEX
    )