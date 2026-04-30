#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import gzip
import tempfile
from datetime import datetime, timedelta

import numpy as np
import paramiko
from sgp4.api import Satrec


# ══════════════════════════════════════════════════════════════════════
# CONFIG PARSER
# ══════════════════════════════════════════════════════════════════════

TLE_REGEX = re.compile(
    r"(S1A|S1B)_OPER_ODF_ORBDET_CODS_(\d{8}T\d{6})_TLE_.*\.TLE\.GZ"
)


def parse_tle_filename(filename: str):
    match = TLE_REGEX.match(filename)
    if not match:
        return None

    sat = match.group(1)
    timestamp = datetime.strptime(match.group(2), "%Y%m%dT%H%M%S")
    return sat, timestamp


# ══════════════════════════════════════════════════════════════════════
# SFTP DOWNLOAD
# ══════════════════════════════════════════════════════════════════════

def download_tle_files(
    host: str,
    port: int,
    username: str,
    password: str,
    remote_dir: str,
    satellite: str,
    months_back: int = 6,
):
    """
    Descarga archivos TLE.GZ del satélite seleccionado
    en la ventana temporal indicada.
    """

    cutoff_date = datetime.utcnow() - timedelta(days=30 * months_back)

    transport = paramiko.Transport((host, port))
    transport.connect(username=username, password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)

    files = sftp.listdir(remote_dir)

    selected_files = []

    for f in files:
        parsed = parse_tle_filename(f)
        if not parsed:
            continue

        sat, ts = parsed

        if sat != satellite:
            continue

        if ts < cutoff_date:
            continue

        selected_files.append((f, ts))

    # ordenar por fecha
    selected_files.sort(key=lambda x: x[1])

    local_paths = []

    tmp_dir = tempfile.mkdtemp(prefix="tle_")

    for fname, ts in selected_files:
        remote_path = f"{remote_dir}/{fname}"
        local_gz = os.path.join(tmp_dir, fname)

        print(f"[TLE] Downloading: {fname}")
        sftp.get(remote_path, local_gz)

        # descomprimir
        local_tle = local_gz[:-3]

        with gzip.open(local_gz, "rb") as f_in:
            with open(local_tle, "wb") as f_out:
                f_out.write(f_in.read())

        local_paths.append((local_tle, ts))

    sftp.close()
    transport.close()

    return local_paths
