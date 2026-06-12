"""
Conversion fichier .mat (Matecho/HAC) vers NetCDF au format ICES AcMeta.

Usage:
    python mat_to_netcdf.py <fichier_mat> [fichier_netcdf_sortie]

Dépendances:
    pip install mat73 numpy xarray netCDF4
"""

import sys
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

try:
    import mat73
except ImportError:
    raise ImportError("Installez mat73 : pip install mat73")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decode_datetime(time_unix):
    """Convertit un timestamp Unix (secondes) en datetime64[ns]."""
    return pd.to_datetime(time_unix, unit="s", utc=True).tz_localize(None)


def str_to_bytes_var(s, dim_name, n_freq):
    """Encode une chaîne en tableau de bytes (|S1) de dim (len(s), n_freq)."""
    arr = np.array([[c.encode() for c in s]] * n_freq, dtype="|S1").T
    return xr.Variable([dim_name, "channel"], arr)


# ---------------------------------------------------------------------------
# Conversion principale
# ---------------------------------------------------------------------------

def mat_to_netcdf(path_mat: str, path_out: str = None):
    print(f"Lecture de {path_mat} ...")
    mat = mat73.loadmat(path_mat)

    # ---- Fréquences ----
    freqs = np.array(mat.get("FMFreqHz", mat.get("FrequencySort",
                    [18000., 38000., 70000., 120000., 200000.])), dtype=np.float64)
    freqs_khz = freqs / 1000.0
    n_freq = len(freqs)

    # ---- Temps / pings ----
    time_unix = np.array(mat["Time"], dtype=np.float64)
    lat       = np.array(mat["Latitude"], dtype=np.float64)
    lon       = np.array(mat["Longitude"], dtype=np.float64)
    n_ping    = len(time_unix)
    time_dt   = decode_datetime(time_unix)

    # ---- Sv depuis sv_surface ----
    # sv_surface : (n_depth, n_ping, n_freq)  — shape Matecho
    sv_raw = np.array(mat["Sv_surface"], dtype=np.float32)
    print(f"  sv_surface shape brut : {sv_raw.shape}")

    # Réorganiser en (n_freq, n_ping, n_depth)
    if sv_raw.ndim == 3:
        n_depth = sv_raw.shape[0]
        # (n_depth, n_ping, n_freq) -> (n_freq, n_ping, n_depth)
        sv = np.transpose(sv_raw, (2, 1, 0))
    else:
        raise ValueError(f"sv_surface inattendu : shape={sv_raw.shape}")

    # ---- Axe de profondeur ----
    depth_axis = np.array(mat["depth_surface"], dtype=np.float64)
    if depth_axis.shape[0] != n_depth:
        # fallback si besoin
        depth_axis = np.arange(n_depth, dtype=np.float64)

    res = float(np.median(np.diff(depth_axis))) if len(depth_axis) > 1 else 1.0

    print(f"  Fréquences  : {freqs_khz} kHz")
    print(f"  Pings       : {n_ping}")
    print(f"  Profondeurs : {n_depth} couches "
          f"({depth_axis[0]:.1f} – {depth_axis[-1]:.1f} m, pas {res:.2f} m)")
    print(f"  Sv shape final (freq, ping, depth) : {sv.shape}")

    # ---- Métadonnées calibration ----
    calib_list    = mat.get("EIParameters", {}).get("Calibration", [{}] * n_freq)
    gains         = []
    sa_corrections = []
    transmit_powers = []

    for cal in calib_list[:n_freq]:
        cw         = cal.get("CW", {}) if isinstance(cal, dict) else {}
        cal_params = cw.get("CalibrationParameters", {}) if isinstance(cw, dict) else {}
        gains.append(float(cal_params.get("Gain", 0.0))           if isinstance(cal_params, dict) else 0.0)
        sa_corrections.append(float(cal_params.get("SaCorrection", 0.0)) if isinstance(cal_params, dict) else 0.0)
        transmit_powers.append(float(cal_params.get("TransmitPower", 0.0)) if isinstance(cal_params, dict) else 0.0)

    gains          = np.array(gains)
    sa_corrections = np.array(sa_corrections)
    transmit_powers = np.array(transmit_powers)

    # ---- Variable 'day' ----
    day_arr = np.array(mat.get("Night1Sunrise2Day3Sunset4", np.ones(n_ping)), dtype=np.int8)

    # ---- ESU size ----
    esu_size = float(mat.get("sizeESU", mat.get("EsuSize", 0.1)))

    # ---- Construction du Dataset ----
    print("Construction du Dataset xarray ...")

    coords = {
        "channel":   ("channel", freqs_khz, {"units": "kHz", "long_name": "Acoustic frequency"}),
        "time":      ("time",    time_dt),
        "depth":     ("depth",   depth_axis, {"units": "m", "long_name": "Depth", "positive": "down"}),
        "latitude":  ("time",    lat, {"units": "degrees_north"}),
        "longitude": ("time",    lon, {"units": "degrees_east"}),
    }

    ds = xr.Dataset(coords=coords)

    # Sv
    ds["Sv"] = xr.DataArray(
        sv, dims=["channel", "time", "depth"],
        attrs={"units": "dB re 1 m-1", "long_name": "Volume backscattering strength"}
    )

    # Day/Night flag
    ds["day"] = xr.DataArray(
        day_arr, dims=["time"],
        attrs={"long_name": "Day/Night flag (1=Night, 2=Sunrise, 3=Day, 4=Sunset)",
               "flag_values": "1 2 3 4",
               "flag_meanings": "night sunrise day sunset"}
    )

    # Fréquence instrument
    ds["instrument_frequency"] = xr.DataArray(
        freqs_khz, dims=["channel"],
        attrs={"units": "kHz", "long_name": "Instrument frequency"}
    )

    # Chaînes de métadonnées instrument
    ds["instrument_transducer_location"]     = str_to_bytes_var("hull-mounted",              "STRING10_1",  n_freq)
    ds["instrument_transducer_manufacturer"] = str_to_bytes_var("Simrad",                    "STRING6_2",   n_freq)
    ds["instrument_transducer_model"]        = str_to_bytes_var("ES18-11/ES38B/ES70-7C",     "STRING23_10", n_freq)
    ds["instrument_sounder_manufacturer"]    = str_to_bytes_var("Kongsberg",                 "STRING9_4",   n_freq)

    # Paramètres de traitement
    ds["data_processing_on_axis_gain"] = xr.DataArray(
        gains, dims=["channel"],
        attrs={"units": "dB", "long_name": "On-axis gain (calibration)"}
    )
    ds["data_processing_on_axis_gain_units"] = str_to_bytes_var("dB", "STRING2_13", n_freq)
    ds["data_processing_Sacorrection"] = xr.DataArray(
        sa_corrections, dims=["channel"],
        attrs={"units": "dB", "long_name": "Sa correction"}
    )

    env = mat.get("UserParam", {}).get("InputFileRead", {}).get("Environment", {})
    sound_speed = float(env.get("SoundSpeedMs", 1492.0)) if isinstance(env, dict) else 1492.0
    ds["data_processing_soundspeed"] = xr.DataArray(
        np.full(n_freq, sound_speed), dims=["channel"],
        attrs={"units": "m s-1", "long_name": "Sound speed used in processing"}
    )
    ds["data_processing_absorption"] = xr.DataArray(
        np.full(n_freq, np.nan), dims=["channel"],
        attrs={"units": "dB m-1", "long_name": "Absorption coefficient"}
    )
    ds["data_processing_transducer_psi"] = xr.DataArray(
        np.full(n_freq, np.nan), dims=["channel"],
        attrs={"units": "sr", "long_name": "Equivalent two-way beam angle"}
    )

    ds["data_transmit_power"] = xr.DataArray(
        transmit_powers, dims=["channel"],
        attrs={"units": "W", "long_name": "Transmit power"}
    )

    transducer_depth = float(mat.get("TransducerDepth", 6.0))
    ds["instrument_transducer_depth"] = xr.DataArray(
        np.full(n_freq, transducer_depth), dims=["channel"],
        attrs={"units": "m", "long_name": "Transducer depth below waterline"}
    )

    ds["data_ping_interval"] = xr.DataArray(
        np.full(n_freq, esu_size), dims=["channel"],
        attrs={"units": "nmi", "long_name": "Ping integration interval (ESU size)"}
    )

    cum_dist = np.array(mat.get("CumulatedGPSDistanceMeter", np.zeros(n_ping)), dtype=np.float64)
    ds["distance"] = xr.DataArray(
        cum_dist, dims=["time"],
        attrs={"units": "m", "long_name": "Cumulated GPS distance"}
    )

    qc = np.array(mat.get("QcMean", np.ones((n_freq, n_ping))), dtype=np.float32)
    ds["data_quality"] = xr.DataArray(
        qc, dims=["channel", "time"],
        attrs={"long_name": "Mean quality index (0–1)"}
    )

    # ---- Attributs globaux (convention ICES AcMeta) ----
    processing_date = mat.get("EIParameters", {}).get("ProcessingDate",
                              datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    cruise_name  = mat.get("UserParam", {}).get("CruiseName", "Unknown")
    matecho_path = mat.get("UserParam", {}).get("MatechoPath", "")

    ds.attrs = {
        "convention_name":          "A metadata convention for processed acoustic data from active acoustic systems",
        "convention_author":        "ICES WGFAST Topic Group, TG-AcMeta",
        "convention_year":          "2016",
        "convention_organisation":  "International Council for the Sea (ICES)",
        "convention_publisher":     "The Series of ICES Survey Protocols",
        "convention_version":       "Version 1.10",
        "survey_name":              cruise_name,
        "survey_platform_name":     "Unknown",
        "survey_platform_type":     "Research vessel",
        "instrument_type":          "echo sounder",
        "instrument_model":         "Simrad EK60/EK80",
        "instrument_serial_number": "Unknown",
        "instrument_transducer_depth": str(transducer_depth),
        "data_type":                "Sv",
        "data_processing_software": f"Matecho ({matecho_path})",
        "data_processing_date":     str(processing_date),
        "data_ping_axis_interval_type":   "Distance (nmi)",
        "data_ping_axis_interval_origin": "Start",
        "data_ping_axis_interval_value":  str(esu_size),
        "data_range_axis_interval_type":   "Range (meters)",
        "data_range_axis_interval_origin": "Middle",
        "data_range_axis_interval_value":  str(res),
        "data_low_threshold":  str(float(mat.get("EILowThreshold", -100.0))),
        "data_high_threshold": "0.0",
        "time_convention": "UTC",
        "history":     f"Created {datetime.now(timezone.utc).isoformat()} from {Path(path_mat).name}",
        "Conventions": "CF-1.8 ICES-AcMeta-1.10",
    }

    # ---- Écriture NetCDF ----
    if path_out is None:
        path_out = Path(path_mat).with_suffix(".nc")

    encoding = {
        "Sv":   {"zlib": True, "complevel": 4, "dtype": "float32"},
        "time": {"units": "seconds since 1970-01-01", "calendar": "proleptic_gregorian", "dtype": "float64"},
    }

    print(f"Écriture vers {path_out} ...")
    ds.to_netcdf(path_out, encoding=encoding)
    print(f"✓ Terminé : {path_out}")
    print(ds)
    return ds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mat_to_netcdf.py <fichier.mat> [sortie.nc]")
        sys.exit(1)

    path_in  = sys.argv[1]
    path_out = sys.argv[2] if len(sys.argv) > 2 else None
    mat_to_netcdf(path_in, path_out)