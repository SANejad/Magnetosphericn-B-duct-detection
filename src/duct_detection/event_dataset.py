"""Event dataset construction utilities for magnetospheric duct detection.

This module downloads MMS variables for event intervals, interpolates them to a
common time base, packages them into event-centered xarray Datasets, and saves
one NetCDF sample per interval.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import xarray as xr

import pyspedas
from pytplot import get_data


@dataclass
class EventConfig:
    """Configuration for event dataset generation."""

    probes: tuple[int, ...] = (1,)
    padding_minutes: int = 5
    output_folder: str = "samples"
    epsd_min_hz: float = 50.0
    epsd_max_hz: float = 800.0


CORE_VARIABLES = {
    "Bt": "fgm_b_gse_srvy_l2_btot",
    "B_vec": "fgm_b_gse_srvy_l2",
    "Density": "des_numberdensity_fast",
    "Position": "mec_r_gse",
    "MLat": "mec_mlat",
    "L_Dipole": "mec_l_dipole",
    "MLT": "mec_mlt",
    "Dst": "mec_dst",
    "Kp": "mec_kp",
    "EPSD": "dsp_epsd_x",
}


def convert_to_datetime(time_data: np.ndarray) -> pd.DatetimeIndex:
    """Convert spacecraft time arrays to timezone-aware pandas datetimes."""
    return pd.to_datetime(time_data, unit="ns", utc=True)


def interpolate_to_common_time(
    original_time: pd.DatetimeIndex,
    original_data: np.ndarray,
    common_time: pd.DatetimeIndex,
) -> np.ndarray:
    """Interpolate 1D or 2D arrays onto a common time grid."""
    common_sec = common_time.astype("int64") / 1e9
    original_sec = original_time.astype("int64") / 1e9

    if original_data.ndim == 1:
        return np.interp(common_sec, original_sec, original_data)

    return np.array(
        [np.interp(common_sec, original_sec, original_data[:, i]) for i in range(original_data.shape[1])]
    ).T


def _load_mms_data(probe: int, start_ext: str, end_ext: str) -> dict[str, Optional[xr.DataArray]]:
    """Download and retrieve MMS variables for a single probe and interval."""
    pyspedas.mms.fgm(probe=probe, trange=[start_ext, end_ext], time_clip=True)
    pyspedas.mms.fpi(
        probe=probe,
        datatype=["des-moms"],
        center_measurement=True,
        trange=[start_ext, end_ext],
        time_clip=True,
    )
    pyspedas.mms.mec(probe=probe, trange=[start_ext, end_ext], time_clip=True)
    pyspedas.mms.dsp(
        probe=probe,
        trange=[start_ext, end_ext],
        datatype=["epsd"],
        data_rate="fast",
        level="l2",
        time_clip=True,
    )

    return {
        "Bt": get_data(f"mms{probe}_{CORE_VARIABLES['Bt']}", xarray=True),
        "B_vec": get_data(f"mms{probe}_{CORE_VARIABLES['B_vec']}", xarray=True),
        "Density": get_data(f"mms{probe}_{CORE_VARIABLES['Density']}", xarray=True),
        "Position": get_data(f"mms{probe}_{CORE_VARIABLES['Position']}", xarray=True),
        "MLat": get_data(f"mms{probe}_{CORE_VARIABLES['MLat']}", xarray=True),
        "L_Dipole": get_data(f"mms{probe}_{CORE_VARIABLES['L_Dipole']}", xarray=True),
        "MLT": get_data(f"mms{probe}_{CORE_VARIABLES['MLT']}", xarray=True),
        "Dst": get_data(f"mms{probe}_{CORE_VARIABLES['Dst']}", xarray=True),
        "Kp": get_data(f"mms{probe}_{CORE_VARIABLES['Kp']}", xarray=True),
        "EPSD": get_data(f"mms{probe}_{CORE_VARIABLES['EPSD']}", xarray=True),
    }


def _interpolate_epsd(epsd: Optional[xr.DataArray], common_time: pd.DatetimeIndex, fmin: float, fmax: float) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate EPSD spectrogram onto the common time grid and band-limit frequencies."""
    if epsd is None:
        return np.array([]), np.empty((len(common_time), 0))

    epsd_time = convert_to_datetime(epsd.time.data)
    epsd_freq = epsd.v.data
    epsd_power = epsd.data

    interp = np.empty((len(common_time), len(epsd_freq)))
    common_sec = common_time.astype("int64") / 1e9
    epsd_sec = epsd_time.astype("int64") / 1e9

    for i in range(len(epsd_freq)):
        interp[:, i] = np.interp(common_sec, epsd_sec, epsd_power[:, i])

    mask = (epsd_freq >= fmin) & (epsd_freq <= fmax)
    return epsd_freq[mask], interp[:, mask]


def build_event_dataset(
    t1: str,
    t2: str,
    probe: int = 1,
    config: Optional[EventConfig] = None,
) -> xr.Dataset:
    """Build one event-centered dataset for a single MMS probe and interval."""
    cfg = config or EventConfig()

    t1_ts = pd.to_datetime(t1)
    t2_ts = pd.to_datetime(t2)
    t1_ext = (t1_ts - timedelta(minutes=cfg.padding_minutes)).strftime("%Y-%m-%d/%H:%M:%S")
    t2_ext = (t2_ts + timedelta(minutes=cfg.padding_minutes)).strftime("%Y-%m-%d/%H:%M:%S")

    data = _load_mms_data(probe, t1_ext, t2_ext)
    if any(data[key] is None for key in ["Bt", "B_vec", "Density", "Position"]):
        raise ValueError(f"Missing core data for MMS{probe} during interval {t1} to {t2}.")

    common_time = convert_to_datetime(data["Bt"].time.data)
    t1_aware = t1_ts.tz_localize("UTC")
    t2_aware = t2_ts.tz_localize("UTC")

    b_vec_interp = interpolate_to_common_time(convert_to_datetime(data["B_vec"].time.data), data["B_vec"].data, common_time)
    density_interp = interpolate_to_common_time(convert_to_datetime(data["Density"].time.data), data["Density"].data, common_time)
    pos_interp = interpolate_to_common_time(convert_to_datetime(data["Position"].time.data), data["Position"].data, common_time)
    mlat_interp = interpolate_to_common_time(convert_to_datetime(data["MLat"].time.data), data["MLat"].data, common_time)
    l_dipole_interp = interpolate_to_common_time(convert_to_datetime(data["L_Dipole"].time.data), data["L_Dipole"].data, common_time)
    mlt_interp = interpolate_to_common_time(convert_to_datetime(data["MLT"].time.data), data["MLT"].data, common_time)
    dst_interp = interpolate_to_common_time(convert_to_datetime(data["Dst"].time.data), data["Dst"].data, common_time)
    kp_interp = interpolate_to_common_time(convert_to_datetime(data["Kp"].time.data), data["Kp"].data, common_time)
    epsd_freq, epsd_interp = _interpolate_epsd(data["EPSD"], common_time, cfg.epsd_min_hz, cfg.epsd_max_hz)

    mask = (common_time >= t1_aware) & (common_time <= t2_aware)
    filtered_time = common_time[mask]
    rel_time_sec = (filtered_time - filtered_time[0]).total_seconds()

    return xr.Dataset(
        data_vars=dict(
            Bt=("time", data["Bt"].data[mask]),
            Bx=("time", b_vec_interp[mask, 0]),
            By=("time", b_vec_interp[mask, 1]),
            Bz=("time", b_vec_interp[mask, 2]),
            Density=("time", density_interp[mask]),
            Rx=("time", pos_interp[mask, 0]),
            Ry=("time", pos_interp[mask, 1]),
            Rz=("time", pos_interp[mask, 2]),
            L_Dipole=("time", l_dipole_interp[mask]),
            MLT=("time", mlt_interp[mask]),
            MLat=("time", mlat_interp[mask]),
            Dst=("time", dst_interp[mask]),
            Kp=("time", kp_interp[mask]),
            EPSD=(("time", "frequency"), epsd_interp[mask]),
        ),
        coords=dict(
            time=("time", filtered_time),
            frequency=("frequency", epsd_freq),
            RelativeTime=("time", rel_time_sec),
        ),
        attrs=dict(
            description=f"MMS{probe} event dataset including EPSD for duct analysis",
            source="NASA MMS via pySPEDAS",
            probe=f"MMS{probe}",
            start_time=t1,
            end_time=t2,
        ),
    )


def save_event_dataset(ds: xr.Dataset, output_path: str) -> None:
    """Save one event dataset to NetCDF."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ds.to_netcdf(output_path)


def export_event_datasets_from_csv(
    interval_csv: str,
    probes: Iterable[int] = (1,),
    config: Optional[EventConfig] = None,
) -> list[str]:
    """Export one NetCDF file per event interval listed in a CSV."""
    cfg = config or EventConfig()
    os.makedirs(cfg.output_folder, exist_ok=True)

    intervals = pd.read_csv(interval_csv)
    outputs: list[str] = []

    for _, row in intervals.iterrows():
        t1 = row["T1"]
        t2 = row["T2"]
        t1_date = t1.split("/")[0]
        t1_fmt = t1.split("/")[1].replace(":", "_")
        t2_fmt = t2.split("/")[1].replace(":", "_")

        for probe in probes:
            ds = build_event_dataset(t1=t1, t2=t2, probe=probe, config=cfg)
            out_file = os.path.join(
                cfg.output_folder,
                f"mms{probe}_multidimensional_{t1_date}_{t1_fmt}_to_{t2_fmt}.nc",
            )
            save_event_dataset(ds, out_file)
            outputs.append(out_file)
            print(f"Saved NetCDF: {out_file}")

    return outputs
