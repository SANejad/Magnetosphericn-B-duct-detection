"""MMS data ingestion and interpolation utilities for duct-detection workflows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import xarray as xr
import pyspedas
from pytplot import get_data


@dataclass(frozen=True)
class MMSInterval:
    t1: str
    t2: str

    @property
    def t1_utc(self) -> pd.Timestamp:
        return pd.to_datetime(self.t1).tz_localize("UTC")

    @property
    def t2_utc(self) -> pd.Timestamp:
        return pd.to_datetime(self.t2).tz_localize("UTC")

    @property
    def t1_ext(self) -> str:
        return (pd.to_datetime(self.t1) - timedelta(minutes=5)).strftime("%Y-%m-%d/%H:%M:%S")

    @property
    def t2_ext(self) -> str:
        return (pd.to_datetime(self.t2) + timedelta(minutes=5)).strftime("%Y-%m-%d/%H:%M:%S")

    @property
    def folder_name(self) -> str:
        t1_date = self.t1.split("/")[0]
        t1_fmt = self.t1.split("/")[1].replace(":", "_")
        t2_fmt = self.t2.split("/")[1].replace(":", "_")
        return f"data_{t1_date}_{t1_fmt}_to_{t2_fmt}"

    @property
    def stem(self) -> str:
        t1_date = self.t1.split("/")[0]
        t1_fmt = self.t1.split("/")[1].replace(":", "_")
        t2_fmt = self.t2.split("/")[1].replace(":", "_")
        return f"{t1_date}_{t1_fmt}_to_{t2_fmt}"


def convert_to_datetime(time_data) -> pd.DatetimeIndex:
    """Convert nanosecond epoch time to timezone-aware UTC datetimes."""
    return pd.to_datetime(time_data, unit="ns", utc=True)


def interpolate_to_common_time(
    original_time: pd.DatetimeIndex,
    original_data: np.ndarray,
    common_time: pd.DatetimeIndex,
) -> np.ndarray:
    """Interpolate 1D or 2D arrays to a common time grid."""
    common_sec = common_time.astype("int64") / 1e9
    original_sec = original_time.astype("int64") / 1e9
    arr = np.asarray(original_data)

    if arr.ndim == 1:
        return np.interp(common_sec, original_sec, arr)

    return np.column_stack(
        [np.interp(common_sec, original_sec, arr[:, i]) for i in range(arr.shape[1])]
    )


def load_mms_core_data(interval: MMSInterval, probe: int = 1) -> dict[str, xr.DataArray]:
    """Download core MMS products for one interval and one probe."""
    trange = [interval.t1_ext, interval.t2_ext]
    pyspedas.mms.fgm(probe=probe, trange=trange, time_clip=True)
    pyspedas.mms.fpi(
        probe=probe,
        datatype=["des-moms"],
        center_measurement=True,
        trange=trange,
        time_clip=True,
    )
    pyspedas.mms.mec(probe=probe, trange=trange, time_clip=True)
    pyspedas.mms.dsp(
        probe=probe,
        trange=trange,
        datatype=["epsd"],
        data_rate="fast",
        level="l2",
        time_clip=True,
    )

    keys = {
        "Bt": f"mms{probe}_fgm_b_gse_srvy_l2_btot",
        "B_vec": f"mms{probe}_fgm_b_gse_srvy_l2",
        "Density": f"mms{probe}_des_numberdensity_fast",
        "Position": f"mms{probe}_mec_r_gse",
        "MLat": f"mms{probe}_mec_mlat",
        "L_Dipole": f"mms{probe}_mec_l_dipole",
        "MLT": f"mms{probe}_mec_mlt",
        "Dst": f"mms{probe}_mec_dst",
        "Kp": f"mms{probe}_mec_kp",
        "EPSD": f"mms{probe}_dsp_epsd_x",
    }
    return {name: get_data(tp_name, xarray=True) for name, tp_name in keys.items()}


def build_interval_dataframe(interval: MMSInterval, probe: int = 1) -> pd.DataFrame:
    """Create a CSV-ready dataframe for one interval."""
    data = load_mms_core_data(interval, probe=probe)
    required = ("Bt", "B_vec", "Density", "Position")
    missing = [name for name in required if data.get(name) is None]
    if missing:
        raise ValueError(f"Missing required MMS products for probe {probe}: {missing}")

    common_time = convert_to_datetime(data["Bt"].time.data)
    mask = (common_time >= interval.t1_utc) & (common_time <= interval.t2_utc)
    filtered_time = common_time[mask]

    B_vec_interp = interpolate_to_common_time(
        convert_to_datetime(data["B_vec"].time.data), data["B_vec"].data, common_time
    )
    density_interp = interpolate_to_common_time(
        convert_to_datetime(data["Density"].time.data), data["Density"].data, common_time
    )
    pos_interp = interpolate_to_common_time(
        convert_to_datetime(data["Position"].time.data), data["Position"].data, common_time
    )

    scalar_fields = {}
    for name in ("MLat", "L_Dipole", "MLT", "Dst", "Kp"):
        da = data.get(name)
        if da is None:
            scalar_fields[name] = np.full(mask.sum(), np.nan)
            continue
        scalar_fields[name] = interpolate_to_common_time(
            convert_to_datetime(da.time.data), da.data, common_time
        )[mask]

    df = pd.DataFrame(
        {
            "Time": filtered_time.strftime("%Y-%m-%d/%H:%M:%S.%f"),
            "RelativeTime": (filtered_time - filtered_time[0]).total_seconds(),
            "Bt": data["Bt"].data[mask],
            "Bx": B_vec_interp[mask, 0],
            "By": B_vec_interp[mask, 1],
            "Bz": B_vec_interp[mask, 2],
            "Density": density_interp[mask],
            "Rx": pos_interp[mask, 0],
            "Ry": pos_interp[mask, 1],
            "Rz": pos_interp[mask, 2],
            "L_Dipole": scalar_fields["L_Dipole"],
            "MLT": scalar_fields["MLT"],
            "MLat": scalar_fields["MLat"],
            "Dst": scalar_fields["Dst"],
            "Kp": scalar_fields["Kp"],
        }
    )
    return df


def save_interval_csv(interval: MMSInterval, output_dir: str | Path, probe: int = 1) -> Path:
    """Build and save one CSV file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = build_interval_dataframe(interval, probe=probe)
    out_path = output_dir / f"mms{probe}_data_{interval.stem}.csv"
    df.to_csv(out_path, index=False)
    return out_path


def build_interval_dataset(
    interval: MMSInterval,
    probe: int = 1,
    epsd_min_hz: float = 50.0,
    epsd_max_hz: float = 800.0,
) -> xr.Dataset:
    """Create a NetCDF-ready dataset with interpolated scalar/vector data and EPSD."""
    data = load_mms_core_data(interval, probe=probe)
    required = ("Bt", "B_vec", "Density", "Position")
    missing = [name for name in required if data.get(name) is None]
    if missing:
        raise ValueError(f"Missing required MMS products for probe {probe}: {missing}")

    common_time = convert_to_datetime(data["Bt"].time.data)
    mask = (common_time >= interval.t1_utc) & (common_time <= interval.t2_utc)
    filtered_time = common_time[mask]
    rel_time_sec = (filtered_time - filtered_time[0]).total_seconds()

    B_vec_interp = interpolate_to_common_time(
        convert_to_datetime(data["B_vec"].time.data), data["B_vec"].data, common_time
    )
    density_interp = interpolate_to_common_time(
        convert_to_datetime(data["Density"].time.data), data["Density"].data, common_time
    )
    pos_interp = interpolate_to_common_time(
        convert_to_datetime(data["Position"].time.data), data["Position"].data, common_time
    )

    scalar_fields = {}
    for name in ("MLat", "L_Dipole", "MLT", "Dst", "Kp"):
        da = data.get(name)
        if da is None:
            scalar_fields[name] = np.full(mask.sum(), np.nan)
            continue
        scalar_fields[name] = interpolate_to_common_time(
            convert_to_datetime(da.time.data), da.data, common_time
        )[mask]

    epsd_freq = np.array([])
    epsd_interp = np.empty((mask.sum(), 0))
    epsd_da = data.get("EPSD")
    if epsd_da is not None:
        epsd_time = convert_to_datetime(epsd_da.time.data)
        epsd_freq = epsd_da.v.data
        epsd_power = epsd_da.data
        full_epsd_interp = np.empty((len(common_time), len(epsd_freq)))
        for i in range(len(epsd_freq)):
            full_epsd_interp[:, i] = np.interp(
                common_time.astype("int64") / 1e9,
                epsd_time.astype("int64") / 1e9,
                epsd_power[:, i],
            )
        freq_mask = (epsd_freq >= epsd_min_hz) & (epsd_freq <= epsd_max_hz)
        epsd_freq = epsd_freq[freq_mask]
        epsd_interp = full_epsd_interp[mask][:, freq_mask]

    ds = xr.Dataset(
        data_vars=dict(
            Bt=("time", data["Bt"].data[mask]),
            Bx=("time", B_vec_interp[mask, 0]),
            By=("time", B_vec_interp[mask, 1]),
            Bz=("time", B_vec_interp[mask, 2]),
            Density=("time", density_interp[mask]),
            Rx=("time", pos_interp[mask, 0]),
            Ry=("time", pos_interp[mask, 1]),
            Rz=("time", pos_interp[mask, 2]),
            L_Dipole=("time", scalar_fields["L_Dipole"]),
            MLT=("time", scalar_fields["MLT"]),
            MLat=("time", scalar_fields["MLat"]),
            Dst=("time", scalar_fields["Dst"]),
            Kp=("time", scalar_fields["Kp"]),
            EPSD=(["time", "frequency"], epsd_interp),
        ),
        coords=dict(
            time=("time", filtered_time),
            frequency=("frequency", epsd_freq),
            RelativeTime=("time", rel_time_sec),
        ),
        attrs=dict(
            description=f"MMS{probe} interpolated dataset with EPSD",
            source="NASA MMS via pySPEDAS",
        ),
    )
    return ds


def save_interval_netcdf(interval: MMSInterval, output_dir: str | Path, probe: int = 1) -> Path:
    """Build and save one NetCDF file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ds = build_interval_dataset(interval, probe=probe)
    out_path = output_dir / f"mms{probe}_multidimensional_{interval.stem}.nc"
    ds.to_netcdf(out_path)
    return out_path


def load_intervals_from_csv(csv_path: str | Path) -> list[MMSInterval]:
    """Load T1/T2 intervals from a CSV file."""
    df = pd.read_csv(csv_path)
    return [MMSInterval(t1=row["T1"], t2=row["T2"]) for _, row in df.iterrows()]
