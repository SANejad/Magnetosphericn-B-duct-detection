"""Plotting utilities for magnetospheric duct detection quicklooks."""

from __future__ import annotations

import os
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr


def plot_bt_epsd(ds: xr.Dataset, file_name: str, output_folder: str) -> str:
    """Plot Bt over an EPSD spectrogram and save as a quicklook image."""
    os.makedirs(output_folder, exist_ok=True)

    bt = ds["Bt"].values
    epsd = ds["EPSD"].values
    freq = ds["frequency"].values
    time = pd.to_datetime(ds["time"].values)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    pcm = ax1.pcolormesh(time, freq, epsd.T, shading="auto", cmap="jet")
    ax1.set_ylabel("Frequency (Hz)")
    fig.colorbar(pcm, ax=ax1, label="EPSD")

    ax2 = ax1.twinx()
    ax2.plot(time, bt, color="white", linewidth=2)
    ax2.set_ylabel("Bt (nT)", color="black")
    ax2.tick_params(axis="y", labelcolor="black")
    ax2.spines["right"].set_color("black")

    ax1.set_title(f"Bt over EPSD: {file_name}")
    plt.tight_layout()

    output_path = os.path.join(output_folder, file_name.replace(".nc", ".jpg"))
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def batch_plot_event_quicklooks(input_folder: str, output_folder: str) -> list[str]:
    """Generate quicklook plots for all NetCDF event files in a folder."""
    os.makedirs(output_folder, exist_ok=True)
    plotted_files: list[str] = []

    for filename in os.listdir(input_folder):
        if not filename.endswith(".nc"):
            continue

        file_path = os.path.join(input_folder, filename)
        try:
            ds = xr.open_dataset(file_path, decode_times=True)
            output_file = plot_bt_epsd(ds, filename, output_folder)
            plotted_files.append(output_file)
        except Exception as exc:
            print(f"Error processing {filename}: {exc}")

    return plotted_files
