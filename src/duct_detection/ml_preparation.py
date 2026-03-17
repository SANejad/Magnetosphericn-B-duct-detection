"""Feature preparation utilities for ML-ready duct-detection datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr


SCALAR_VARIABLES = ["Bt", "Density", "MLT", "MLat", "Dst", "Kp"]


def load_label_map(labels_csv: str | Path | None) -> dict[str, int]:
    """Load a label table with columns `file` and `label`."""
    if labels_csv is None:
        return {}
    labels_path = Path(labels_csv)
    if not labels_path.exists():
        return {}
    labels = pd.read_csv(labels_path)
    if not {"file", "label"}.issubset(labels.columns):
        raise ValueError("Label CSV must contain columns: file, label")
    return {str(row["file"]): int(row["label"]) for _, row in labels.iterrows()}


def _iter_event_datasets(sample_folder: str | Path) -> Iterable[tuple[Path, xr.Dataset]]:
    sample_path = Path(sample_folder)
    for path in sorted(sample_path.glob("*.nc")):
        yield path, xr.open_dataset(path, decode_times=True)


def _safe_stats(arr: np.ndarray, prefix: str) -> dict[str, float]:
    arr = np.asarray(arr, dtype=float)
    return {
        f"{prefix}_mean": float(np.nanmean(arr)),
        f"{prefix}_std": float(np.nanstd(arr)),
        f"{prefix}_min": float(np.nanmin(arr)),
        f"{prefix}_max": float(np.nanmax(arr)),
    }


def _epsd_band_features(ds: xr.Dataset, n_bands: int = 4) -> dict[str, float]:
    epsd = np.asarray(ds["EPSD"].values, dtype=float)
    if epsd.size == 0:
        return {f"EPSD_band{i+1}_mean": np.nan for i in range(n_bands)}

    n_freq = epsd.shape[1]
    bands = np.array_split(np.arange(n_freq), n_bands)
    features: dict[str, float] = {}
    for i, idx in enumerate(bands, start=1):
        if len(idx) == 0:
            features[f"EPSD_band{i}_mean"] = np.nan
        else:
            features[f"EPSD_band{i}_mean"] = float(np.nanmean(epsd[:, idx]))
    features.update(_safe_stats(epsd, "EPSD"))
    return features


def extract_rf_features(ds: xr.Dataset) -> dict[str, float]:
    """Summarize one event dataset into a tabular feature vector."""
    features: dict[str, float] = {}
    for var in SCALAR_VARIABLES:
        if var in ds:
            features.update(_safe_stats(ds[var].values, var))
    features.update(_epsd_band_features(ds))
    return features


def build_rf_training_arrays(
    sample_folder: str | Path,
    labels: dict[str, int] | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray, pd.DataFrame]:
    """Build a feature matrix for tabular models from event NetCDF files."""
    label_map = labels or {}
    rows: list[dict[str, object]] = []

    for path, ds in _iter_event_datasets(sample_folder):
        features = extract_rf_features(ds)
        row: dict[str, object] = {"file": path.name, **features, "label": label_map.get(path.name, np.nan)}
        rows.append(row)
        ds.close()

    if not rows:
        raise ValueError(f"No NetCDF event files found in {sample_folder}")

    metadata = pd.DataFrame(rows)
    feature_columns = [c for c in metadata.columns if c not in {"file", "label"}]
    X = metadata[feature_columns].to_numpy(dtype=float)
    y = metadata["label"].to_numpy(dtype=float)
    return feature_columns, X, y, metadata


def _resample_2d(image: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    """Resample a 2D array to a fixed shape using sequential 1D interpolation."""
    in_t, in_f = image.shape
    out_t, out_f = out_shape

    t_old = np.linspace(0.0, 1.0, in_t)
    t_new = np.linspace(0.0, 1.0, out_t)
    resized_t = np.vstack([np.interp(t_new, t_old, image[:, j]) for j in range(in_f)]).T

    f_old = np.linspace(0.0, 1.0, in_f)
    f_new = np.linspace(0.0, 1.0, out_f)
    resized = np.vstack([np.interp(f_new, f_old, resized_t[i, :]) for i in range(out_t)])
    return resized


def build_cnn_tensor_stack(
    sample_folder: str | Path,
    labels: dict[str, int] | None = None,
    out_shape: tuple[int, int] = (128, 64),
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Convert event spectrograms into fixed-size tensors for CNN/CNN-LSTM models."""
    label_map = labels or {}
    tensors: list[np.ndarray] = []
    rows: list[dict[str, object]] = []

    for path, ds in _iter_event_datasets(sample_folder):
        epsd = np.asarray(ds["EPSD"].values, dtype=float)
        if epsd.size == 0:
            ds.close()
            continue

        log_epsd = np.log10(np.clip(epsd, a_min=1e-12, a_max=None))
        image = _resample_2d(log_epsd, out_shape)
        image = image[..., np.newaxis]
        tensors.append(image.astype(np.float32))
        rows.append({"file": path.name, "label": label_map.get(path.name, np.nan)})
        ds.close()

    if not tensors:
        raise ValueError(f"No EPSD tensors were built from {sample_folder}")

    X = np.stack(tensors, axis=0)
    metadata = pd.DataFrame(rows)
    y = metadata["label"].to_numpy(dtype=float)
    return X, y, metadata


def save_ml_bundle(
    output_dir: str | Path,
    prefix: str,
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    feature_names: list[str] | None = None,
) -> None:
    """Save arrays and metadata for one ML-ready bundle."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    np.save(output_path / f"{prefix}_X.npy", X)
    np.save(output_path / f"{prefix}_y.npy", y)
    metadata.to_csv(output_path / f"{prefix}_metadata.csv", index=False)
    if feature_names is not None:
        pd.Series(feature_names, name="feature").to_csv(output_path / f"{prefix}_feature_names.csv", index=False)
