from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from datetime import datetime


# ──────────────────────────────────────────────────────────────────────────────
# Filename suffix builder (kept consistent with plotting.py)
# ──────────────────────────────────────────────────────────────────────────────

def _build_suffix(
    technique: str,
    kernel: Optional[str] = None,
    smooth: Optional[float] = None,
    variogram_model: Optional[str] = None,
    nugget: Optional[float] = None,
) -> str:
    """
    Mirror the filename tags used across the project:
      - rbf / rbf_compact:  _{kernel}_s{smooth}
      - kriging:            _{variogram_model}_s{nugget}
      - others:             ""
    """
    s = ""
    tech = (technique or "").lower()
    if tech in {"rbf", "rbf_compact"}:
        if kernel:
            s += f"_{kernel}"
        if smooth is not None:
            s += f"_s{smooth}"
    elif tech == "kriging":
        if variogram_model:
            s += f"_{variogram_model}"
        if nugget is not None:
            s += f"_s{nugget}"
    return s


# ──────────────────────────────────────────────────────────────────────────────
# Core saver
# ──────────────────────────────────────────────────────────────────────────────

def _grid_to_long_df(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    value_name: str,
) -> pd.DataFrame:
    """
    Convert 2D meshgrids (X, Y, Z) into a long-form DataFrame with columns:
      Time, Temperature, <value_name>
    """
    # Ensure float arrays (avoids dtype surprises on CSV roundtrip)
    gx = np.asarray(grid_x, dtype=float)
    gy = np.asarray(grid_y, dtype=float)
    gz = np.asarray(grid_z, dtype=float)

    df = pd.DataFrame(
        {
            "Time": gx.ravel(order="C"),
            "Temperature": gy.ravel(order="C"),
            value_name: gz.ravel(order="C"),
        }
    )

    # Sort rows primarily by Temperature then by Time so pivoting is stable
    df.sort_values(["Temperature", "Time"], inplace=True, kind="mergesort", ignore_index=True)
    return df


def save_interpolated_grid_versioned(
    output_dir: Path,
    property_name: str,
    technique: str,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    *,
    kernel: Optional[str] = None,
    smooth: Optional[float] = None,
    variogram_model: Optional[str] = None,
    nugget: Optional[float] = None,
) -> Path:
    """
    Save a long-form grid CSV with a canonical filename:

        <property>_<technique><suffix>_grid.csv

    where <suffix> follows the convention in _build_suffix(). This function
    intentionally writes to the **canonical** name (no numeric versions)
    so plotting code can reliably find the latest artifact.

    Parameters
    ----------
    output_dir : Path
        Directory where the CSV will be written (e.g., outputs/<technique>/grids).
    property_name : str
        Column name for the Z values, e.g., 'Rp0.2' or 'Rp0.2_sigma'.
    technique : str
        Technique tag used in filenames (akima, rbf, kriging, linear, nearest, etc.).

    Returns
    -------
    Path
        The full path to the saved CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = _build_suffix(
        technique,
        kernel=kernel,
        smooth=smooth,
        variogram_model=variogram_model,
        nugget=nugget,
    )
    out_csv = output_dir / f"{property_name}_{technique}{suffix}_grid.csv"

    df = _grid_to_long_df(grid_x, grid_y, grid_z, value_name=property_name)

    # Attach non-numeric metadata columns (harmless to loaders; useful for audit)
    meta = {
        "technique": technique,
        "kernel": kernel if kernel is not None else "",
        "smooth": str(smooth) if smooth is not None else "",
        "variogram_model": variogram_model if variogram_model is not None else "",
        "nugget": str(nugget) if nugget is not None else "",
    }
    # Repeat metadata once per row (strings → not picked up as numeric by plotting)
    for k, v in meta.items():
        df[k] = v

    df.to_csv(out_csv, index=False)
    return out_csv


__all__ = [
    "save_interpolated_grid_versioned",
]

# ──────────────────────────────────────────────────────────────────────────────
# Metrics saver
# ──────────────────────────────────────────────────────────────────────────────

def save_metrics_records(
    metrics_dir: Path,
    technique: str,
    rows: List[Dict[str, Any]],
    timestamp: Optional[str] = None,
) -> Path:
    """
    Save per-technique metrics as a timestamped CSV:
      outputs/<technique>/metrics/metrics_<technique>_<YYYYmmdd_HHMMSS>.csv
    """
    if not rows:
        raise ValueError("No metric rows to save.")
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = metrics_dir / f"metrics_{technique}_{ts}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv
