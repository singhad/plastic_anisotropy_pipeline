from typing import List, Tuple
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial import cKDTree

from utils import get_base_path, setup_repo_structure, setup_logging, get_artifact_dirs
from data_loader import read_dataset, clean_data
from saving import save_interpolated_grid_versioned, save_metrics_records
from plotting import plot_property_from_grid


# ---------------- core interpolation ----------------

def interpolate_property_nearest(
    df_clean: pd.DataFrame,
    property_name: str,
    grid_size: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Piecewise-constant interpolation using NearestNDInterpolator.
    """
    x = df_clean["Time"].to_numpy()
    y = df_clean["Temperature"].to_numpy()
    z = df_clean[property_name].to_numpy()

    xi = np.linspace(float(x.min()), float(x.max()), grid_size)
    yi = np.linspace(float(y.min()), float(y.max()), grid_size)
    grid_x, grid_y = np.meshgrid(xi, yi)

    interp = NearestNDInterpolator(list(zip(x, y)), z)
    grid_z = interp(grid_x, grid_y)
    return grid_x, grid_y, grid_z


# ---------------- uncertainty (proximity) ----------------

def _proximity_sigma_grid(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    samples_xy: np.ndarray,
    k: int = 3,
    normalize: bool = True
) -> np.ndarray:
    """
    Distance-to-data heuristic:
    sigma := mean distance to k nearest samples (optionally normalized by median).
    """
    tree = cKDTree(samples_xy)
    q = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    dists, _ = tree.query(q, k=min(k, len(samples_xy)))
    dmean = dists if dists.ndim == 1 else dists.mean(axis=1)
    sigma = dmean.reshape(grid_x.shape)
    if normalize:
        med = float(np.median(sigma))
        if med > 0:
            sigma = sigma / med
    return sigma


# ---------------- benchmarks ----------------

def _fit_and_score_nearest(
    df_clean: pd.DataFrame,
    property_name: str
) -> Tuple[float, float, float]:
    """
    In-sample metrics at sample points using the same nearest interpolant.
    """
    x = df_clean["Time"].to_numpy()
    y = df_clean["Temperature"].to_numpy()
    z = df_clean[property_name].to_numpy()

    interp = NearestNDInterpolator(list(zip(x, y)), z)
    zhat = interp(x, y)

    resid = zhat - z
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae  = float(np.mean(np.abs(resid)))
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((z - np.mean(z))**2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return rmse, mae, r2


# ---------------- pipeline ----------------

def execute_nearest_pipeline(
    properties: List[str] = ("Rp0.2", "Rm", "At", "HV", "E", "v", "Z", "A32", "R_bar", "delta_R"),
    grid_size: int = 100,
    y_strain: float = 1.5,
    compute_uncertainty: bool = False,
    uncertainty_k: int = 3,
    benchmark: bool = True
) -> None:
    """
    Run nearest-neighbour interpolation for each property and save:
      - grids:  outputs/nearest/grids/<prop>_nearest_grid.csv
      - plots:  plots/nearest/plots/<prop>_nearest_plot.png
    Optional:
      - proximity sigma: <prop>_sigma_nearest_grid.csv + plot
      - in-sample metrics: RMSE/MAE/R2 (logged)
    """
    base_path = get_base_path()
    setup_repo_structure(base_path)

    technique = "nearest"
    grids_dir, metrics_dir, plots_dir = get_artifact_dirs(base_path, technique)

    log_dir = base_path / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_{technique}_{timestamp}.log"
    logger = setup_logging(log_file)
    logger.info("NEAREST interpolation pipeline started.")

    metrics_rows = []

    try:
        datasets_path = base_path / "datasets"
        df = read_dataset(datasets_path / "all_mechanical_properties.csv")
        df_r_values = read_dataset(datasets_path / "all_r_values.csv")
        df_clean = clean_data(df, df_r_values, y_strain)

        for prop in properties:
            logger.info(f"Interpolating property: {prop}")
            gx, gy, gz = interpolate_property_nearest(df_clean, prop, grid_size=grid_size)

            # Save prediction grid + plot
            save_interpolated_grid_versioned(grids_dir, prop, technique, gx, gy, gz)
            plot_property_from_grid(
                input_dir=grids_dir,
                df_clean=df_clean,
                property_name=prop,
                technique=technique,
                save_path=plots_dir
            )
            logger.info(f"Saved grid and plot for {prop}")

            # Benchmarks (in-sample)
            if benchmark:
                rmse, mae, r2 = _fit_and_score_nearest(df_clean, prop)
                metrics_rows.append({
                    "technique": technique,
                    "property": prop,
                    "grid_size": grid_size,
                    "n_samples": int(len(df_clean)),
                    "rmse": rmse, "mae": mae, "r2": r2
                })
                logger.info(f"[{technique}] {prop}: RMSE={rmse:.4g} MAE={mae:.4g} R2={r2:.4g}")

            # Proximity-based uncertainty (sigma)
            if compute_uncertainty:
                samples_xy = df_clean[["Time", "Temperature"]].to_numpy()
                sigma = _proximity_sigma_grid(gx, gy, samples_xy, k=uncertainty_k)
                sprop = f"{prop}_sigma"

                save_interpolated_grid_versioned(grids_dir, sprop, technique, gx, gy, sigma)
                plot_property_from_grid(
                    input_dir=grids_dir,
                    df_clean=df_clean,
                    property_name=sprop,
                    technique=technique,
                    save_path=plots_dir
                )
                logger.info(f"Saved proximity-sigma grid and plot for {prop}")

            if benchmark and metrics_rows:
                save_metrics_records(metrics_dir, technique, metrics_rows, timestamp)
                logger.info(f"Saved metrics CSV with {len(metrics_rows)} rows to {metrics_dir}")

        logger.info("NEAREST pipeline completed.")

    except Exception as e:
        logger.exception(f"Nearest pipeline failed: {e}")
        raise
