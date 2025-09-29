from typing import List, Tuple
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.interpolate import Rbf
from scipy.spatial import cKDTree

from utils import get_base_path, setup_repo_structure, setup_logging, get_artifact_dirs
from data_loader import read_dataset, clean_data
from saving import save_interpolated_grid_versioned, save_metrics_records
from plotting import plot_property_from_grid


# ---------------- core interpolation ----------------

def interpolate_property_rbf_compact(
    df_clean: pd.DataFrame,
    property_name: str,
    function_type: str = "gaussian",
    epsilon: float = 1.0,
    smooth: float = 0.01,
    grid_size: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    'Compact-ish' RBF surface. SciPy's Rbf is global, but smaller epsilon and
    small smooth often produce more local behaviour resembling compact support.
    """
    x = df_clean["Time"].to_numpy()
    y = df_clean["Temperature"].to_numpy()
    z = df_clean[property_name].to_numpy()

    xi = np.linspace(float(x.min()), float(x.max()), grid_size)
    yi = np.linspace(float(y.min()), float(y.max()), grid_size)
    grid_x, grid_y = np.meshgrid(xi, yi)

    rbf = Rbf(x, y, z, function=function_type, epsilon=float(epsilon), smooth=float(smooth))
    grid_z = rbf(grid_x, grid_y)
    return grid_x, grid_y, grid_z


# ---------------- uncertainty (proximity) ----------------

def _proximity_sigma_grid(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    samples_xy: np.ndarray,
    k: int = 3,
    normalize: bool = True,
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

def _fit_and_score_rbf_compact(
    df_clean: pd.DataFrame, property_name: str, function_type: str, epsilon: float, smooth: float
) -> Tuple[float, float, float]:
    """
    In-sample metrics at sample points using the same RBF configuration.
    """
    x = df_clean["Time"].to_numpy()
    y = df_clean["Temperature"].to_numpy()
    z = df_clean[property_name].to_numpy()

    rbf = Rbf(x, y, z, function=function_type, epsilon=float(epsilon), smooth=float(smooth))
    zhat = rbf(x, y)

    resid = zhat - z
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae  = float(np.mean(np.abs(resid)))
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((z - np.mean(z))**2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return rmse, mae, r2


# ---------------- pipeline ----------------

def execute_rbf_compact_pipeline(
    properties: List[str] = ("Rp0.2", "Rm", "At", "HV", "E", "v", "Z", "A32", "R_bar", "delta_R"),
    grid_size: int = 100,
    y_strain: float = 1.5,
    function_type: str = "gaussian",
    epsilon: float = 1.0,
    smooth: float = 0.01,
    compute_uncertainty: bool = False,
    uncertainty_k: int = 3,
    benchmark: bool = True,
) -> None:
    """
    Run compact-ish RBF interpolation for each property.

    Saves:
      grids → outputs/rbf_compact/grids/<prop>_rbf_compact_<kernel>_s<smooth>_grid.csv
      plots → plots/rbf_compact/plots/<prop>_rbf_compact_<kernel>_s<smooth>_plot.png

    Optional:
      proximity sigma → <prop>_sigma_rbf_compact_<kernel>_s<smooth>_grid.csv + plot
      in-sample metrics (RMSE/MAE/R2) logged per property.
    """
    base_path = get_base_path()
    setup_repo_structure(base_path)

    technique = "rbf_compact"
    grids_dir, metrics_dir, plots_dir = get_artifact_dirs(base_path, technique)

    log_dir = base_path / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_{technique}_{timestamp}.log"
    logger = setup_logging(log_file)
    logger.info("RBF (compact-ish) interpolation pipeline started.")

    metrics_rows = []

    try:
        datasets_path = base_path / "datasets"
        df = read_dataset(datasets_path / "all_mechanical_properties.csv")
        df_r_values = read_dataset(datasets_path / "all_r_values.csv")
        df_clean = clean_data(df, df_r_values, y_strain)

        for prop in properties:
            logger.info(
                f"Interpolating property: {prop} "
                f"(kernel={function_type}, epsilon={epsilon}, smooth={smooth})"
            )
            gx, gy, gz = interpolate_property_rbf_compact(
                df_clean, prop,
                function_type=function_type, epsilon=epsilon, smooth=smooth,
                grid_size=grid_size
            )

            # Save prediction grid + plot
            save_interpolated_grid_versioned(
                grids_dir, prop, technique, gx, gy, gz,
                kernel=function_type, smooth=smooth
            )
            plot_property_from_grid(
                input_dir=grids_dir,
                df_clean=df_clean,
                property_name=prop,
                technique=technique,
                kernel=function_type,
                smooth=smooth,
                save_path=plots_dir,
            )
            logger.info(f"Saved grid and plot for {prop}")

            # Benchmarks (in-sample)
            if benchmark:
                rmse, mae, r2 = _fit_and_score_rbf_compact(
                    df_clean, prop, function_type, float(epsilon), float(smooth)
                )
                metrics_rows.append({
                    "technique": technique,
                    "property": prop,
                    "grid_size": grid_size,
                    "n_samples": int(len(df_clean)),
                    "rmse": rmse, "mae": mae, "r2": r2,
                    "kernel": function_type, "epsilon": float(epsilon), "smooth": float(smooth)
                })
                logger.info(
                    f"[{technique}] {prop} (kernel={function_type}, epsilon={epsilon}, smooth={smooth}): "
                    f"RMSE={rmse:.4g} MAE={mae:.4g} R2={r2:.4g}"
                )

            # Proximity-based uncertainty (sigma)
            if compute_uncertainty:
                samples_xy = df_clean[["Time", "Temperature"]].to_numpy()
                sigma = _proximity_sigma_grid(gx, gy, samples_xy, k=uncertainty_k)
                sprop = f"{prop}_sigma"

                save_interpolated_grid_versioned(
                    grids_dir, sprop, technique, gx, gy, sigma,
                    kernel=function_type, smooth=smooth
                )
                plot_property_from_grid(
                    input_dir=grids_dir,
                    df_clean=df_clean,
                    property_name=sprop,
                    technique=technique,
                    kernel=function_type,
                    smooth=smooth,
                    save_path=plots_dir,
                )
                logger.info(f"Saved proximity-sigma grid and plot for {prop}")

            if benchmark and metrics_rows:
                save_metrics_records(metrics_dir, technique, metrics_rows, timestamp)
                logger.info(f"Saved metrics CSV with {len(metrics_rows)} rows to {metrics_dir}")

        logger.info("RBF_COMPACT pipeline completed.")

    except Exception as e:
        logger.exception(f"RBF compact pipeline failed: {e}")
        raise
