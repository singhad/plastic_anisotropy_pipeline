from typing import List, Tuple, Optional
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from scipy.interpolate import Rbf
from scipy.spatial import cKDTree

from utils import get_base_path, setup_repo_structure, setup_logging, get_artifact_dirs
from data_loader import read_dataset, clean_data
from saving import save_interpolated_grid_versioned, save_metrics_records
from plotting import plot_property_from_grid


# ---------------- utilities for stability ----------------

def _dedupe_xy_mean(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Collapse duplicate (Time, Temperature) rows by taking the mean of the value column.
    Keeps numeric stability high and avoids contradictory duplicates.
    """
    cols = ["Time", "Temperature", value_col]
    if not set(cols).issubset(df.columns):
        raise ValueError(f"DataFrame missing required columns: {cols}")
    agg = (df[cols]
           .groupby(["Time", "Temperature"], as_index=False, sort=False)[value_col]
           .mean())
    return agg

def _zscore(arr: np.ndarray) -> Tuple[np.ndarray, float, float]:
    mu = float(np.mean(arr))
    sigma = float(np.std(arr)) or 1.0
    return (arr - mu) / sigma, mu, sigma


# ---------------- core interpolation ----------------

def _fit_rbf_robust(
    x_s: np.ndarray,
    y_s: np.ndarray,
    z: np.ndarray,
    kernel: str,
    smooth: float,
    retry_smooths: Optional[List[float]] = None,
) -> Tuple[Rbf, float]:
    """
    Try to fit an RBF; if singular/ill-conditioned, retry with larger smooth values.
    Returns the fitted Rbf and the *effective* smooth used.
    """
    if retry_smooths is None:
        # Start with the requested smooth (even if 0.0), then escalate
        retry_smooths = [smooth, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

    last_err = None
    for s in retry_smooths:
        try:
            model = Rbf(x_s, y_s, z, function=kernel, smooth=float(s))
            # quick sanity check (build matrix happens in __init__)
            return model, float(s)
        except LinAlgError as e:
            last_err = e
            continue
        except Exception as e:
            last_err = e
            continue
    # If we get here, exhaustively failed
    raise RuntimeError(f"RBF failed to fit after retries; last error: {last_err}")

def interpolate_property_rbf(
    df_clean: pd.DataFrame,
    property_name: str,
    kernel: str = "linear",
    smooth: float = 0.0,
    grid_size: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Standard RBF surface with robust fitting:
      - dedupe (Time, Temperature)
      - z-score scale X,Y
      - escalate smooth if needed
    Returns: grid_x, grid_y, grid_z, smooth_eff
    """
    # Collapse duplicates to prevent contradictions / singularities
    df_prop = _dedupe_xy_mean(df_clean, property_name)

    x = df_prop["Time"].to_numpy(dtype=float)
    y = df_prop["Temperature"].to_numpy(dtype=float)
    z = df_prop[property_name].to_numpy(dtype=float)

    # Scale for conditioning
    x_s, x_mu, x_sd = _zscore(x)
    y_s, y_mu, y_sd = _zscore(y)

    # Grid on original scale
    xi = np.linspace(float(x.min()), float(x.max()), grid_size)
    yi = np.linspace(float(y.min()), float(y.max()), grid_size)
    grid_x, grid_y = np.meshgrid(xi, yi)

    # Corresponding scaled grid for evaluation
    gx_s = (grid_x - x_mu) / (x_sd if x_sd != 0 else 1.0)
    gy_s = (grid_y - y_mu) / (y_sd if y_sd != 0 else 1.0)

    # Fit robustly (may increase smooth)
    rbf, smooth_eff = _fit_rbf_robust(x_s, y_s, z, kernel=kernel, smooth=smooth)

    # Predict on scaled grid, return on original axes
    grid_z = rbf(gx_s, gy_s)
    return grid_x, grid_y, grid_z, smooth_eff


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

def _fit_and_score_rbf(
    df_clean: pd.DataFrame, property_name: str, kernel: str, smooth: float
) -> Tuple[float, float, float]:
    """
    In-sample metrics at sample points using the robust-fitting strategy.
    """
    df_prop = _dedupe_xy_mean(df_clean, property_name)
    x = df_prop["Time"].to_numpy(dtype=float)
    y = df_prop["Temperature"].to_numpy(dtype=float)
    z = df_prop[property_name].to_numpy(dtype=float)

    x_s, x_mu, x_sd = _zscore(x)
    y_s, y_mu, y_sd = _zscore(y)

    rbf, smooth_eff = _fit_rbf_robust(x_s, y_s, z, kernel=kernel, smooth=smooth)
    zhat = rbf(x_s, y_s)

    resid = zhat - z
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae  = float(np.mean(np.abs(resid)))
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((z - np.mean(z))**2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return rmse, mae, r2


# ---------------- pipeline ----------------

def execute_rbf_pipeline(
    properties: List[str] = ("Rp0.2", "Rm", "At", "HV", "E", "v", "Z", "A32", "R_bar", "delta_R"),
    kernels: List[str] = ("linear", "cubic", "thin_plate"),
    smooths: List[float] = (0.0, 0.1, 0.5, 1.0),
    grid_size: int = 100,
    y_strain: float = 1.5,
    compute_uncertainty: bool = False,
    uncertainty_k: int = 3,
    benchmark: bool = True,
) -> None:
    """
    Run standard RBF interpolation for each (kernel, smooth) combo and property.

    Saves:
      grids → outputs/rbf/grids/<prop>_rbf_<kernel>_s<smooth_eff>_grid.csv
      plots → plots/rbf/plots/<prop>_rbf_<kernel>_s<smooth_eff>_plot.png

    where smooth_eff is the effective value used after stability retries.
    """
    base_path = get_base_path()
    setup_repo_structure(base_path)

    technique = "rbf"
    grids_dir, metrics_dir, plots_dir = get_artifact_dirs(base_path, technique)

    log_dir = base_path / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_{technique}_{timestamp}.log"
    logger = setup_logging(log_file)
    logger.info("RBF interpolation pipeline started.")

    metrics_rows = []

    try:
        datasets_path = base_path / "datasets"
        df = read_dataset(datasets_path / "all_mechanical_properties.csv")
        df_r_values = read_dataset(datasets_path / "all_r_values.csv")
        df_clean = clean_data(df, df_r_values, y_strain)

        for kernel in kernels:
            for smooth in smooths:
                logger.info(f"Config: kernel={kernel}, smooth={smooth}")
                for prop in properties:
                    try:
                        logger.info(f"Interpolating property: {prop}")
                        gx, gy, gz, smooth_eff = interpolate_property_rbf(
                            df_clean, prop, kernel=kernel, smooth=float(smooth), grid_size=grid_size
                        )

                        # Save prediction grid + plot using effective smooth
                        save_interpolated_grid_versioned(
                            grids_dir, prop, technique, gx, gy, gz,
                            kernel=kernel, smooth=smooth_eff
                        )
                        plot_property_from_grid(
                            input_dir=grids_dir,
                            df_clean=df_clean,
                            property_name=prop,
                            technique=technique,
                            kernel=kernel,
                            smooth=smooth_eff,
                            save_path=plots_dir,
                        )
                        logger.info(
                            f"Saved grid and plot for {prop} (kernel={kernel}, smooth_eff={smooth_eff})"
                        )

                        # Benchmarks (in-sample)
                        if benchmark:
                            rmse, mae, r2 = _fit_and_score_rbf(df_clean, prop, kernel, float(smooth))
                            metrics_rows.append({
                                "technique": technique,
                                "property": prop,
                                "kernel": kernel,
                                "smooth_requested": float(smooth),
                                "smooth_effective": float(smooth_eff),
                                "grid_size": grid_size,
                                "n_samples": int(len(df_clean)),
                                "rmse": rmse, "mae": mae, "r2": r2,
                            })
                            logger.info(
                                f"[{technique}] {prop} (kernel={kernel}, requested_smooth={smooth}): "
                                f"RMSE={rmse:.4g} MAE={mae:.4g} R2={r2:.4g}"
                            )

                        # Proximity-based uncertainty (sigma)
                        if compute_uncertainty:
                            samples_xy = df_clean[["Time", "Temperature"]].to_numpy()
                            sigma = _proximity_sigma_grid(gx, gy, samples_xy, k=uncertainty_k)
                            sprop = f"{prop}_sigma"

                            save_interpolated_grid_versioned(
                                grids_dir, sprop, technique, gx, gy, sigma,
                                kernel=kernel, smooth=smooth_eff
                            )
                            plot_property_from_grid(
                                input_dir=grids_dir,
                                df_clean=df_clean,
                                property_name=sprop,
                                technique=technique,
                                kernel=kernel,
                                smooth=smooth_eff,
                                save_path=plots_dir,
                            )
                            logger.info(
                                f"Saved proximity-sigma grid and plot for {prop} "
                                f"(kernel={kernel}, smooth_eff={smooth_eff})"
                            )

                        if benchmark and metrics_rows:
                            save_metrics_records(metrics_dir, technique, metrics_rows, timestamp)
                            logger.info(f"Saved metrics CSV with {len(metrics_rows)} rows to {metrics_dir}")


                    except Exception as e:
                        logger.exception(
                            f"Skipping combo (kernel={kernel}, smooth={smooth}) for {prop} due to error: {e}"
                        )
                        continue

        logger.info("RBF pipeline completed.")

    except Exception as e:
        logger.exception(f"RBF pipeline failed: {e}")
        raise
