from typing import List, Tuple, Optional
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from pykrige.ok import OrdinaryKriging

from utils import get_base_path, setup_repo_structure, setup_logging, get_artifact_dirs
from data_loader import read_dataset, clean_data
from saving import save_interpolated_grid_versioned, save_metrics_records
from plotting import plot_property_from_grid


# ---------------- helpers ----------------

def _build_grid_vectors(x: np.ndarray, y: np.ndarray, grid_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xi = np.linspace(float(np.min(x)), float(np.max(x)), grid_size)
    yi = np.linspace(float(np.min(y)), float(np.max(y)), grid_size)
    gx, gy = np.meshgrid(xi, yi)
    return xi, yi, gx, gy


def _dedupe_xy_mean(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Collapse duplicates at the same (Time, Temperature) by averaging the value."""
    cols = ["Time", "Temperature", value_col]
    if not set(cols).issubset(df.columns):
        raise ValueError(f"DataFrame missing required columns: {cols}")
    return (df[cols]
            .groupby(["Time", "Temperature"], as_index=False, sort=False)[value_col]
            .mean())


def _override_nugget_on_model(ok: OrdinaryKriging, variogram_model: str, nugget: Optional[float]) -> OrdinaryKriging:
    """
    Overwrite the fitted variogram nugget while keeping sill & range.
    Works across PyKrige versions by preferring update_variogram_model and
    falling back to re-instantiation with explicit parameters.
    """
    if nugget is None:
        return ok

    try:
        fitted = ok.variogram_model_parameters  # dict or list, depending on version/model
    except Exception:
        return ok

    params_mod = None
    if isinstance(fitted, dict):
        params_mod = dict(fitted)
        params_mod["nugget"] = float(nugget)
    else:
        try:
            if len(fitted) >= 2:
                sill, r = float(fitted[0]), float(fitted[1])
                params_mod = [sill, r, float(nugget)]
        except Exception:
            params_mod = None

    if params_mod is None:
        return ok

    try:
        if hasattr(ok, "update_variogram_model"):
            ok.update_variogram_model(variogram_model=variogram_model, variogram_parameters=params_mod)
            return ok
    except Exception:
        pass

    # Fallback: rebuild with explicit params
    try:
        x, y, z = ok.X_DATA, ok.Y_DATA, ok.Z_DATA
        ok_new = OrdinaryKriging(
            x, y, z,
            variogram_model=variogram_model,
            variogram_parameters=params_mod,
            enable_plotting=False,
            enable_statistics=False,
            coordinates_type="euclidean",
        )
        return ok_new
    except Exception:
        return ok


def _build_ok(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    variogram_model: str,
    nugget: Optional[float],
    exact_values: bool,
    pseudo_inv: bool = False
) -> OrdinaryKriging:
    """Construct OK, fit variogram, then inject nugget if provided."""
    ok = OrdinaryKriging(
        x, y, z,
        variogram_model=variogram_model,
        variogram_parameters=None,           # let PyKrige fit sill/range(/nugget)
        enable_plotting=False,
        enable_statistics=False,
        coordinates_type="euclidean",
        exact_values=exact_values,
        pseudo_inv=pseudo_inv,
    )
    ok = _override_nugget_on_model(ok, variogram_model, nugget)
    return ok


def _exec_grid_robust(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    xi: np.ndarray, yi: np.ndarray,
    variogram_model: str,
    nugget: Optional[float],
    logger
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Try execute("grid", xi, yi) with robustness fallbacks:
      1) as-is;
      2) if singular and nugget is zero/None -> tiny nugget 1e-6, exact_values=False;
      3) if still singular -> pseudo_inv=True;
      4) if still singular -> neighborhood solve with n_closest_points=16.
    """
    # exact_values: True when nugget ~ 0; False when nugget > 0
    exact = not (nugget is not None and float(nugget) > 0.0)

    # Attempt 1: as requested
    try:
        ok = _build_ok(x, y, z, variogram_model, nugget, exact_values=exact, pseudo_inv=False)
        gz, gvar = ok.execute("grid", xi, yi)
        return np.asarray(np.ma.getdata(gz)), np.asarray(np.ma.getdata(gvar))
    except LinAlgError as e:
        logger.warning(f"Kriging singular matrix on first attempt (nugget={nugget}). Trying fallbacks...")

    # Attempt 2: tiny nugget if none
    tiny_nugget = None
    if nugget is None or float(nugget) == 0.0:
        tiny_nugget = 1e-6
        try:
            ok = _build_ok(x, y, z, variogram_model, tiny_nugget, exact_values=False, pseudo_inv=False)
            gz, gvar = ok.execute("grid", xi, yi)
            logger.warning(f"Kriging stabilized by tiny nugget={tiny_nugget}.")
            return np.asarray(np.ma.getdata(gz)), np.asarray(np.ma.getdata(gvar))
        except LinAlgError:
            logger.warning("Tiny nugget fallback failed; trying pseudo-inverse...")

    # Attempt 3: pseudo-inverse
    try:
        ok = _build_ok(x, y, z, variogram_model, (nugget if nugget is not None and nugget > 0 else (tiny_nugget or 0.0)),
                       exact_values=False, pseudo_inv=True)
        gz, gvar = ok.execute("grid", xi, yi)
        logger.warning("Kriging stabilized with pseudo_inv=True.")
        return np.asarray(np.ma.getdata(gz)), np.asarray(np.ma.getdata(gvar))
    except LinAlgError:
        logger.warning("Pseudo-inverse fallback failed; trying neighborhood solve (n_closest_points=16)...")

    # Attempt 4: neighborhood solve
    ok = _build_ok(x, y, z, variogram_model, (nugget if nugget is not None else (tiny_nugget or 0.0)),
                   exact_values=False, pseudo_inv=True)
    gz, gvar = ok.execute("grid", xi, yi, n_closest_points=16)  # local solve
    logger.warning("Kriging stabilized with neighborhood solve (n_closest_points=16) + pseudo_inv.")
    return np.asarray(np.ma.getdata(gz)), np.asarray(np.ma.getdata(gvar))
    # return np.asarray(np.ma.getdata(gz))


# ---------------- core interpolation ----------------

def interpolate_property_kriging(
    df_clean: pd.DataFrame,
    property_name: str,
    grid_size: int = 100,
    variogram_model: str = "gaussian",
    nugget: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Ordinary Kriging on a uniform grid (robust execution).
    Returns meshgrid X, Y, prediction Z, and sigma (sqrt of kriging variance).
    """
    # Deduplicate just in case; harmless if already clean
    dfp = _dedupe_xy_mean(df_clean, property_name)

    x = dfp["Time"].to_numpy(dtype=float)
    y = dfp["Temperature"].to_numpy(dtype=float)
    z = dfp[property_name].to_numpy(dtype=float)

    xi, yi, grid_x, grid_y = _build_grid_vectors(x, y, grid_size)

    # Robust execution with fallbacks
    gz, gvar = _exec_grid_robust(x, y, z, xi, yi, variogram_model, nugget, logger=_KRIG_LOGGER)
    # gz = _exec_grid_robust(x, y, z, xi, yi, variogram_model, nugget, logger=_KRIG_LOGGER)
    sigma = np.sqrt(np.maximum(gvar, 0.0))
    return grid_x, grid_y, gz, sigma
    # return grid_x, grid_y, gz


# ---------------- benchmarks ----------------

def _fit_and_score_kriging(
    df_clean: pd.DataFrame,
    property_name: str,
    variogram_model: str,
    nugget: Optional[float],
) -> Tuple[float, float, float, float]:
    """
    In-sample metrics at sample points using robust settings mirrored from the grid path.
    Returns (RMSE, MAE, R2, mean_sigma_at_samples).
    """
    dfp = _dedupe_xy_mean(df_clean, property_name)
    x = dfp["Time"].to_numpy(dtype=float)
    y = dfp["Temperature"].to_numpy(dtype=float)
    z = dfp[property_name].to_numpy(dtype=float)

    # mirror the robust builder: prefer positive nugget & pseudo-inverse if needed
    exact = not (nugget is not None and float(nugget) > 0.0)
    try:
        ok = _build_ok(x, y, z, variogram_model, nugget, exact_values=exact, pseudo_inv=False)
        zhat, var = ok.execute("points", x, y)
    except LinAlgError:
        # tiny nugget then pseudo_inv
        try:
            ok = _build_ok(x, y, z, variogram_model, (1e-6 if (nugget is None or nugget == 0.0) else nugget),
                           exact_values=False, pseudo_inv=True)
            zhat, var = ok.execute("points", x, y)
        except LinAlgError:
            # final fallback: local solve
            ok = _build_ok(x, y, z, variogram_model, (1e-6 if (nugget is None or nugget == 0.0) else nugget),
                           exact_values=False, pseudo_inv=True)
            zhat, var = ok.execute("points", x, y, n_closest_points=min(16, len(x)))

    zhat = np.asarray(np.ma.getdata(zhat), dtype=float)
    var = np.asarray(np.ma.getdata(var), dtype=float)

    resid = zhat - z
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae  = float(np.mean(np.abs(resid)))
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((z - np.mean(z))**2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    mean_sigma = float(np.mean(np.sqrt(np.maximum(var, 0.0))))
    return rmse, mae, r2, mean_sigma


# ---------------- pipeline ----------------

# Module-level logger handle injected by setup_logging at runtime
_KRIG_LOGGER = None  # set inside execute_kriging_pipeline


def execute_kriging_pipeline(
    properties: List[str] = ("Rp0.2", "Rm", "At", "HV", "E", "v", "Z", "A32", "R_bar", "delta_R"),
    variogram_models: List[str] = ("gaussian", "spherical", "exponential"),
    nuggets: List[float] = (0.0,),
    grid_size: int = 100,
    y_strain: float = 1.5,
    compute_uncertainty: bool = False,   # kept for CLI parity; kriging always saves sigma
    uncertainty_k: int = 3,             # unused for kriging; kept for uniform CLI
    benchmark: bool = False,
) -> None:
    """
    Run Ordinary Kriging for each (variogram_model, nugget) combo and property.

    Saves:
      grids → outputs/kriging/grids/<prop>_kriging_<model>_s<nugget>_grid.csv
      plots → plots/kriging/plots/<prop>_kriging_<model>_s<nugget>_plot.png

    Also saves native kriging uncertainty:
      <prop>_sigma_kriging_<model>_s<nugget>_grid.csv + plot
    """
    base_path = get_base_path()
    setup_repo_structure(base_path)

    technique = "kriging"
    grids_dir, metrics_dir, plots_dir = get_artifact_dirs(base_path, technique)

    log_dir = base_path / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_{technique}_{timestamp}.log"
    logger = setup_logging(log_file)
    global _KRIG_LOGGER
    _KRIG_LOGGER = logger
    logger.info("Kriging pipeline started.")

    metrics_rows = []

    try:
        datasets_path = base_path / "datasets"
        df = read_dataset(datasets_path / "all_mechanical_properties.csv")
        df_r_values = read_dataset(datasets_path / "all_r_values.csv")
        df_clean = clean_data(df, df_r_values, y_strain)

        for model in variogram_models:
            for nugget in nuggets:
                logger.info(f"Config: variogram_model={model}, nugget={nugget}")
                for prop in properties:
                    try:
                        logger.info(f"Interpolating property: {prop}")
                        gx, gy, gz, sigma = interpolate_property_kriging(
                            df_clean,
                            property_name=prop,
                            grid_size=grid_size,
                            variogram_model=model,
                            nugget=float(nugget),
                        )

                        # Save prediction grid + plot
                        save_interpolated_grid_versioned(
                            grids_dir, prop, technique, gx, gy, gz,
                            variogram_model=model, nugget=nugget
                        )
                        plot_property_from_grid(
                            input_dir=grids_dir,
                            df_clean=df_clean,
                            property_name=prop,
                            technique=technique,
                            variogram_model=model,
                            nugget=nugget,
                            save_path=plots_dir,
                        )
                        logger.info(f"Saved grid and plot for {prop} (model={model}, nugget={nugget})")

                        # Save sigma grid + plot (native kriging uncertainty)
                        sprop = f"{prop}_sigma"
                        save_interpolated_grid_versioned(
                            grids_dir, sprop, technique, gx, gy, sigma,
                            variogram_model=model, nugget=nugget
                        )
                        # Uncomment to plot sigma grids
                        # plot_property_from_grid(
                        #     input_dir=grids_dir,
                        #     df_clean=df_clean,
                        #     property_name=sprop,
                        #     technique=technique,
                        #     variogram_model=model,
                        #     nugget=nugget,
                        #     save_path=plots_dir,
                        # )
                        logger.info(f"Saved sigma grid and plot for {prop} (model={model}, nugget={nugget})")

                        # Benchmarks (in-sample) + mean sigma at samples
                        if benchmark:
                            rmse, mae, r2, mean_sigma = _fit_and_score_kriging(
                                df_clean, prop, model, float(nugget)
                            )
                            metrics_rows.append({
                                "technique": technique,
                                "property": prop,
                                "variogram_model": model,
                                "nugget": float(nugget),
                                "grid_size": grid_size,
                                "n_samples": int(len(df_clean)),
                                "rmse": rmse, "mae": mae, "r2": r2,
                                "mean_sigma_at_samples": mean_sigma,
                            })
                            logger.info(
                                f"[{technique}] {prop} (model={model}, nugget={nugget}): "
                                f"RMSE={rmse:.4g} MAE={mae:.4g} R2={r2:.4g} mean_sigma={mean_sigma:.4g}"
                            )

                        if benchmark and metrics_rows:
                            save_metrics_records(metrics_dir, technique, metrics_rows, timestamp)
                            logger.info(f"Saved metrics CSV with {len(metrics_rows)} rows to {metrics_dir}")

                    except Exception as e:
                        logger.exception(
                            f"Skipping (model={model}, nugget={nugget}) for {prop} due to error: {e}"
                        )
                        continue

        logger.info("Kriging pipeline completed.")

    except Exception as e:
        logger.exception(f"Kriging pipeline failed: {e}")
        raise
