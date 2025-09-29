from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.spatial import Delaunay

# ---------------------------------------------------------------------
# Helpers: filename suffixes, CSV -> meshgrid loader, contour renderer
# ---------------------------------------------------------------------

def _build_suffix(
    technique: str,
    kernel: Optional[str] = None,
    smooth: Optional[float] = None,
    variogram_model: Optional[str] = None,
    nugget: Optional[float] = None,
) -> str:
    """
    Mirror the filename tags used in saving.py:
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


def _guess_xy_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Try to infer the X/Y column names used in long-form saved CSVs.
    Prefers ('Time', 'Temperature'), falls back to ('x', 'y') / ('grid_x', 'grid_y').
    """
    if {"Time", "Temperature"}.issubset(df.columns):
        return "Time", "Temperature"
    if {"x", "y"}.issubset(df.columns):
        return "x", "y"
    if {"grid_x", "grid_y"}.issubset(df.columns):
        return "grid_x", "grid_y"
    # best-effort fallback: first two numeric columns
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) >= 2:
        return numeric_cols[0], numeric_cols[1]
    raise ValueError("Could not determine X/Y columns in grid CSV.")


def _load_grid(csv_path: Path, value_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a grid CSV saved by save_interpolated_grid_versioned and return (grid_x, grid_y, grid_z).

    Expected (long-form) columns:
      - X: 'Time' (preferred) or 'x' / 'grid_x'
      - Y: 'Temperature' (preferred) or 'y' / 'grid_y'
      - Value: <value_col>  (e.g., 'Rp0.2' or 'Rp0.2_sigma')

    Robust to column order and extra metadata columns.
    """
    df = pd.read_csv(csv_path)
    xcol, ycol = _guess_xy_columns(df)

    # if the expected value column is missing, try a best-effort guess (third numeric col)
    if value_col not in df.columns:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in {xcol, ycol}]
        if not numeric_cols:
            raise ValueError(f"Value column '{value_col}' not found in {csv_path.name}.")
        value_col = numeric_cols[0]

    xs = np.sort(df[xcol].unique().astype(float))
    ys = np.sort(df[ycol].unique().astype(float))

    # Pivot to 2D grid (index=y, columns=x) so mesh aligns as meshgrid(xs, ys)
    grid = (
        df.pivot_table(index=ycol, columns=xcol, values=value_col)
          .reindex(index=ys, columns=xs)
    )
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_z = grid.values.astype(float)
    return grid_x, grid_y, grid_z


def _mask_outside_convex_hull(grid_x, grid_y, samples_xy):
    tri = Delaunay(samples_xy)            # triangulation over samples
    pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    outside = tri.find_simplex(pts) < 0   # -1 => outside hull
    return outside.reshape(grid_x.shape)


def _plot_contour(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    df_clean: pd.DataFrame,
    property_name: str,
    title: str,
    levels: int = 25,
    padding_x: float = 1.0,
    padding_y: float = 10.0,
    mask_outside: bool = True,
):
    fig, ax = plt.subplots(figsize=(8, 6))

    # build masked Z
    Z = np.ma.masked_invalid(grid_z)
    if mask_outside and {"Time","Temperature"}.issubset(df_clean.columns):
        samples_xy = df_clean[["Time","Temperature"]].to_numpy()
        hull_mask = _mask_outside_convex_hull(grid_x, grid_y, samples_xy)
        Z = np.ma.array(Z, mask=(Z.mask | hull_mask))

    cs = ax.contourf(grid_x, grid_y, Z, levels=levels, cmap="rainbow")
    ax.set_facecolor("white")  # masked regions render as white
    cbar = plt.colorbar(cs, ax=ax)
    cbar.set_label(property_name)
 
    # overlay experimental points
    if {"Time", "Temperature"}.issubset(df_clean.columns):
        ax.scatter(df_clean["Time"], df_clean["Temperature"], color="black", marker="x")

    ax.set_title(title)
    ax.set_xlabel("Aging Time [h]")
    ax.set_ylabel("Aging Temperature [°C]")
    ax.grid(True)

    # padding
    try:
        xmin, xmax = float(np.min(grid_x)), float(np.max(grid_x))
        ymin, ymax = float(np.min(grid_y)), float(np.max(grid_y))
        ax.set_xlim(xmin - padding_x, xmax + padding_x)
        ax.set_ylim(ymin - padding_y, ymax + padding_y)
    except Exception:
        pass

    return fig


# ---------------------------------------------------------------------
# Public plotting APIs
# ---------------------------------------------------------------------

def plot_property_from_grid(
    input_dir: Path,
    df_clean: pd.DataFrame,
    property_name: str,
    technique: str,
    *,
    kernel: Optional[str] = None,
    smooth: Optional[float] = None,
    variogram_model: Optional[str] = None,
    nugget: Optional[float] = None,
    levels: int = 25,
    padding_x: float = 1.0,
    padding_y: float = 10.0,
    save_path: Optional[Path] = None,
    title: Optional[str] = None
):
    """
    Load the grid CSV matching the provided args and render the standard contour plot.
    Reads from:  input_dir / f"{property_name}_{technique}{suffix}_grid.csv"
    Saves to:    save_path / f"{property_name}_{technique}{suffix}_plot.png"  (if save_path is given)
    """
    suffix = _build_suffix(technique, kernel, smooth, variogram_model, nugget)
    grid_csv = input_dir / f"{property_name}_{technique}{suffix}_grid.csv"
    if not grid_csv.exists():
        print(f"[plot_property_from_grid] Grid CSV not found: {grid_csv}")
        return

    grid_x, grid_y, grid_z = _load_grid(grid_csv, property_name)

    # default title if not provided
    if title is None:
        tag = ""
        if technique in {"rbf", "rbf_compact"}:
            bits = []
            if kernel: bits.append(f"kernel={kernel}")
            if smooth is not None: bits.append(f"smooth={smooth}")
            tag = f" ({', '.join(bits)})" if bits else ""
        elif technique == "kriging":
            bits = []
            if variogram_model: bits.append(f"model={variogram_model}")
            if nugget is not None: bits.append(f"nugget={nugget}")
            tag = f" ({', '.join(bits)})" if bits else ""
        title = f"{property_name} — {technique}{tag}"

    fig = _plot_contour(
        grid_x, grid_y, grid_z, df_clean,
        property_name=property_name, title=title,
        levels=levels, padding_x=padding_x, padding_y=padding_y
    )

    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        out_png = save_path / f"{property_name}_{technique}{suffix}_plot.png"
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_heatmap_from_grid(
    input_dir: Path,
    df_clean: pd.DataFrame,
    title: str,
    basename: str,
    technique: str,
    *,
    kernel: Optional[str] = None,
    smooth: Optional[float] = None,
    variogram_model: Optional[str] = None,
    nugget: Optional[float] = None,
    levels: int = 25,
    padding_x: float = 1.0,
    padding_y: float = 10.0,
    save_path: Optional[Path] = None,
):
    """
    Generic loader + plotter when the value column name is not a canonical property name,
    e.g., for '<prop>_sigma' produced by kriging or proximity-σ.
    """
    suffix = _build_suffix(technique, kernel, smooth, variogram_model, nugget)
    grid_csv = input_dir / f"{basename}_{technique}{suffix}_grid.csv"
    if not grid_csv.exists():
        print(f"[plot_heatmap_from_grid] Grid CSV not found: {grid_csv}")
        return

    grid_x, grid_y, grid_z = _load_grid(grid_csv, basename)
    fig = _plot_contour(
        grid_x, grid_y, grid_z, df_clean,
        property_name=basename, title=title,
        levels=levels, padding_x=padding_x, padding_y=padding_y
    )

    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        out_png = save_path / f"{basename}_{technique}{suffix}_plot.png"
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_gradient_from_grid(
    input_dir: Path,
    df_clean: pd.DataFrame,
    property_name: str,
    technique: str,
    *,
    kernel: Optional[str] = None,
    smooth: Optional[float] = None,
    variogram_model: Optional[str] = None,
    nugget: Optional[float] = None,
    stride: int = 6,
    levels: int = 25,
    save_path: Optional[Path] = None,
):
    """
    Loads a saved property grid, computes gradient magnitude + direction (quiver),
    and saves as <property>_<technique>..._grad_plot.png
    """
    suffix = _build_suffix(technique, kernel, smooth, variogram_model, nugget)
    grid_csv = input_dir / f"{property_name}_{technique}{suffix}_grid.csv"
    if not grid_csv.exists():
        print(f"[plot_gradient_from_grid] Grid CSV not found: {grid_csv}")
        return

    gx, gy, gz = _load_grid(grid_csv, property_name)

    # gradients in grid coordinates
    dzy, dzx = np.gradient(gz, edge_order=2)
    gmag = np.sqrt(dzx**2 + dzy**2)

    fig, ax = plt.subplots(figsize=(8, 6))

    # # Uncomment to mask outside convex hull when plotting using the saved grids, 
    # # like the plots generated automatically
    # Z = np.ma.masked_invalid(gmag)
    # if {"Time","Temperature"}.issubset(df_clean.columns):
    #     samples_xy = df_clean[["Time","Temperature"]].to_numpy()
    #     hull_mask = _mask_outside_convex_hull(gx, gy, samples_xy)
    #     Z = np.ma.array(Z, mask=(Z.mask | hull_mask))
    # c = ax.contourf(gx, gy, Z, levels=levels, cmap="rainbow")
    # ax.set_facecolor("white")
    
    c = ax.contourf(gx, gy, gmag, levels=levels, cmap="rainbow")
    plt.colorbar(c, ax=ax, label=f"|∇{property_name}|")

    if {"Time", "Temperature"}.issubset(df_clean.columns):
        ax.scatter(df_clean["Time"], df_clean["Temperature"], color="black", marker="x", s=12)

    # sparse quiver overlay for direction
    try:
        ax.quiver(
            gx[::stride, ::stride], gy[::stride, ::stride],
            dzx[::stride, ::stride], dzy[::stride, ::stride],
            scale=50
        )
    except Exception:
        # If the grid is too small or stride too large, skip quiver gracefully
        pass

    # Title + labels
    tag = ""
    if technique in {"rbf", "rbf_compact"}:
        bits = []
        if kernel: bits.append(f"kernel={kernel}")
        if smooth is not None: bits.append(f"smooth={smooth}")
        tag = f" ({', '.join(bits)})" if bits else ""
    elif technique == "kriging":
        bits = []
        if variogram_model: bits.append(f"model={variogram_model}")
        if nugget is not None: bits.append(f"nugget={nugget}")
        tag = f" ({', '.join(bits)})" if bits else ""

    ax.set_title(f"{property_name} — GRADIENT, {technique}{tag}")
    ax.set_xlabel("Aging Time [h]")
    ax.set_ylabel("Aging Temperature [°C]")
    ax.grid(True)

    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        out_png = save_path / f"{property_name}_{technique}{suffix}_grad_plot.png"
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
