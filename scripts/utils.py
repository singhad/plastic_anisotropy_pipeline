import os
import sys
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple, Union
from datetime import datetime
import random

# Set environment variables to pin the repo root explicitly
PROJECT_ROOT_ENV_VARS = ("Kyriakos_Internship_Bernal_Institute", "code")


def _looks_like_repo_root(p: Path) -> bool:
    """
    Heuristic for project root:
    - Prefer a folder that contains a 'scripts' directory.
    - 'datasets' may or may not exist yet; don't require it strictly.
    """
    return (p / "scripts").exists()


def get_base_path(start: Optional[Union[Path, str]] = None) -> Path:
    """
    Resolve the project root folder robustly.

    Resolution order:
      1) If any of PROJECT_ROOT_ENV_VARS is set and exists, use that.
      2) Walk up from `start` (or CWD) until a folder containing 'scripts/' is found.
      3) Fallback to `Path(start).resolve()` or `Path.cwd().resolve()` if nothing is found.

    Examples
    --------
    >>> get_base_path() / "datasets"  # -> path to datasets/
    """
    # 1) Environment variable override (most explicit)
    for var in PROJECT_ROOT_ENV_VARS:
        env_path = os.environ.get(var)
        if env_path:
            p = Path(env_path).expanduser().resolve()
            if p.exists():
                return p

    # 2) Walk up from start (or CWD)
    here = Path(start).expanduser().resolve() if start else Path.cwd().resolve()
    for p in [here, *here.parents]:
        if _looks_like_repo_root(p):
            return p

    # 3) Fallback: the current location
    return here


def setup_repo_structure(base_path: Path) -> None:
    """
    Ensure core top-level folders exist:
      - datasets/
      - outputs/
      - plots/
      - logs/
    Technique-specific subfolders are created on-demand by `get_artifact_dirs`.
    """
    for name in ("datasets", "outputs", "plots", "logs"):
        (base_path / name).mkdir(parents=True, exist_ok=True)


def get_artifact_dirs(base_path: Path, technique: str) -> Tuple[Path, Path, Path]:
    """
    Return (grids_dir, metrics_dir, plots_dir) for a given technique and ensure they exist.

    Layout:
      CSV grids:   <base>/outputs/<technique>/grids/
      Metrics:     <base>/outputs/<technique>/metrics/
      Plot images: <base>/plots/<technique>/

    This keeps the tree uniform AND future-proof—adding a new technique requires
    no extra path wiring anywhere else.
    """
    grids_dir = base_path / "outputs" / technique / "grids"
    metrics_dir = base_path / "outputs" / technique / "metrics"
    plots_dir = base_path / "plots" / technique
    grids_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return grids_dir, metrics_dir, plots_dir


def setup_logging(log_file: Path, level: int = logging.INFO) -> logging.Logger:
    """
    Create a logger that logs to both file and console without duplicating handlers.

    Parameters
    ----------
    log_file : Path
        Full path to the desired log file.
    level : int
        Logging level (default: logging.INFO).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("plastic_anisotropy_pipeline")
    logger.setLevel(level)
    logger.propagate = False  # keep messages from bubbling to root multiple times

    # Clear existing handlers to avoid duplicates across multiple runs in the same process
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


def timestamp_str(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """
    Convenience helper for consistent timestamps in filenames.
    """
    return datetime.now().strftime(fmt)


def set_seed(seed: int = 42) -> None:
    """
    Set Python and NumPy RNG seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)


__all__ = [
    "get_base_path",
    "setup_repo_structure",
    "get_artifact_dirs",
    "setup_logging",
    "timestamp_str",
    "set_seed",
]
