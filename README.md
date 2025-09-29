# plastic_anisotropy_pipeline

# AM Plastic Anisotropy — Interpolation & Contour Mapping

A Python toolkit to use interpolators (Akima, RBF, Kriging, Linear, Nearest, Biharmonic) for exploring how **aging time** and **aging temperature** affect plastic anisotropy and other mechanical properties in additively manufactured maraging steel.

> **Status:** ***Active***. Uncertainty maps are **OFF by default** (future work) but can be turned on by using the `--compute-uncertainty` flag in the CLI. Benchmarks/metrics saving is ON when running everything using the `--method all` flag or when `--benchmark` flag is provided for a specific method in the CLI.
---

## Features

- Multiple interpolators: **Akima, RBF, Kriging, Linear, Nearest, Biharmonic (Clough–Tocher), RBF-compact**.
- Multiple mechanical properties: **Rp0.2, Rm, At, HV, E, v, Z, A32, R_bar, delta_R**.
- Other adjustable parameters: `--grid-size`,`--y-strain`.
- Uniform contour plotting + grid exports for downstream analysis.
- Reproducible output structure (grids/plots/metrics by technique).
- CLI for batch runs across multiple properties.
- Optional benchmarks (RMSE/MAE/R²) via `--benchmark`.
- Optional uncertainty quantification via `--compute-uncertainty` (future work).
- Modular design for easy extension to new materials/properties.

---

## Repository layout

```plaintext
plastic_anisotropy_pipeline/
├── environment.yml                 # Conda environment file
├── requirements.txt                # List of dependencies
├── .gitignore                      # Gitignore file
├── LICENSE                         # License file
├── datasets/                       # Input CSV files
│   ├── all_mechanical_properties.csv
│   ├── all_r_values.csv
├── outputs/                        # Interpolated grids & benchmarks in separate folders for each interpolator(.csv)
├── plots/                          # Generated plots in separate folders for each interpolator (.png)
├── logs/                           # Log files (.log)
├── scripts/
│   ├── __init__.py                 # Package initializer with version info
│   ├── cli.py                      # command-line entry point
│   ├── data_loader.py              # dataset ingestion & merging
│   ├── plotting.py                 # contour plotting
│   ├── saving.py                   # file paths & persistence
│   └── utils.py                    # logging & helpers
├── interpolators/                  # Interpolator implementations
│   ├── __init__.py                 # Interpolator package initializer
│   ├── akima.py                    # Akima interpolator
│   ├── rbf.py                      # Radial Basis Function interpolator
│   ├── kriging.py                  # Kriging interpolator
│   ├── linear.py                   # Linear interpolator
│   ├── nearest.py                  # Nearest neighbor interpolator
│   ├── biharmonic.py               # Biharmonic (Clough–Tocher) interpolator
│   └── rbf_compact.py              # Compact RBF interpolator
└── README.md                       # Project documentation
```


---

## Installation

### Option A: Conda (recommended)

```bash
# (Optional) Update conda first
conda update -n base -c conda-forge conda

# Create and activate env from environment.yml
conda env create -f environment.yml
conda activate plastic_anisotropy_pipeline
```
### Option B: Pure pip/venv
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt # installing dependencies from requirements.txt
```
---
## Quickstart

Run interpolation for all properties with default settings:
```bash
python scripts/cli.py --method all
```
This will generate contour plots, interpolated grids and benchmarks for all mechanical properties in the `datasets/` folder, saving outputs in the `outputs/<interpolator>` and `plots/<interpolator>` directories.

### Available CLI options
```bash
# Required
--method {akima,rbf,kriging,nearest,linear,biharmonic,rbf_compact,all}
    Which technique to run, or 'all' to sweep everything (including benchmarks).

# Core
--properties PROP [PROP ...]
    Mechanical property columns to process.
    default: Rp0.2 Rm At HV E v Z A32 R_bar delta_R

--grid-size INT
    Grid resolution along each axis (e.g., 100 → 100x100).
    default: 100

--y-strain FLOAT
    Axial Strain value in percent (used for r-value flows).
    default: 1.5

# Benchmarking (opt-in; OFF by default)
--benchmark
    Log in-sample fit metrics (RMSE/MAE/R²) and save CSV metrics.
    default: False (flag turns it on)

# Uncertainty (opt-in; OFF by default)
--compute-uncertainty
    Compute proximity-based sigma maps (kriging already has native variance).
    default: False (flag turns it on)

--uncertainty-k INT
    k for k-NN proximity uncertainty (only used if uncertainty is enabled).
    default: 3

# RBF (standard) sweeps
--rbf-kernels K1 [K2 ...]
    RBF kernels to sweep.
    default: linear cubic thin_plate

--rbf-smooths S1 [S2 ...]
    RBF smooth values to sweep.
    default: 0.0 0.1 0.5 1.0

# Kriging sweeps
--kriging-models M1 [M2 ...]
    Variogram models to sweep.
    default: gaussian spherical exponential

--kriging-nuggets N1 [N2 ...]
    Nugget values to sweep.
    default: 0.0

# RBF-compact knobs
--rbf-compact-kernel {gaussian,linear,cubic,thin_plate}
    Kernel for compact-ish RBF flavour.
    default: gaussian

--rbf-compact-epsilon FLOAT
    Epsilon (shape parameter) for compact-ish RBF.
    default: 1.0

--rbf-compact-smooth FLOAT
    Smooth for compact-ish RBF.
    default: 0.01
```
### Sample runs
```bash
# Minimal Akima run
python scripts/cli.py --method akima --properties Rp0.2 --grid-size 200

# Run everything for four properties
python scripts/cli.py --method all --properties Rp0.2 Rm At HV

# Kriging with benchmarks
python scripts/cli.py --method kriging --properties Rp0.2 --benchmark

# Standard RBF sweep (kernels × smooths)
python scripts/cli.py --method rbf --properties Rp0.2 \
  --rbf-kernels linear cubic thin_plate \
  --rbf-smooths 0.0 0.01 0.1

# Kriging models × nuggets sweep
python scripts/cli.py --method kriging --properties Rm \
  --kriging-models gaussian spherical exponential \
  --kriging-nuggets 0.0 1e-6 1e-3
```
---
## Uncertainty (future work)

Uncertainty maps/plots are **not computed by default**. The code paths are preserved for later study and can be enabled on demand.

**How to enable (opt-in):**
```bash
# For all methods
python scripts/cli.py --method all --compute-uncertainty

# Kriging (native variance)
python scripts/cli.py --method kriging --properties Rp0.2 --compute-uncertainty

# Non-kriging methods (proximity-based sigma)
python scripts/cli.py --method rbf --properties Rp0.2 --compute-uncertainty --uncertainty-k 3
```
**Notes:**
- Default is **OFF** to avoid confusion while interpretation/calibration is pending.
- When enabled, uncertainty artifacts are saved alongside grids/plots under the method’s output folders.

---

## Troubleshooting

- **VS Code uses the wrong Python (pyenv vs conda)**
    - Select the interpreter: *Command Palette → Python: Select Interpreter → pick plastic_anisotropy_pipeline*.
    - Or, run without activation:
        ```bash
        conda run -n plastic_anisotropy_pipeline python scripts/cli.py --method all
        ```
- **Benchmarks not saving**
    - Pass the `--benchmark` flag or use `--method all` to enable metrics saving.
    ```bash
    python scripts/cli.py --method kriging --properties Rp0.2 --benchmark
    ```
    - Internally, pipelines receive a `benchmark` boolean. Ensure downstream helpers are called with keyword args (not positional) to avoid flag mis-ordering.

---

## Roadmap
- [X] Interpolators ported (Akima, RBF, Kriging, Linear, Nearest, Biharmonic, RBF-compact).
- [X] CLI orchestration across properties.
- [X] Benchmark CSVs (--benchmark) and standardized output structure.
- [ ] Uncertainty guidance (methodology + examples) — future work.
- [ ] Publish full documentation via GitHub Pages (MkDocs + Material + mkdocstrings).
- [ ] Add small unit “smoke tests” (loader, one grid per technique).
- [ ] Global color-scale presets per property for cross-figure comparability.
- [ ] Transforming the idea into a low-code/no-code (LC/NC) framework (future work).

---
## Citation
If this toolkit supports your work, please cite both the originating study and this repository.

---
## Acknowledgements
This project is a collaboration that builds upon the prior experimental investigations and R-based workflows developed by Dr. Kyriakos Kourousis's research group at the Bernal Institute at the University of Limerick, and extends this work through new computational developments carried out within Prof. Tiziana Margaria's SCCE group at CSIS, University of Limerick.

I thank Prof. Tiziana Margaria and Dr. Kyriakos Kourousis for initiating this project and supporting my work. I am grateful to Dr. Kourousis for datasets, R code, outputs, and regular discussions, to Prof. Margaria for her roadmap toward a low-code/no-code framework, and to Dr. Stephen Ryan for valuable input on LC/NC and R code.

