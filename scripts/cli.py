import argparse
import sys
from pathlib import Path

# Ensure 'scripts/' is on the path so 'interpolators' resolves when running as 'python scripts/cli.py'
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

# Import first-class pipelines from the new package
from interpolators import (
    execute_akima_pipeline,
    execute_rbf_pipeline,
    execute_kriging_pipeline,
    execute_nearest_pipeline,
    execute_linear_pipeline,
    execute_biharmonic_pipeline,
    execute_rbf_compact_pipeline,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run interpolation pipelines for mechanical property estimation."
    )

    # Core selection
    p.add_argument(
        "--method",
        choices=[
            "akima",
            "rbf",
            "kriging",
            "nearest",
            "linear",
            "biharmonic",
            "rbf_compact",
            "all",
        ],
        required=True,
        help="Which technique to run, or 'all' to sweep everything.",
    )
    p.add_argument(
        "--properties",
        nargs="+",
        default=["Rp0.2", "Rm", "At", "HV", "E", "v", "Z", "A32", "R_bar", "delta_R"],
        help="Mechanical property columns to process.",
    )
    p.add_argument(
        "--grid-size",
        type=int,
        default=100,
        help="Grid resolution along each axis (e.g., 100 → 100x100).",
    )
    p.add_argument(
        "--y-strain",
        type=float,
        default=1.5,
        help="Axial Strain value in percent (default: 1.5)."
    )
    
    # Benchmarking
    p.add_argument(
        "--benchmark",
        action="store_true",
        help="Log in-sample fit metrics (RMSE/MAE/R2).",
    )

    # Uncertainty (applies to all techniques - False by default; kriging has native sigma)
    p.add_argument(
        "--compute-uncertainty",
        action="store_true",
        help="Compute proximity-based sigma maps (kriging already saves its own sigma).",
    )
    p.add_argument(
        "--uncertainty-k",
        type=int,
        default=3,
        help="k for k-NN proximity uncertainty (default: 3).",
    )
    
    # RBF sweep
    p.add_argument(
        "--rbf-kernels",
        nargs="+",
        default=["linear", "cubic", "thin_plate"],
        help="RBF kernels to sweep (standard RBF).",
    )
    p.add_argument(
        "--rbf-smooths",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.5, 1.0],
        help="RBF smooth values to sweep (standard RBF).",
    )

    # Kriging sweep
    p.add_argument(
        "--kriging-models",
        nargs="+",
        default=["gaussian", "spherical", "exponential"],
        help="Variogram models to sweep.",
    )
    p.add_argument(
        "--kriging-nuggets",
        nargs="+",
        type=float,
        default=[0.0],
        help="Nugget values to sweep.",
    )

    # Compact-ish RBF knobs
    p.add_argument(
        "--rbf-compact-kernel",
        choices=["gaussian", "linear", "cubic", "thin_plate"],
        default="gaussian",
        help="Kernel for compact-ish RBF flavour.",
    )
    p.add_argument(
        "--rbf-compact-epsilon",
        type=float,
        default=1.0,
        help="Epsilon (shape parameter) for compact-ish RBF.",
    )
    p.add_argument(
        "--rbf-compact-smooth",
        type=float,
        default=0.01,
        help="Smooth for compact-ish RBF.",
    )

    return p


def _run_all_methods(args: argparse.Namespace) -> None:
    """Sweep every technique with provided flags."""
    # Force full artifacts for --method all
    args.compute_uncertainty = False  # Change to True to always compute uncertainty
    args.benchmark = True
    args.uncertainty_k = 3
    
    # AKIMA
    execute_akima_pipeline(
        properties=args.properties,
        grid_size=args.grid_size,
        y_strain=args.y_strain,
        compute_uncertainty=args.compute_uncertainty,
        uncertainty_k=args.uncertainty_k,
        benchmark=args.benchmark,
    )

    # Standard RBF (kernels × smooths)
    for kernel in args.rbf_kernels:
        for smooth in args.rbf_smooths:
            execute_rbf_pipeline(
                properties=args.properties,
                kernels=[kernel],
                smooths=[smooth],
                grid_size=args.grid_size,
                y_strain=args.y_strain,
                compute_uncertainty=args.compute_uncertainty,
                uncertainty_k=args.uncertainty_k,
                benchmark=args.benchmark,
            )

    # Kriging (models × nuggets)
    for model in args.kriging_models:
        for nugget in args.kriging_nuggets:
            execute_kriging_pipeline(
                properties=args.properties,
                variogram_models=[model],
                nuggets=[nugget],
                grid_size=args.grid_size,
                compute_uncertainty=args.compute_uncertainty,  # harmless flag; kriging uses native sigma
                uncertainty_k=args.uncertainty_k,              # not used by kriging pipeline
                benchmark=args.benchmark,
            )

    # Nearest
    execute_nearest_pipeline(
        properties=args.properties,
        grid_size=args.grid_size,
        y_strain=args.y_strain,
        compute_uncertainty=args.compute_uncertainty,
        uncertainty_k=args.uncertainty_k,
        benchmark=args.benchmark,
    )

    # Linear
    execute_linear_pipeline(
        properties=args.properties,
        grid_size=args.grid_size,
        y_strain=args.y_strain,
        compute_uncertainty=args.compute_uncertainty,
        uncertainty_k=args.uncertainty_k,
        benchmark=args.benchmark,
    )

    # Biharmonic (Clough–Tocher)
    execute_biharmonic_pipeline(
        properties=args.properties,
        grid_size=args.grid_size,
        compute_uncertainty=args.compute_uncertainty,
        uncertainty_k=args.uncertainty_k,
        benchmark=args.benchmark,
    )

    # RBF Compact-ish
    execute_rbf_compact_pipeline(
        properties=args.properties,
        grid_size=args.grid_size,
        y_strain=args.y_strain,
        function_type=args.rbf_compact_kernel,
        epsilon=args.rbf_compact_epsilon,
        smooth=args.rbf_compact_smooth,
        compute_uncertainty=args.compute_uncertainty,
        uncertainty_k=args.uncertainty_k,
        benchmark=args.benchmark,
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.method == "akima":
        execute_akima_pipeline(
            properties=args.properties,
            grid_size=args.grid_size,
            compute_uncertainty=args.compute_uncertainty,
            uncertainty_k=args.uncertainty_k,
            benchmark=args.benchmark,
        )

    elif args.method == "rbf":
        for kernel in args.rbf_kernels:
            for smooth in args.rbf_smooths:
                execute_rbf_pipeline(
                    properties=args.properties,
                    grid_size=args.grid_size,
                    kernels=[kernel],
                    smooths=[smooth],
                    compute_uncertainty=args.compute_uncertainty,
                    uncertainty_k=args.uncertainty_k,
                    benchmark=args.benchmark,
                )

    elif args.method == "kriging":
        for model in args.kriging_models:
            for nugget in args.kriging_nuggets:
                execute_kriging_pipeline(
                    properties=args.properties,
                    grid_size=args.grid_size,
                    variogram_models=[model],
                    nuggets=[nugget],
                    compute_uncertainty=args.compute_uncertainty,
                    uncertainty_k=args.uncertainty_k,
                    benchmark=args.benchmark,
                )

    elif args.method == "nearest":
        execute_nearest_pipeline(
            properties=args.properties,
            grid_size=args.grid_size,
            compute_uncertainty=args.compute_uncertainty,
            uncertainty_k=args.uncertainty_k,
            benchmark=args.benchmark,
        )

    elif args.method == "linear":
        execute_linear_pipeline(
            properties=args.properties,
            grid_size=args.grid_size,
            compute_uncertainty=args.compute_uncertainty,
            uncertainty_k=args.uncertainty_k,
            benchmark=args.benchmark,
        )

    elif args.method == "biharmonic":
        execute_biharmonic_pipeline(
            properties=args.properties,
            grid_size=args.grid_size,
            y_strain=args.y_strain,
            compute_uncertainty=args.compute_uncertainty,
            uncertainty_k=args.uncertainty_k,
            benchmark=args.benchmark,
        )

    elif args.method == "rbf_compact":
        execute_rbf_compact_pipeline(
            properties=args.properties,
            grid_size=args.grid_size,
            function_type=args.rbf_compact_kernel,
            epsilon=args.rbf_compact_epsilon,
            smooth=args.rbf_compact_smooth,
            compute_uncertainty=args.compute_uncertainty,
            uncertainty_k=args.uncertainty_k,
            benchmark=args.benchmark,
        )

    elif args.method == "all":
        _run_all_methods(args)

    else:
        raise ValueError(f"Unsupported method: {args.method}")


if __name__ == "__main__":
    main()