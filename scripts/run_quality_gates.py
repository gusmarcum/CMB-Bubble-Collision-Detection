"""
Run project quality gates from one entrypoint.

This script is deliberately operational: it compiles the active scripts and runs
the physics, dataset, training-dry-run, and evaluation-output checks that guard
against the major scientific and ML failure modes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCRIPTS = [
    "scripts/phase2_signal_model.py",
    "scripts/phase2_physics_checks.py",
    "scripts/phase2_audit_dataset.py",
    "scripts/phase2_generate_training.py",
    "scripts/phase2_extract_smica_null_controls.py",
    "scripts/phase_dataset_utils.py",
    "scripts/phase3_train_unet.py",
    "scripts/phase3_evaluate_run.py",
    "scripts/phase3_audit_outputs.py",
    "scripts/phase3_template_baseline.py",
    "scripts/phase3_compare_screeners.py",
    "scripts/phase3_boundary_analysis.py",
    "scripts/phase3_template_fit_candidates.py",
    "scripts/phase3_score_null_controls.py",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run compile, physics, dataset, training, and evaluation quality gates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-h5", type=str, default="")
    parser.add_argument("--eval-dir", type=str, default="")
    parser.add_argument("--skip-train-dry-run", action="store_true")
    return parser.parse_args()


def run(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main():
    args = parse_args()
    py = sys.executable

    run([py, "-m", "py_compile", *DEFAULT_SCRIPTS])
    run([py, "scripts/phase2_physics_checks.py", "--output-json", "runs/phase2_physics_checks.json"])

    if args.data_h5:
        run([py, "scripts/phase2_audit_dataset.py", "--data-h5", args.data_h5])
        if not args.skip_train_dry_run:
            run(
                [
                    py,
                    "scripts/phase3_train_unet.py",
                    "--data-h5",
                    args.data_h5,
                    "--run-name",
                    "quality_gate_dry_run",
                    "--dry-run",
                    "--num-workers",
                    "0",
                ]
            )

    if args.eval_dir:
        run([py, "scripts/phase3_audit_outputs.py", "--eval-dir", args.eval_dir])

    print("\nQuality gates passed.")


if __name__ == "__main__":
    main()
