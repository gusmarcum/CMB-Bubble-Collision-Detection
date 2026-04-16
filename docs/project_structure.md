# Project Structure And Artifact Policy

This repository is a research pipeline. Keep source, documentation, and small reproducibility metadata in git. Keep generated datasets, checkpoints, score caches, Planck FITS maps, and run artifacts out of git.

## Canonical Source

- `scripts/phase1_explore.py`: Planck input inspection and patch-geometry checks.
- `scripts/phase2_signal_model.py`: Feeney-style bubble-collision template and signal-model utilities.
- `scripts/phase2_generate_training.py`: current Phase 2 training-data generator.
- `scripts/phase2_audit_dataset.py`: dataset provenance and leakage audit.
- `scripts/phase2_generate_stratified_validation.py`: independent stratified validation set generation.
- `scripts/phase2_extract_smica_null_controls.py`: real-map null-control extraction.
- `scripts/phase3_train_unet.py`: U-Net training harness.
- `scripts/phase3_evaluate_run.py`: model evaluation and artifact generation.
- `scripts/phase3_screen_and_verify.py`: structured candidate-output CLI.
- `scripts/phase3_template_baseline.py`: matched-template and classical baseline support.
- `scripts/phase3_real_sky_recalibration.py`: real-SMICA threshold recalibration.
- `scripts/phase3_threshold_volume_sweep.py`: threshold/false-positive-volume tradeoff analysis.
- `scripts/phase3_two_pass_policy.py`: loose ML proposal plus matched-template verifier analysis.

## Research Harnesses To Keep

These scripts encode tested research branches and should not be deleted just because one branch failed:

- `scripts/phase3_sensitivity_curve.py`
- `scripts/phase3_ml_gain_heatmap.py`
- `scripts/phase3_null_burden_matched_fpr.py`
- `scripts/phase3_real_sky_injection.py`
- `scripts/phase3_real_sky_smoothed_sensitivity.py`
- `scripts/phase3_cache_matched_filter_channel.py`
- `scripts/phase3_nside512_probe.py`
- `scripts/phase3_weak_family_breakdown.py`
- `scripts/phase3_ensemble_evaluate.py`
- `scripts/phase3_per_radius_threshold.py`

## Local Artifacts

The following are intentionally local and regenerable:

- `data/*.fits`: Planck map products.
- `data/training*/`: generated training datasets.
- `data/validation*/`: generated validation datasets.
- `runs/`: model checkpoints, eval outputs, score caches, preview images, and large HDF5 intermediates.

Only keep local artifacts that are active inputs for current experiments. Prefer copying compact decision notes into `docs/` before deleting large cached HDF5 files.
