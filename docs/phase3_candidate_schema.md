# Phase 3 Candidate Artifact Schema

Phase 3 emits candidate tables from `scripts/phase3_screen_and_verify.py`. The table is a two-branch handoff product, not a discovery statistic.

## Operating Modes

- `union`: emit a candidate when either the broad proposal branch or the boundary-aware verifier passes. This is proposal mode. It intentionally favors recall and must be filtered downstream.
- `both`: emit a candidate only when both branches pass. This is high-confidence shortlist mode. It favors purity and null control at the cost of recall.
- `proposal`: emit only original proposal-branch candidates.
- `verifier`: emit only boundary-aware verifier candidates.

## Core Fields

- `rank`: one-indexed rank after sorting by `rank_score` descending.
- `sample_index`: row index in the source HDF5 dataset.
- `rank_score`: operational ranking score combining proposal score, verifier score, template-fit improvement, and consensus bonus. It is not calibrated probability.
- `proposal_pass`: whether the broad original V4 branch emitted a candidate at its frozen threshold.
- `verifier_pass`: whether the boundary-aware branch emitted a candidate at its frozen threshold.
- `risk_tag`: categorical routing label. Current values are `priority_consensus`, `boundary_only_low_null`, `proposal_only_template_supported`, and `proposal_only_high_null_risk`.
- `truth_label`: synthetic validation truth label when available. Real-map screening outputs should treat this as unavailable or zero depending on source.

## Sky And Patch Provenance

- `patch_center_glon_deg`, `patch_center_glat_deg`: Galactic coordinates of the extracted patch center.
- `candidate_glon_deg`, `candidate_glat_deg`: estimated Galactic coordinates of the candidate center.
- `candidate_dx_deg`, `candidate_dy_deg`: candidate offset from patch center in local patch coordinates.
- `radius_est_deg`: candidate angular-size estimate from the selected branch mask.
- `coord_pool_idx`: coordinate-pool index used to reproduce extraction.
- `cmb_realization_idx`: synthetic CMB realization index when applicable.
- `background_id`: stable background/provenance identifier.

## Branch Scores

- `proposal_score_max`, `proposal_score_mean`, `proposal_positive_fraction`: original V4 branch mask statistics.
- `proposal_threshold`: frozen threshold used by the proposal branch.
- `verifier_score_max`, `verifier_score_mean`, `verifier_positive_fraction`: boundary-aware branch mask statistics.
- `verifier_threshold`: frozen threshold used by the verifier branch.

## Template-Fit Handoff

- `template_fit_status`: template-fit status for emitted candidates when a template-fit JSONL is provided.
- `template_delta_chi2_vs_plane_null`: improvement of the local Feeney-style template fit over a plane/null nuisance model. This is a handoff diagnostic, not Bayesian evidence.
- `template_theta_crit_fit_deg`, `template_z0_fit`, `template_zcrit_fit`: fitted template parameters for downstream inspection.

## Truth Fields For Synthetic Evaluation

- `truth_theta_crit_deg`, `truth_z0`, `truth_zcrit`, `truth_edge_sigma_deg`: injected signal parameters when available.

## Guardrails

- Do not compare `union` mode directly against FPR-capped single-branch results as if it were the same operating point.
- Do not interpret `rank_score` or branch scores as posterior probabilities.
- Do not treat a candidate row as a detection claim. It is input to classical/statistical follow-up.
