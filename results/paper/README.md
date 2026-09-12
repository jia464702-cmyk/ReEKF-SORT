# Paper result records

This directory contains lightweight, auditable summaries derived from the
frozen paper records. It intentionally excludes datasets, detector weights,
detection caches, full run directories, and challenge submission archives.

## Files

- `ablation_summary.csv` records validation/ablation metrics used in the paper.
- `test_submission_artifacts.csv` records the verified shape and checksum of
  each test submission. Official server metrics are still pending and must not
  be inferred from these files.

## Frozen final configuration

```yaml
confidence_cost_mode: absolute
lambda_conf: 1.2
virtual_update_interval: 6
virtual_obs_noise_scale: 10.0
use_virtual_observation: true
use_confidence_cost: true
use_angle_cost: false
lambda_angle: 0.0
```

The `final_no_angle` rows use this configuration. The with-angle variant is an
ablation only. Reported FPS is association/update throughput with detections
loaded from cache; it is not end-to-end detector-plus-tracker throughput.

## Interpretation boundary

- DanceTrack validation supports the confidence-continuity contribution, but
  the confidence-only row has slightly higher HOTA than the combined final row.
- MOT17 shows a substantial identity-switch reduction relative to OC-SORT while
  HOTA remains effectively unchanged.
- MOT20 is a negative result for the final configuration relative to OC-SORT and
  is disclosed as a dense-scene limitation.

When official test evaluations become available, append the official score,
submission identifier, submission date, and public result-page URL. Do not
replace validation or ablation rows with test claims.
