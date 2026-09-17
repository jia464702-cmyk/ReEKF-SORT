# Authoritative paper result records

This directory is the single public source of lightweight result summaries for
the frozen ReEKF-SORT configuration. Large datasets, detector weights, caches,
raw run directories, and challenge submission archives are intentionally kept
outside Git.

## Frozen configuration

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

## Included records

- `ablation_summary.csv` contains the currently verified MOT17 and MOT20
  fixed-split baseline/method pairs.
- DanceTrack is intentionally not listed until its complete final baseline and
  method rows, configuration provenance, and paired-sequence statistics can be
  published together. Values in `results/paper/` belong to an older sweep and
  must not be substituted for that final record.

Tracker FPS measures association/update throughput from cached detections; it
is not end-to-end detector-plus-tracker speed. The listed fixed-split metrics
are not official challenge test-server scores.

When official test evaluations become available, record the score, submission
identifier, submission date, and public result-page URL without replacing the
validation or ablation rows.
