# ReEKF-SORT

ReEKF-SORT is a motion-only online tracker for non-linear motion and occlusion.
It extends a SORT-style tracking-by-detection pipeline with three components:

- a speed-angle extended Kalman filter state for causal motion prediction
- virtual observations generated only from pre-occlusion motion history
- confidence-continuity association, with motion-angle association retained only
  for ablation

## What BoxMOT Needs For ReEKF-SORT

- Detector only. ReID is not required.
- Axis-aligned bounding boxes only.
- The Python backend is the reference implementation used for paper results and ablations.
- Use the same detector cache for all ablation variants so changes measure the tracker only.

## Paper Final Configuration

The paper final `Ours` setting is the default no-angle config in
`boxmot/configs/trackers/reekfsort.yaml`:

| Parameter | Paper final default | Effect |
| --- | --- | --- |
| `use_virtual_observation` | `true` | Enables causal pseudo-observations while a track is unmatched. |
| `use_confidence_cost` | `true` | Enables confidence-continuity cost in association. |
| `use_angle_cost` | `false` | Disables motion-angle association in the final setting. |
| `lambda_conf` | `1.2` | Weight for confidence-continuity cost. |
| `lambda_angle` | `0.0` | Keeps the angle-cost term inactive. |
| `confidence_cost_mode` | `absolute` | Paper-final absolute confidence gap; other modes are ablations only. |
| `virtual_update_interval` | `6` | Injects a causal pseudo-observation every six unmatched frames. |
| `virtual_obs_noise_scale` | `10.0` | Prevents pseudo-observations from overpowering later real detections. |

`boxmot/configs/trackers/reekfsort_with_angle.yaml` preserves the with-angle/full
candidate for ablation only. It is not the paper final `Ours` configuration.

## Reproduction Commands

Datasets and detector weights are not included in the source package. Prepare
them before evaluation:

```bash
python scripts/reproduce/prepare_reekfsort_assets.py
```

Generate detections once per benchmark, then replay ReEKF-SORT from the cached detections:

```bash
boxmot generate --benchmark dancetrack --split val --detector yolox_x_dancetrack
boxmot eval --benchmark dancetrack --split val --tracker reekfsort --tracker-backend python

boxmot generate --benchmark mot17 --split ablation --detector yolox_x_mot17_ablation
boxmot eval --benchmark mot17 --split ablation --tracker reekfsort --tracker-backend python

boxmot generate --benchmark mot20 --split ablation --detector yolox_x_mot20_ablation
boxmot eval --benchmark mot20 --split ablation --tracker reekfsort --tracker-backend python
```

For MOT17 public detections, make the detection source explicit:

```bash
boxmot generate --benchmark mot17 --split ablation --detection-source public
boxmot eval --benchmark mot17 --split ablation --tracker reekfsort --detection-source public --tracker-backend python
```

Record the detector checkpoint URL, dataset split, BoxMOT commit, Python version,
CUDA/PyTorch versions, and the exact commands above in any paper artifact.

## Ablation Switches

The paper ablations are controlled by temporarily editing defaults in
`boxmot/configs/trackers/reekfsort.yaml`:

| Parameter | Default | Effect |
| --- | --- | --- |
| `use_virtual_observation` | `true` | Enables causal pseudo-observations while a track is unmatched. |
| `use_confidence_cost` | `true` | Enables confidence-continuity cost in association. |
| `use_angle_cost` | `false` | Disabled for the paper final no-angle default. |
| `lambda_conf` | `1.2` | Weight for confidence-continuity cost. |
| `lambda_angle` | `0.0` | Weight for motion-angle consistency cost. |

Recommended ablation matrix:

| Name | `use_virtual_observation` | `use_confidence_cost` | `use_angle_cost` |
| --- | --- | --- | --- |
| `reekfsort_motion_only` | `false` | `false` | `false` |
| `reekfsort_virtual_only` | `true` | `false` | `false` |
| `reekfsort_conf_only` | `false` | `true` | `false` |
| `reekfsort_angle_only` | `false` | `false` | `true` |
| `reekfsort_no_virtual` | `false` | `true` | `false` |
| `reekfsort_no_conf` | `true` | `false` | `false` |
| `reekfsort_no_angle` | `true` | `true` | `false` |
| `reekfsort_full` | `true` | `true` | `false` |
| `reekfsort_with_angle` | `true` | `true` | `true` |

The angle-cost branch is kept for controlled ablations. The final paper setting
uses virtual observation plus confidence cost without angle cost because the
with-angle/full candidate is not the final default tradeoff.

Report HOTA, DetA, AssA, IDF1, MOTA, ID switches, and FPS for each row.
DanceTrack is the primary stress test for non-linear motion and occlusion;
MOT17 and MOT20 are cross-benchmark checks. The current MOT20 result is a
negative result relative to OC-SORT and must be reported as a dense-scene
limitation rather than generalized as an improvement.

## Scripts

The reproducibility scripts under `scripts/reproduce/` run the benchmark matrix
with the same tracker implementation and cached detections:

```bash
python scripts/reproduce/prepare_reekfsort_assets.py
python scripts/reproduce/reekfsort_ablation.py --benchmark dancetrack --split val
python scripts/reproduce/reekfsort_ablation.py --benchmark mot17 --split ablation
python scripts/reproduce/reekfsort_ablation.py --benchmark mot20 --split ablation
python scripts/reproduce/reekfsort_sweeps.py --benchmark dancetrack --split val --sweep confidence_mode
python scripts/reproduce/reekfsort_sweeps.py --benchmark dancetrack --split val --sweep lambda_conf
```

On Windows, the full ablation matrix can be launched with:

```powershell
.\scripts\reproduce\reekfsort_ablation_all.ps1
```

Dataset and detector-weight sources are recorded in
[`docs/trackers/reekfsort_assets.md`](reekfsort_assets.md). ReEKF-SORT is
motion-only, so no ReID checkpoint is needed for these experiments.

::: boxmot.trackers.bbox.reekfsort.reekfsort.ReEkfSort
