# ReEKF-SORT Open Source Release

Paper title for The Visual Computer resubmission:

> Online multi-object occlusion tracking based on causal motion prediction and confidence-continuity association

ReEKF-SORT is a motion-only online multi-object tracker for non-linear pedestrian motion and occlusion. It keeps the SORT-style tracking-by-detection workflow, replaces the linear motion state with a speed-angle extended Kalman filter, and adds causal virtual observations plus confidence-continuity association.

This repository package contains source code, configs, documentation, and reproduction scripts only. Datasets, detector weights, run outputs, caches, and exported model files are intentionally not bundled.

## Paper Final Configuration

The paper final `Ours` configuration is the no-angle setting in `boxmot/configs/trackers/reekfsort.yaml`:

```yaml
confidence_cost_mode:
  default: absolute
lambda_conf:
  default: 1.2
virtual_update_interval:
  default: 6
virtual_obs_noise_scale:
  default: 10.0
use_virtual_observation:
  default: true
use_confidence_cost:
  default: true
use_angle_cost:
  default: false
lambda_angle:
  default: 0.0
```

`boxmot/configs/trackers/reekfsort_with_angle.yaml` preserves the with-angle/full candidate for ablation only. It is not the paper final `Ours` setting.

## Citation And License Status

`CITATION.cff` records software citation metadata for this release and uses the
The Visual Computer resubmission title above. Before a paper-final public
release, manually confirm the author order, affiliations, publication year,
DOI or arXiv URL, and final project GitHub URL. If the paper is not formally
published yet, keep the citation as submitted software/preprint metadata
instead of inventing a DOI or final publication record.

Preserve upstream project references when preparing the final manuscript or
artifact page: BoxMOT, TrackEval, YOLOX, DanceTrack, and MOTChallenge. This
release follows the upstream BoxMOT AGPL-3.0 license. Third-party components
retain their original licenses, including TrackEval under MIT and YOLOX under
Apache-2.0.

## Environment

Use Python 3.10 or newer. A minimal source checkout workflow is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements/reekfsort_eval.txt
python -m pip install yolox==0.3.0 --no-deps
```

The `yolox==0.3.0 --no-deps` step avoids pulling outdated YOLOX dependency pins over the evaluation environment. Install the CUDA-enabled PyTorch build that matches your machine before running GPU evaluation.

## Dataset Preparation

The release package does not include DanceTrack, MOT17, or MOT20 data. Prepare the paper evaluation splits with:

```bash
python scripts/reproduce/prepare_reekfsort_assets.py
```

The expected locations after preparation are:

- DanceTrack val: `boxmot/engine/eval/trackeval/data/test1/val`
- DanceTrack test: `boxmot/engine/eval/trackeval/data/test1/test`
- MOT17 ablation: `boxmot/engine/eval/trackeval/data/MOT17/ablation`
- MOT17 test: `boxmot/engine/eval/trackeval/data/MOT17/test`
- MOT20 ablation: `boxmot/engine/eval/trackeval/data/MOT20/ablation`
- MOT20 test: `boxmot/engine/eval/trackeval/data/MOT20/test`

Asset sources and expected local paths are documented in `docs/trackers/reekfsort_assets.md`.

Use official or stable dataset pages when preparing data:

| Benchmark | Required split | Official or stable source | Expected local path | Redistribution note |
| --- | --- | --- | --- | --- |
| DanceTrack | `val` | [Project](https://dancetrack.github.io/), [GitHub](https://github.com/DanceTrack/DanceTrack), [Hugging Face mirror](https://huggingface.co/datasets/noahcao/dancetrack) | `boxmot/engine/eval/trackeval/data/test1/val` | DanceTrack is for non-commercial research purposes only; do not redistribute it in this release. |
| MOT17 | `ablation` | [MOTChallenge/Codabench MOT17](https://www.codabench.org/competitions/10049/) | `boxmot/engine/eval/trackeval/data/MOT17/ablation` | Download from official challenge pages or documented mirrors; do not bundle data. |
| MOT20 | `ablation` | [MOTChallenge/Codabench MOT20](https://www.codabench.org/competitions/10050/) | `boxmot/engine/eval/trackeval/data/MOT20/ablation` | Download from official challenge pages or documented mirrors; do not bundle data. |

## Detector Weights

The release package does not include YOLOX checkpoints. Download or prepare the
detector weights before evaluation, place them under `models/`, and verify the
checksums:

| Weight file | Download source | SHA256 | Provenance |
| --- | --- | --- | --- |
| `models/yolox_x_dancetrack.pt` | `https://huggingface.co/Lekim89/yolox/resolve/main/yolox_x_dancetrack.pt` | `f76e036f872a57710d9aebdfad2730e1c0e78bc3805e0526fa4b04a9e3c1d13d` | Third-party/release reproduction YOLOX checkpoint; not bundled and not claimed as an official Megvii YOLOX checkpoint. |
| `models/yolox_x_MOT17_ablation.pt` | `https://huggingface.co/Lekim89/yolox/resolve/main/yolox_x_MOT17_ablation.pt` | `26cb8d2808664e5068a4c812d53becbc948b47fd6eacf2b45db049ab40c48b1a` | Third-party/release reproduction YOLOX checkpoint; not bundled and not claimed as an official Megvii YOLOX checkpoint. |
| `models/yolox_x_MOT17_test.pt` | `https://huggingface.co/Lekim89/yolox/resolve/main/yolox_x_MOT17_test.pt` | `e3945f3523fde1e107708aacd64dab0670c34e371d136e54587cac7a50d3cfba` | Test-split checkpoint used to build the recorded MOT17 submission; not bundled. |
| `models/yolox_x_MOT20_ablation.pt` | `https://huggingface.co/Lekim89/yolox/resolve/main/yolox_x_MOT20_ablation.pt` | `c8a49d9a58ab6dbc59e2f5daebb552513f6f5474acd9f62b47b4ef738e4900a3` | Third-party/release reproduction YOLOX checkpoint; not bundled and not claimed as an official Megvii YOLOX checkpoint. |
| `models/yolox_x_MOT20_test.pt` | `https://huggingface.co/Lekim89/yolox/resolve/main/yolox_x_MOT20_test.pt` | `021d7bc47fe20ae690007454cd2192df6237c2f4446ae92737a79523de89de64` | Test-split checkpoint used to build the recorded MOT20 submission; not bundled. |

If these URLs are not the final stable hosting location, publish the weights to
a controlled Hugging Face repository, GitHub Release, or Zenodo record and update
both the URLs and SHA256 values. ReEKF-SORT itself is motion-only, so no ReID
checkpoint is required for the reported experiments.

## Single Dataset Evaluation

Generate detections once, then evaluate ReEKF-SORT from cached detections:

```bash
boxmot generate --benchmark dancetrack --split val --detector yolox_x_dancetrack
boxmot eval --benchmark dancetrack --split val --tracker reekfsort --tracker-backend python --project runs/reekfsort_reproduce --name dancetrack_val --exist-ok

boxmot generate --benchmark mot17 --split ablation --detector yolox_x_mot17_ablation
boxmot eval --benchmark mot17 --split ablation --tracker reekfsort --tracker-backend python --project runs/reekfsort_reproduce --name mot17_ablation --exist-ok

boxmot generate --benchmark mot20 --split ablation --detector yolox_x_mot20_ablation
boxmot eval --benchmark mot20 --split ablation --tracker reekfsort --tracker-backend python --project runs/reekfsort_reproduce --name mot20_ablation --exist-ok
```

Set `BOXMOT_NO_REID=1` for the motion-only paper configuration. For no-GT test
submission generation, make the split-specific detector explicit so an
ablation checkpoint cannot be selected accidentally:

```bash
BOXMOT_NO_REID=1 python -m boxmot.engine.cli eval --benchmark dancetrack --split test --detector models/yolox_x_dancetrack.pt --tracker reekfsort --tracker-backend python --device 0 --project runs --name dancetrack_test --exist-ok --imgsz 800,1440 --batch-size 1 --no-auto-batch
BOXMOT_NO_REID=1 python -m boxmot.engine.cli eval --benchmark mot17 --split test --detector models/yolox_x_MOT17_test.pt --tracker reekfsort --tracker-backend python --device 0 --project runs --name mot17_test --exist-ok --imgsz 800,1440 --batch-size 1 --no-auto-batch
BOXMOT_NO_REID=1 python -m boxmot.engine.cli eval --benchmark mot20 --split test --detector models/yolox_x_MOT20_test.pt --tracker reekfsort --tracker-backend python --device 0 --project runs --name mot20_test --exist-ok --imgsz 800,1440 --batch-size 1 --no-auto-batch
```

These commands package MOTChallenge 10-column text files but do not calculate
official test metrics. Upload each generated archive to its corresponding
benchmark server and record the submission identifier and result-page URL.

For MOT17 public detections:

```bash
boxmot generate --benchmark mot17 --split ablation --detection-source public
boxmot eval --benchmark mot17 --split ablation --tracker reekfsort --detection-source public --tracker-backend python --project runs/reekfsort_reproduce --name mot17_public_ablation --exist-ok
```

## Batch Ablation

Run the full paper ablation matrix:

```bash
bash scripts/reproduce/reekfsort_ablation_all.sh
```

Run one benchmark or selected rows:

```bash
python scripts/reproduce/reekfsort_ablation.py --benchmark dancetrack --split val
python scripts/reproduce/reekfsort_ablation.py --benchmark mot17 --split ablation --only reekfsort_no_angle reekfsort_full
python scripts/reproduce/reekfsort_ablation.py --benchmark mot20 --split ablation --only reekfsort_motion_only reekfsort_no_angle
```

## Angle-Cost Ablation Conclusion

The angle-cost branch is retained for controlled ablation, but it is not enabled in the final paper configuration. The final `Ours` setting uses virtual observation plus confidence cost and disables angle cost because the with-angle/full candidate did not provide the final default tradeoff across the evaluated benchmarks.

## Results

The repository publishes compact, reviewable summaries in
[`results/paper`](results/paper/README.md). Full run directories, detector
caches, datasets, weights, and test-submission archives remain excluded.

| Benchmark | Comparison | HOTA | MOTA | IDF1 | IDSW | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| DanceTrack val | revised backbone | 51.098 | 88.823 | 50.399 | 2011 | controlled ablation reference |
| DanceTrack val | final no-angle configuration | 55.206 | 89.428 | 55.086 | 1556 | improves association over the revised backbone |
| MOT17 ablation | OC-SORT | 66.441 | 74.550 | 77.901 | 229 | external tracker baseline |
| MOT17 ablation | final no-angle configuration | 66.418 | 75.103 | 77.745 | 168 | similar HOTA with 61 fewer identity switches |
| MOT20 ablation | OC-SORT | 72.253 | 87.406 | 86.948 | 660 | dense-scene baseline |
| MOT20 ablation | final no-angle configuration | 71.009 | 87.358 | 85.149 | 839 | negative result; dense-scene limitation |

MOT20 is reported as a limitation rather than a positive result: the final
configuration lowers HOTA and IDF1 and increases identity switches relative to
OC-SORT. Timing values in the detailed CSV are tracker-only measurements with
detections loaded from cache, not end-to-end detector-plus-tracker speed.

Evaluation and ablation outputs are written under `runs/` by default:

- `runs/reekfsort_reproduce/` for single benchmark reproduction commands.
- `runs/reekfsort_ablation/` for `scripts/reproduce/reekfsort_ablation.py`.

These directories are generated artifacts and are excluded from the open-source release archive.

## Open Source Package

Generate the code-only release archive with:

```bash
bash scripts/reproduce/prepare_open_source_package.sh
```

The script writes:

- `dist/reekfsort_open_source_release.tar.gz`
- `dist/reekfsort_open_source_filelist.txt`
- `dist/reekfsort_open_source_checksums.sha256`
