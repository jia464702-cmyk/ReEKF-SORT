# ReEKF-SORT

**Online Multi-Object Tracking with Causal Pseudo-Observation Regularization and Confidence-Continuity Association**

ReEKF-SORT is a motion-only, online multi-object tracker designed for pedestrian tracking under occlusion and ambiguous motion-only association. It follows the tracking-by-detection paradigm and extends a SORT/OC-SORT-style pipeline with:

- a speed-angle extended Kalman filter (EKF) motion state;
- causal pseudo-observations generated only from pre-occlusion history;
- a continuous confidence-consistency cost for association;
- no ReID model and no additional training.

This repository contains source code, tracker configurations, tests, documentation, and reproduction scripts. Datasets, detector checkpoints, detection caches, generated submissions, and large experiment outputs are intentionally excluded.

> **Paper status:** manuscript in preparation for journal submission. Official test-server metrics should be treated as pending until the corresponding submission IDs and result pages are archived.

## Highlights

- **Motion-only online tracking:** no appearance embedding or ReID checkpoint is required.
- **Confidence-continuity association:** detection-to-track confidence changes are incorporated as a continuous association cost.
- **Causal pseudo-observations:** unmatched tracks are updated without using future frames.
- **MOT17 identity-switch reduction:** on the fixed MOT17 ablation split, ID switches decrease from 229 to 168 relative to the OC-SORT baseline using the same cached detections.
- **Reproducibility:** final parameters, detector checksums, ablation scripts, no-GT submission packaging, manifests, and unit tests are included.
- **Limitations reported openly:** the current method produces a negative result on dense MOT20 scenes and is slower than OC-SORT.

## Paper-final configuration

The paper-final `Ours` setting is defined in [`boxmot/configs/trackers/reekfsort.yaml`](boxmot/configs/trackers/reekfsort.yaml).

| Parameter | Final value | Purpose |
| --- | ---: | --- |
| `confidence_cost_mode` | `absolute` | Absolute confidence gap between a detection and a predicted track. |
| `lambda_conf` | `1.2` | Confidence-continuity cost weight. |
| `virtual_update_interval` | `6` | Interval between causal pseudo-observation updates while unmatched. |
| `virtual_obs_noise_scale` | `10.0` | Makes pseudo-observations less certain than real detections. |
| `use_virtual_observation` | `true` | Enables causal pseudo-observations. |
| `use_confidence_cost` | `true` | Enables confidence-continuity association. |
| `use_angle_cost` | `false` | Disables the explicit angle association term in the final method. |
| `lambda_angle` | `0.0` | Keeps the angle-cost term inactive. |
| `delta_t` | `3` | Observation-centric motion-history window. |
| `inertia` | `0.1` | Observation-centric direction-consistency weight. |
| `det_thresh` | `0.6` | Frozen tracker detection threshold. |
| `max_age` | `30` | Maximum unmatched lifetime. |
| `min_hits` | `3` | Minimum confirmation hits. |
| `iou_threshold` | `0.3` | Association IoU threshold. |

The EKF still models motion direction internally. “Angle disabled” means that the additional angle term is disabled in the association cost; it does not remove the speed-angle EKF state.

[`boxmot/configs/trackers/reekfsort_with_angle.yaml`](boxmot/configs/trackers/reekfsort_with_angle.yaml) is retained only for a controlled angle-cost ablation. It is not the paper-final configuration.

## Repository layout

```text
boxmot/trackers/bbox/reekfsort/   ReEKF-SORT implementation
boxmot/configs/trackers/          Final and ablation configurations
boxmot/configs/benchmarks/        Dataset and detector profiles
scripts/reproduce/                Preparation, evaluation, sweep, and packaging scripts
tests/unit/                       ReEKF-SORT and submission-packaging tests
docs/trackers/                    Method and server-reproduction documentation
paper_results/                    Small, reviewable paper-result records (when included)
```

## Environment

Python 3.11 is recommended. Install a CUDA-enabled PyTorch build compatible with your machine before GPU evaluation.

```bash
git clone https://github.com/jia464702-cmyk/ReEKF-SORT.git
cd ReEKF-SORT

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements/reekfsort_eval.txt
python -m pip install yolox==0.3.0 --no-deps
```

The final `--no-deps` command prevents the old dependency pins published by YOLOX 0.3.0 from replacing the tested ONNX packages in the evaluation environment.

Verify the environment:

```bash
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())"
python -m boxmot.engine.cli --help
```

## Datasets

The repository does not redistribute DanceTrack, MOT17, or MOT20.

| Benchmark | Paper split | Expected location | Source |
| --- | --- | --- | --- |
| DanceTrack | `val` | `boxmot/engine/eval/trackeval/data/test1/val` | [Project](https://dancetrack.github.io/), [repository](https://github.com/DanceTrack/DanceTrack), or the documented mirror |
| MOT17 | fixed `ablation` | `boxmot/engine/eval/trackeval/data/MOT17/ablation` | [MOTChallenge/Codabench MOT17](https://www.codabench.org/competitions/10049/) |
| MOT20 | fixed `ablation` | `boxmot/engine/eval/trackeval/data/MOT20/ablation` | [MOTChallenge/Codabench MOT20](https://www.codabench.org/competitions/10050/) |

Review the dataset licenses and then prepare the documented assets:

```bash
python scripts/reproduce/prepare_reekfsort_assets.py --accept-dataset-terms
```

If the datasets and weights were prepared manually, verify them without downloading anything:

```bash
python scripts/reproduce/prepare_reekfsort_assets.py --verify-only
```

Offline MOT ablation preparation is also available:

```bash
python scripts/reproduce/prepare_mot_ablation_offline.py --benchmark mot17
python scripts/reproduce/prepare_mot_ablation_offline.py --benchmark mot20
python scripts/reproduce/prepare_mot_ablation_offline.py --benchmark both --verify-only
```

## Detector checkpoints

ReEKF-SORT itself is motion-only, so no ReID checkpoint is used in the reported experiments. Detector checkpoints are not bundled.

| Checkpoint | SHA256 |
| --- | --- |
| `models/yolox_x_dancetrack.pt` | `f76e036f872a57710d9aebdfad2730e1c0e78bc3805e0526fa4b04a9e3c1d13d` |
| `models/yolox_x_MOT17_ablation.pt` | `26cb8d2808664e5068a4c812d53becbc948b47fd6eacf2b45db049ab40c48b1a` |
| `models/yolox_x_MOT20_ablation.pt` | `c8a49d9a58ab6dbc59e2f5daebb552513f6f5474acd9f62b47b4ef738e4900a3` |
| `models/yolox_x_MOT17_test.pt` | `e3945f3523fde1e107708aacd64dab0670c34e371d136e54587cac7a50d3cfba` |
| `models/yolox_x_MOT20_test.pt` | `021d7bc47fe20ae690007454cd2192df6237c2f4446ae92737a79523de89de64` |

The configured download URLs and provenance notes are maintained in the benchmark YAML files and [`docs/trackers/reekfsort_assets.md`](docs/trackers/reekfsort_assets.md). Verify every downloaded checkpoint before evaluation.

## Reproduction

All reported tracker comparisons must reuse the same detector cache. The commands below disable ReID explicitly.

### 1. Generate detections once

```bash
BOXMOT_NO_REID=1 python -m boxmot.engine.cli generate \
  --benchmark dancetrack --split val \
  --detector yolox_x_dancetrack --device 0

BOXMOT_NO_REID=1 python -m boxmot.engine.cli generate \
  --benchmark mot17 --split ablation \
  --detector yolox_x_mot17_ablation --device 0

BOXMOT_NO_REID=1 python -m boxmot.engine.cli generate \
  --benchmark mot20 --split ablation \
  --detector yolox_x_mot20_ablation --device 0
```

### 2. Evaluate the final tracker from cache

```bash
BOXMOT_NO_REID=1 python -m boxmot.engine.cli eval \
  --benchmark dancetrack --split val \
  --tracker reekfsort --tracker-backend python \
  --device 0 --project runs --name DT-VAL-REEKFSORT-FINAL --exist-ok --show-timing

BOXMOT_NO_REID=1 python -m boxmot.engine.cli eval \
  --benchmark mot17 --split ablation \
  --tracker reekfsort --tracker-backend python \
  --device 0 --project runs --name M17-ABL-REEKFSORT-FINAL --exist-ok --show-timing

BOXMOT_NO_REID=1 python -m boxmot.engine.cli eval \
  --benchmark mot20 --split ablation \
  --tracker reekfsort --tracker-backend python \
  --device 0 --project runs --name M20-ABL-REEKFSORT-FINAL --exist-ok --show-timing
```

The timing reported for cached evaluation is tracker association/update time only. It excludes YOLOX inference and must not be presented as end-to-end system FPS.

### 3. Controlled ablations

Run only the distinct rows needed by the paper:

```bash
BOXMOT_NO_REID=1 python scripts/reproduce/reekfsort_ablation.py \
  --benchmark dancetrack --split val --project runs --device 0 \
  --only reekfsort_revised_backbone \
         reekfsort_revised_virtual_only \
         reekfsort_revised_conf_only \
         reekfsort_revised_full \
         reekfsort_revised_with_angle
```

Replace the benchmark and split with `mot17/ablation` or `mot20/ablation` for cross-dataset validation. Each run writes a manifest containing the command and complete tracker configuration.

Avoid reporting duplicate aliases as independent experiments. In particular, `conf_only`/`no_virtual`, `virtual_only`/`no_conf`, and `no_angle`/`full` may describe identical switch combinations in older experiment scripts.

## Verified local ablation results

The following values were obtained with the same cached YOLOX detections for each baseline/method pair. ReID was disabled. These are local fixed-split results, not official MOTChallenge test-server results.

| Dataset | Method | HOTA | MOTA | IDF1 | AssA | DetA | IDSW | Tracker FPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MOT17 ablation | OC-SORT | 66.441 | 74.550 | 77.901 | 69.112 | 64.237 | 229 | 243.1 |
| MOT17 ablation | ReEKF-SORT | 66.418 | 75.103 | 77.745 | 68.662 | 64.634 | 168 | 140.9 |
| MOT20 ablation | OC-SORT | 72.253 | 87.406 | 86.948 | 71.292 | 73.413 | 660 | 44.1 |
| MOT20 ablation | ReEKF-SORT | 71.009 | 87.358 | 85.149 | 69.037 | 73.244 | 839 | 9.5 |

The tracker-FPS values above were measured before the disabled-angle fast path
introduced in version 0.1.1. That optimization does not change the reported
accuracy metrics, but current-code speed must be remeasured before citing an
updated runtime result.

On MOT17, ReEKF-SORT reduces identity switches from 229 to 168: 61 fewer switches, or 26.6%. Aggregate HOTA and IDF1 remain approximately level and should not be described as large improvements.

MOT20 is a negative result. In the dense MOT20 ablation split, ReEKF-SORT decreases HOTA and IDF1, increases identity switches, and is substantially slower. This suggests that repeated pseudo-observation updates and the current Python association implementation are not yet suitable for the most crowded scenes.

Small authoritative paper-result records are stored under `paper_results/`. The older `results/paper/` directory is retained only as a historical snapshot and must not be cited as the final configuration. Large caches, datasets, weights, raw runs, and submission archives must remain outside Git.

## Official test submissions

The repository supports MOTChallenge 10-column submission packaging for splits without ground truth. Preparing a submission does not produce official metrics; the ZIP must still be uploaded to the corresponding benchmark server.

Prepare locally downloaded test data:

```bash
python scripts/reproduce/prepare_mot_test_offline.py --benchmark mot17
python scripts/reproduce/prepare_mot_test_offline.py --benchmark mot20
python scripts/reproduce/prepare_mot_test_offline.py --benchmark both --verify-only
```

Generate MOT17 and MOT20 submissions with the split-specific detector checkpoints:

```bash
BOXMOT_NO_REID=1 python -m boxmot.engine.cli eval \
  --benchmark mot17 --split test \
  --detector models/yolox_x_MOT17_test.pt \
  --tracker reekfsort --tracker-backend python \
  --device 0 --project runs --name M17-TEST-REEKFSORT-FINAL \
  --exist-ok --show-timing --imgsz 800,1440 --batch-size 1 --no-auto-batch

BOXMOT_NO_REID=1 python -m boxmot.engine.cli eval \
  --benchmark mot20 --split test \
  --detector models/yolox_x_MOT20_test.pt \
  --tracker reekfsort --tracker-backend python \
  --device 0 --project runs --name M20-TEST-REEKFSORT-FINAL \
  --exist-ok --show-timing --imgsz 800,1440 --batch-size 1 --no-auto-batch
```

Before publishing test claims, archive:

- the exact submission ZIP and SHA256;
- `submission_manifest.json` and `submission_result.json`;
- the official submission ID and submission date;
- the official result-page URL or screenshot;
- the server-reported metrics.

Do not infer official test metrics from the number of generated rows.

## Tests

Run the core release tests:

```bash
python -m pytest \
  tests/unit/test_no_gt_submission.py \
  tests/unit/test_reekfsort.py -q
```

Verify that the vendored TrackEval package is imported from this checkout:

```bash
python -c "from boxmot.engine.eval.metrics.run_mot_challenge import trackeval; print(trackeval.__file__)"
```

The printed path should resolve inside this repository rather than an unrelated globally installed TrackEval package.

## Known limitations

- ReEKF-SORT currently supports axis-aligned bounding boxes only.
- It is motion-only and may fail when long occlusion cannot be resolved from causal motion history.
- The final parameters were selected on validation data and frozen across MOT17/MOT20 cross-dataset experiments; they are not claimed to be universally optimal.
- Dense MOT20 scenes expose both accuracy and runtime limitations.
- Reported cached-evaluation FPS is tracker-only, not end-to-end detector-plus-tracker throughput.
- Official test-server results remain pending until submission records are published.

## Open-source package

Create a code-only release archive with:

```bash
bash scripts/reproduce/prepare_open_source_package.sh
```

The packaging script excludes datasets, detector weights, caches, generated submissions, large result files, personal paths, and common temporary artifacts. It writes the release archive, file list, and SHA256 checksums under `dist/`.

## Upstream projects and license

ReEKF-SORT is implemented on top of BoxMOT and uses components or interfaces from BoxMOT, TrackEval, YOLOX, DanceTrack, and MOTChallenge. Preserve their citations and licenses when redistributing or publishing derived work.

This repository follows the upstream BoxMOT **AGPL-3.0** license. Vendored third-party components retain their original licenses, including TrackEval under MIT and YOLOX under Apache-2.0. See [`LICENSE`](LICENSE) and the relevant third-party directories for details.

## Citation

The manuscript is not yet formally published. Do not invent a DOI or final bibliographic record. Until an accepted version is available, cite the software repository and the current manuscript metadata in [`CITATION.cff`](CITATION.cff).

```bibtex
@software{reekfsort2026,
  title  = {ReEKF-SORT: Online Multi-Object Tracking with Causal Pseudo-Observation Regularization and Confidence-Continuity Association},
  author = {Ma, Jianhong and Liu, Jia and Jin, Yan and Chen, Yongxia},
  year   = {2026},
  url    = {https://github.com/jia464702-cmyk/ReEKF-SORT}
}
```

Before a paper-final release, confirm the author order, affiliations, year, manuscript status, and any DOI or preprint URL against the current manuscript.
