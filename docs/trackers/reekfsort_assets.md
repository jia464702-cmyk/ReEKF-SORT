# ReEKF-SORT Data And Weight Sources

The open-source release package does not include datasets, detector weights,
run outputs, or caches. This page records the external assets expected by the
reproduction scripts.

## Datasets

| Benchmark | Split | Expected local path | Official or stable source | Helper download source | Redistribution note | Package status |
| --- | --- | --- | --- | --- | --- | --- |
| `dancetrack` | `val` | `boxmot/engine/eval/trackeval/data/test1` | [Project](https://dancetrack.github.io/), [GitHub](https://github.com/DanceTrack/DanceTrack), [Hugging Face mirror](https://huggingface.co/datasets/noahcao/dancetrack) | `https://huggingface.co/datasets/noahcao/dancetrack/resolve/main/test1.zip` | DanceTrack is for non-commercial research purposes only; do not redistribute it in this release. | `not included` |
| `mot17` | `ablation` | `boxmot/engine/eval/trackeval/data/MOT17` | [MOTChallenge/Codabench MOT17](https://www.codabench.org/competitions/10049/) | `hf://Lekim89/mot17-parquet/images/train` | Download from official challenge pages or documented mirrors; do not bundle data. | `not included` |
| `mot20` | `ablation` | `boxmot/engine/eval/trackeval/data/MOT20` | [MOTChallenge/Codabench MOT20](https://www.codabench.org/competitions/10050/) | `hf://Lekim89/MOT20/ablation` | Download from official challenge pages or documented mirrors; do not bundle data. | `not included` |

## Detector Weights

| Benchmark | Split | Detector | Expected local path | Source URL | SHA256 | Provenance | Image size | Confidence | Package status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `dancetrack` | `val` | `yolox_x_dancetrack` | `models/yolox_x_dancetrack.pt` | `https://huggingface.co/Lekim89/yolox/resolve/main/yolox_x_dancetrack.pt` | `f76e036f872a57710d9aebdfad2730e1c0e78bc3805e0526fa4b04a9e3c1d13d` | Third-party/release reproduction YOLOX checkpoint; not bundled and not claimed as an official Megvii YOLOX checkpoint. | `(800, 1440)` | `0.01` | `not included` |
| `mot17` | `ablation` | `yolox_x_mot17_ablation` | `models/yolox_x_MOT17_ablation.pt` | `https://huggingface.co/Lekim89/yolox/resolve/main/yolox_x_MOT17_ablation.pt` | `26cb8d2808664e5068a4c812d53becbc948b47fd6eacf2b45db049ab40c48b1a` | Third-party/release reproduction YOLOX checkpoint; not bundled and not claimed as an official Megvii YOLOX checkpoint. | `(800, 1440)` | `0.01` | `not included` |
| `mot20` | `ablation` | `yolox_x_mot20_ablation` | `models/yolox_x_MOT20_ablation.pt` | `https://huggingface.co/Lekim89/yolox/resolve/main/yolox_x_MOT20_ablation.pt` | `c8a49d9a58ab6dbc59e2f5daebb552513f6f5474acd9f62b47b4ef738e4900a3` | Third-party/release reproduction YOLOX checkpoint; not bundled and not claimed as an official Megvii YOLOX checkpoint. | `(800, 1440)` | `0.01` | `not included` |

If these URLs are not the final stable hosting location, publish the weights to
a controlled Hugging Face repository, GitHub Release, or Zenodo record and update
both the URLs and SHA256 values.

## ReID Weights

ReEKF-SORT is a motion-only tracker. No ReID weights are required for the reported ReEKF-SORT experiments.

## Commands

```bash
python scripts/reproduce/prepare_reekfsort_assets.py
python scripts/reproduce/reekfsort_ablation.py --benchmark dancetrack --split val
python scripts/reproduce/reekfsort_ablation.py --benchmark mot17 --split ablation
python scripts/reproduce/reekfsort_ablation.py --benchmark mot20 --split ablation
```
