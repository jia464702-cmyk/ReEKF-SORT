#!/usr/bin/env python3
"""Run the paper's targeted ReEKF-SORT parameter and confidence sweeps.

Generate/cache detections first. The script temporarily changes tracker defaults,
restores the original YAML in a finally block, and writes one JSON manifest per run.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


SWEEPS = {
    "confidence_mode": [
        {"confidence_cost_mode": "none", "lambda_conf": 0.0},
        {"confidence_cost_mode": "raw", "lambda_conf": 1.2},
        {"confidence_cost_mode": "absolute", "lambda_conf": 1.2},
        {"confidence_cost_mode": "normalized", "lambda_conf": 1.2},
    ],
    "lambda_conf": [
        {"lambda_conf": value} for value in (0.0, 0.4, 0.8, 1.2, 1.6, 2.0)
    ],
    "virtual_interval": [
        {"virtual_update_interval": value} for value in (1, 2, 4, 6)
    ],
    "virtual_interval_extension": [
        {"virtual_update_interval": 8},
        {"virtual_update_interval": 10},
        {"use_virtual_observation": False},
    ],
    "virtual_noise": [
        {"virtual_obs_noise_scale": value} for value in (1.0, 5.0, 10.0, 20.0, 50.0)
    ],
}

DATASET_THRESHOLDS = {"dancetrack": 0.6, "mot17": 0.6, "mot20": 0.6}

CONFIDENCE_RECORD_IDS = {
    "none": "DT-CONF-NONE-001",
    "raw": "DT-CONF-RAW-001",
    "absolute": "DT-CONF-ABS-001",
    "normalized": "DT-CONF-NORM-001",
}

LAMBDA_CONF_RECORD_IDS = {
    0.0: "DT-LCONF-000-001",
    0.4: "DT-LCONF-040-001",
    0.8: "DT-LCONF-080-001",
    1.2: "DT-LCONF-120-001",
    1.6: "DT-LCONF-160-001",
    2.0: "DT-LCONF-200-001",
}

VIRTUAL_INTERVAL_RECORD_IDS = {
    1: "DT-VINT-1-001",
    2: "DT-VINT-2-001",
    4: "DT-VINT-4-001",
    6: "DT-VINT-6-001",
    8: "DT-VINT-8-001",
    10: "DT-VINT-10-001",
}

VIRTUAL_NOISE_RECORD_IDS = {
    1.0: "DT-VNOISE-1-001",
    5.0: "DT-VNOISE-5-001",
    10.0: "DT-VNOISE-10-001",
    20.0: "DT-VNOISE-20-001",
    50.0: "DT-VNOISE-50-001",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def config_path() -> Path:
    return repo_root() / "boxmot" / "configs" / "trackers" / "reekfsort.yaml"


def apply_defaults(path: Path, values: dict[str, object]) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key, value in values.items():
        if key not in cfg:
            raise KeyError(f"Missing ReEKF-SORT config key: {key}")
        cfg[key]["default"] = value
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg


def slug(value: object) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def result_snapshot(root: Path) -> dict[Path, int]:
    """Return mtimes so a newly written result can be identified reliably."""
    if not root.exists():
        return {}
    return {p.resolve(): p.stat().st_mtime_ns for p in root.rglob("experiment_results.json")}


def find_updated_result(root: Path, before: dict[Path, int]) -> Path:
    changed = []
    for candidate in root.rglob("experiment_results.json"):
        resolved = candidate.resolve()
        mtime = candidate.stat().st_mtime_ns
        if before.get(resolved) != mtime:
            changed.append(candidate)
    if not changed:
        raise FileNotFoundError("Evaluation finished but no new experiment_results.json was found")
    return max(changed, key=lambda p: p.stat().st_mtime_ns)


def record_id(args: argparse.Namespace, values: dict[str, object], name: str) -> str:
    if args.benchmark == "dancetrack" and args.sweep == "confidence_mode":
        return CONFIDENCE_RECORD_IDS[str(values["confidence_cost_mode"])]
    if args.benchmark == "dancetrack" and args.sweep == "lambda_conf":
        return LAMBDA_CONF_RECORD_IDS[float(values["lambda_conf"])]
    if args.benchmark == "dancetrack" and args.sweep in {
        "virtual_interval",
        "virtual_interval_extension",
    }:
        if values.get("use_virtual_observation") is False:
            return "DT-VINT-OFF-001"
        return VIRTUAL_INTERVAL_RECORD_IDS[int(values["virtual_update_interval"])]
    if args.benchmark == "dancetrack" and args.sweep == "virtual_noise":
        return VIRTUAL_NOISE_RECORD_IDS[float(values["virtual_obs_noise_scale"])]
    return name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", required=True, choices=sorted(SWEEPS))
    parser.add_argument("--benchmark", required=True, choices=sorted(DATASET_THRESHOLDS))
    parser.add_argument("--split", required=True)
    parser.add_argument("--detector", default=None)
    parser.add_argument("--detection-source", choices=["public", "private"], default=None)
    parser.add_argument(
        "--project",
        default="runs",
        help="Cache and output root. Keep this as runs so existing dets_n_embs caches are reused.",
    )
    parser.add_argument("--device", default="0", help="Runtime device used if cache generation is needed.")
    parser.add_argument(
        "--confidence-mode-for-lambda",
        choices=("raw", "absolute", "normalized"),
        default="absolute",
        help="Confidence formulation held fixed during the lambda_conf sweep.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = config_path()
    original = path.read_text(encoding="utf-8")
    base = {
        "det_thresh": DATASET_THRESHOLDS[args.benchmark],
        "use_observation_centric": True,
        "use_ocr": True,
        "use_virtual_observation": True,
        "use_confidence_cost": True,
        "use_angle_cost": False,
        "lambda_angle": 0.0,
        "confidence_cost_mode": "absolute",
        "lambda_conf": 1.2,
        "virtual_update_interval": 6,
        "virtual_obs_noise_scale": 10.0,
    }
    if args.sweep == "lambda_conf":
        base["confidence_cost_mode"] = args.confidence_mode_for_lambda
    if args.sweep == "virtual_noise":
        # D1 plus boundary extension selected I_v=6 on DanceTrack-val.
        base["virtual_update_interval"] = 6
    try:
        for values in SWEEPS[args.sweep]:
            current = dict(base)
            current.update(values)
            label = "_".join(f"{key}-{slug(value)}" for key, value in values.items())
            name = f"{args.benchmark}_{args.sweep}_{label}"
            cfg = apply_defaults(path, current)
            cmd = [
                sys.executable,
                "-m",
                "boxmot.engine.cli",
                "eval",
                "--benchmark",
                args.benchmark,
                "--split",
                args.split,
                "--tracker",
                "reekfsort",
                "--tracker-backend",
                "python",
                "--project",
                args.project,
                "--name",
                name,
                "--exist-ok",
                "--device",
                args.device,
                "--show-timing",
            ]
            if args.detector:
                cmd.extend(["--detector", args.detector])
            if args.detection_source:
                cmd.extend(["--detection-source", args.detection_source])
            if args.dry_run:
                print(" ".join(cmd))
                continue

            output_dir = repo_root() / args.project
            output_dir.mkdir(parents=True, exist_ok=True)
            paper_dir = output_dir / "paper_records" / record_id(args, values, name)
            paper_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "experiment": name,
                "sweep": args.sweep,
                "benchmark": args.benchmark,
                "split": args.split,
                "command": cmd,
                "tracker_config": cfg,
            }
            manifest_path = paper_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            (paper_dir / "reekfsort_config.yaml").write_text(
                path.read_text(encoding="utf-8"), encoding="utf-8"
            )
            env = os.environ.copy()
            env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
            env.setdefault("BOXMOT_NO_REID", "1")
            before = result_snapshot(output_dir)
            subprocess.run(cmd, cwd=repo_root(), check=True, env=env)
            result_path = find_updated_result(output_dir, before)
            saved_result = paper_dir / "experiment_results.json"
            shutil.copy2(result_path, saved_result)
            manifest["source_result"] = str(result_path.relative_to(repo_root()))
            manifest["saved_result"] = str(saved_result.relative_to(repo_root()))
            manifest["status"] = "completed"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"Saved paper record: {saved_result.relative_to(repo_root())}")
    finally:
        path.write_text(original, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
