#!/usr/bin/env python3
"""Fair tracker-only timing on the cached DanceTrack-val detections.

The benchmark excludes detector inference, ReID, image decoding, cache loading,
tracker construction, and disk I/O from the timed region. Each method/repeat is
run in a fresh process so absolute peak RSS is comparable.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from boxmot.trackers.tracker_zoo import create_tracker, get_tracker_config

try:
    import resource
except ImportError:  # Windows has no POSIX resource module; measurements run on Linux.
    resource = None


METHODS = ("ocsort", "speed_angle_ekf", "ekf_pseudo", "paper_final")
METHOD_NOTES = {
    "ocsort": "OC-SORT Python baseline",
    "speed_angle_ekf": "speed-angle EKF only; OCM/OCR, pseudo observation and confidence cost disabled",
    "ekf_pseudo": "revised backbone plus pseudo observation; confidence and angle costs disabled",
    "paper_final": "frozen paper method: absolute confidence, lambda=1.2, Iv=6, kappa=10, angle cost off",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def yaml_defaults(tracker: str) -> dict:
    with get_tracker_config(tracker).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return {name: spec["default"] for name, spec in raw.items()}


def method_spec(name: str) -> tuple[str, dict]:
    if name == "ocsort":
        return "ocsort", yaml_defaults("ocsort")
    cfg = yaml_defaults("reekfsort")
    cfg.update(
        confidence_cost_mode="absolute",
        lambda_conf=1.2,
        virtual_update_interval=6,
        virtual_obs_noise_scale=10.0,
        use_angle_cost=False,
        lambda_angle=0.0,
    )
    if name == "speed_angle_ekf":
        cfg.update(
            use_observation_centric=False,
            use_ocr=False,
            use_virtual_observation=False,
            use_confidence_cost=False,
        )
    elif name == "ekf_pseudo":
        cfg.update(
            use_observation_centric=True,
            use_ocr=True,
            use_virtual_observation=True,
            use_confidence_cost=False,
        )
    elif name == "paper_final":
        cfg.update(
            use_observation_centric=True,
            use_ocr=True,
            use_virtual_observation=True,
            use_confidence_cost=True,
        )
    else:
        raise ValueError(f"Unknown method: {name}")
    return "reekfsort", cfg


def find_cache(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
    else:
        path = repo_root() / "runs" / "dets_n_embs" / "dancetrack" / "val" / "yolox_x_dancetrack" / "dets"
    files = sorted(path.glob("*.npy"))
    if not path.is_dir() or len(files) != 25:
        raise FileNotFoundError(f"Expected 25 DanceTrack-val detection caches under {path}; found {len(files)}")
    return path


def load_sequences(cache_dir: Path) -> list[tuple[str, list[np.ndarray]]]:
    sequences = []
    for path in sorted(cache_dir.glob("*.npy")):
        arr = np.load(path).astype(np.float32, copy=False)
        if arr.ndim != 2 or arr.shape[1] < 7:
            raise ValueError(f"Unexpected detection cache shape {arr.shape}: {path}")
        frame_ids = arr[:, 0].astype(np.int64)
        max_frame = int(frame_ids.max()) if len(frame_ids) else 0
        frames: list[np.ndarray] = []
        start = 0
        for frame_id in range(1, max_frame + 1):
            while start < len(arr) and frame_ids[start] < frame_id:
                start += 1
            end = start
            while end < len(arr) and frame_ids[end] == frame_id:
                end += 1
            frames.append(np.asarray(arr[start:end, 1:7], dtype=np.float32))
            start = end
        sequences.append((path.stem, frames))
    return sequences


def make_tracker(method: str):
    tracker_type, cfg = method_spec(method)
    return create_tracker(
        tracker_type=tracker_type,
        evolve_param_dict=cfg,
        per_class=False,
        tracker_backend="python",
    )


def run_worker(method: str, cache_dir: Path, warmup: int, output: Path) -> None:
    if resource is None:
        raise RuntimeError("Efficiency measurements require Linux or another POSIX platform")
    sequences = load_sequences(cache_dir)
    img = np.empty((1080, 1920, 3), dtype=np.uint8)
    warmup_frames = [dets for _, frames in sequences for dets in frames[:warmup]][:warmup]
    warmup_tracker = make_tracker(method)
    for dets in warmup_frames:
        warmup_tracker.update(dets, img)
    del warmup_tracker
    gc.collect()

    latencies_ms: list[float] = []
    detection_count = 0
    sequence_count = 0
    for _, frames in sequences:
        tracker = make_tracker(method)
        sequence_count += 1
        for dets in frames:
            detection_count += len(dets)
            start_ns = time.perf_counter_ns()
            tracker.update(dets, img)
            latencies_ms.append((time.perf_counter_ns() - start_ns) / 1_000_000.0)
        del tracker

    total_ms = float(sum(latencies_ms))
    result = {
        "method": method,
        "frames": len(latencies_ms),
        "sequences": sequence_count,
        "detections": detection_count,
        "warmup_frames": len(warmup_frames),
        "total_timed_ms": total_ms,
        "fps": 1000.0 * len(latencies_ms) / total_ms,
        "latency_mean_ms": float(np.mean(latencies_ms)),
        "latency_std_ms": float(np.std(latencies_ms, ddof=1)),
        "latency_p50_ms": float(np.percentile(latencies_ms, 50)),
        "latency_p95_ms": float(np.percentile(latencies_ms, 95)),
        "latency_p99_ms": float(np.percentile(latencies_ms, 99)),
        "peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_parent(args: argparse.Namespace) -> None:
    root = repo_root()
    cache_dir = find_cache(args.cache_dir)
    output_dir = (root / args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    worker_dir = output_dir / "worker_results"
    rows: list[dict] = []
    base_env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        base_env[key] = str(args.threads)

    for repeat in range(1, args.repeats + 1):
        offset = (repeat - 1) % len(METHODS)
        order = METHODS[offset:] + METHODS[:offset]
        for method in order:
            worker_output = worker_dir / f"{method}_repeat{repeat}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--method",
                method,
                "--cache-dir",
                str(cache_dir),
                "--warmup",
                str(args.warmup),
                "--worker-output",
                str(worker_output),
            ]
            print(f"[repeat {repeat}/{args.repeats}] {method}", flush=True)
            completed = subprocess.run(command, env=base_env, text=True, capture_output=True)
            if completed.returncode != 0:
                print(completed.stdout)
                print(completed.stderr, file=sys.stderr)
                raise SystemExit(f"Efficiency worker failed for {method}, repeat {repeat}")
            result = json.loads(worker_output.read_text(encoding="utf-8"))
            result["repeat"] = repeat
            result["method_note"] = METHOD_NOTES[method]
            rows.append(result)
            print(
                f"  FPS={result['fps']:.3f}, mean={result['latency_mean_ms']:.3f} ms, "
                f"P95={result['latency_p95_ms']:.3f} ms, peak RSS={result['peak_rss_mib']:.1f} MiB",
                flush=True,
            )

    summary_rows = []
    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        summary_rows.append(
            {
                "method": method,
                "repeats": len(group),
                "frames_per_repeat": group[0]["frames"],
                "detections_per_repeat": group[0]["detections"],
                "fps_mean": statistics.mean(row["fps"] for row in group),
                "fps_std": statistics.stdev(row["fps"] for row in group),
                "latency_mean_ms": statistics.mean(row["latency_mean_ms"] for row in group),
                "latency_p95_ms_mean": statistics.mean(row["latency_p95_ms"] for row in group),
                "latency_p95_ms_max": max(row["latency_p95_ms"] for row in group),
                "peak_rss_mib_mean": statistics.mean(row["peak_rss_mib"] for row in group),
                "peak_rss_mib_max": max(row["peak_rss_mib"] for row in group),
                "requires_reid": False,
                "extra_training": False,
                "definition": METHOD_NOTES[method],
            }
        )

    write_csv(output_dir / "efficiency_repeats.csv", rows)
    write_csv(output_dir / "efficiency_summary.csv", summary_rows)
    metadata = {
        "experiment_id": "DT-EFFICIENCY-001",
        "protocol": "tracker.update only; actual cached DanceTrack-val detections; fresh worker per method/repeat",
        "excluded": ["detector", "ReID", "image decoding", "cache loading", "disk I/O", "tracker construction"],
        "warmup_frames": args.warmup,
        "repeats": args.repeats,
        "threads": args.threads,
        "image_shape": [1080, 1920, 3],
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cache_dir": str(cache_dir),
        "cache_files": [{"name": p.name, "sha256": sha256(p)} for p in sorted(cache_dir.glob("*.npy"))],
        "tracker_config_sha256": {
            "ocsort": sha256(get_tracker_config("ocsort")),
            "reekfsort": sha256(get_tracker_config("reekfsort")),
        },
        "script_sha256": sha256(Path(__file__).resolve()),
        "method_configurations": {method: method_spec(method)[1] for method in METHODS},
    }
    payload = {"metadata": metadata, "repeats": rows, "summary": summary_rows}
    (output_dir / "efficiency_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# ReEKF-SORT tracker-only efficiency",
        "",
        f"Actual DanceTrack-val cached detections; {args.warmup} warmup frames; {args.repeats} fresh-process repeats; {args.threads} CPU thread(s).",
        "Detector, ReID, image decoding, disk I/O, cache loading, and tracker construction are excluded.",
        "",
        "| Method | FPS mean ± std | Mean ms/frame | Mean P95 ms | Max peak RSS MiB |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {row['fps_mean']:.3f} ± {row['fps_std']:.3f} | "
            f"{row['latency_mean_ms']:.3f} | {row['latency_p95_ms_mean']:.3f} | {row['peak_rss_mib_max']:.1f} |"
        )
    lines += ["", "All methods use the Python backend and require neither ReID nor extra training."]
    (output_dir / "efficiency_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir")
    parser.add_argument("--output", default="runs/paper_records/DT-EFFICIENCY-001")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.worker:
        if not parsed.method or not parsed.worker_output:
            raise SystemExit("Worker mode requires --method and --worker-output")
        run_worker(parsed.method, find_cache(parsed.cache_dir), parsed.warmup, Path(parsed.worker_output))
    else:
        if parsed.repeats < 3:
            raise SystemExit("At least three repeats are required by the paper protocol")
        run_parent(parsed)
