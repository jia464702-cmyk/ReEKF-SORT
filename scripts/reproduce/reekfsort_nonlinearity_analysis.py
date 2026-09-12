#!/usr/bin/env python3
"""Stratify DanceTrack GT trajectories by motion-direction nonlinearity.

Official HOTA/AssA/IDF1 are dataset-level metrics and cannot be decomposed by a
post-hoc GT trajectory group without oracle-labeling tracker predictions. This
script therefore reports transparent trajectory diagnostics: coverage, dominant
tracker-ID consistency, ID switches per 1000 GT-visible frames, fragmentation,
and reappearance recovery.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from reekfsort_occlusion_analysis import (
    DEFAULT_METHODS,
    build_gt_index,
    evaluate_event,
    find_events,
    frame_assignments,
    load_mot,
    resolve_path,
)


GROUPS = ("low", "medium", "high")


def wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def trajectory_score(observations: dict[int, tuple[np.ndarray, bool]]) -> float | None:
    visible = sorted((frame, bbox) for frame, (bbox, is_visible) in observations.items() if is_visible)
    velocities = []
    for (frame_a, box_a), (frame_b, box_b) in zip(visible, visible[1:]):
        dt = frame_b - frame_a
        if dt <= 0:
            continue
        center_a = box_a[:2] + box_a[2:4] * 0.5
        center_b = box_b[:2] + box_b[2:4] * 0.5
        velocity = (center_b - center_a) / dt
        if np.linalg.norm(velocity) > 1e-6:
            velocities.append(float(math.atan2(velocity[1], velocity[0])))
    if len(velocities) < 2:
        return None
    changes = [abs(wrap_angle(b - a)) for a, b in zip(velocities, velocities[1:])]
    return float(np.mean(changes)) if changes else None


def quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q, method="linear"))


def group_for(score: float, low_boundary: float, high_boundary: float) -> str:
    if score <= low_boundary:
        return "low"
    if score <= high_boundary:
        return "medium"
    return "high"


def parse_methods(values: list[str] | None) -> dict[str, Path]:
    raw = DEFAULT_METHODS if not values else dict(item.split("=", 1) for item in values)
    return {name: resolve_path(path) for name, path in raw.items()}


def safe_rate(numerator: int, denominator: int) -> float | None:
    return round(100.0 * numerator / denominator, 3) if denominator else None


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-root", default="boxmot/engine/eval/trackeval/data/test1/val")
    parser.add_argument("--method", action="append", help="Optional NAME=RESULT_DIRECTORY override")
    parser.add_argument("--visibility-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--post-window", type=int, default=5)
    parser.add_argument("--output", default="runs/paper_records/DT-NONLINEAR-001")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gt_root = resolve_path(args.gt_root)
    output = resolve_path(args.output)
    methods = parse_methods(args.method)
    sequence_dirs = sorted(path for path in gt_root.iterdir() if (path / "gt" / "gt.txt").is_file())
    if not sequence_dirs:
        raise FileNotFoundError(f"No sequence GT files found under {gt_root}")

    gt_by_sequence = {}
    identities = {}
    events = []
    score_rows = []
    for sequence_dir in sequence_dirs:
        sequence = sequence_dir.name
        gt_by_frame, by_identity = build_gt_index(
            load_mot(sequence_dir / "gt" / "gt.txt"), args.visibility_threshold
        )
        gt_by_sequence[sequence] = gt_by_frame
        for gt_id, observations in by_identity.items():
            key = (sequence, gt_id)
            identities[key] = observations
            score = trajectory_score(observations)
            if score is not None:
                score_rows.append({"sequence": sequence, "gt_id": gt_id, "score_rad": score})
        events.extend(find_events(sequence, by_identity))
    if len(score_rows) < 3:
        raise RuntimeError("Too few trajectories have at least two valid direction changes")

    scores = [float(row["score_rad"]) for row in score_rows]
    low_boundary, high_boundary = quantile(scores, 0.33), quantile(scores, 0.67)
    trajectory_groups = {}
    for row in score_rows:
        group = group_for(float(row["score_rad"]), low_boundary, high_boundary)
        row["score_deg"] = round(math.degrees(float(row["score_rad"])), 6)
        row["group"] = group
        trajectory_groups[(row["sequence"], int(row["gt_id"]))] = group

    output.mkdir(parents=True, exist_ok=True)
    print(
        f"Trajectories: {len(score_rows)}; boundaries: "
        f"{math.degrees(low_boundary):.3f} / {math.degrees(high_boundary):.3f} deg",
        flush=True,
    )
    summary_rows = []
    per_track_rows = []
    for method, result_dir in methods.items():
        missing = [seq.name for seq in sequence_dirs if not (result_dir / f"{seq.name}.txt").is_file()]
        if missing:
            raise FileNotFoundError(f"{method}: missing {len(missing)} files, first={missing[0]}")
        print(f"[method] {method}: {result_dir}", flush=True)
        method_assignments = {}
        for index, sequence_dir in enumerate(sequence_dirs, start=1):
            sequence = sequence_dir.name
            print(f"  [{index:02d}/{len(sequence_dirs):02d}] {sequence}", flush=True)
            tracker_data = load_mot(result_dir / f"{sequence}.txt")
            method_assignments[sequence] = frame_assignments(
                gt_by_sequence[sequence], tracker_data, args.iou_threshold
            )

        method_track_rows = []
        for row in score_rows:
            sequence, gt_id = row["sequence"], int(row["gt_id"])
            observations = identities[(sequence, gt_id)]
            visible_frames = sorted(frame for frame, (_, visible) in observations.items() if visible)
            matched_ids = [method_assignments[sequence].get(frame, {}).get(gt_id) for frame in visible_frames]
            present_ids = [identity for identity in matched_ids if identity is not None]
            switches = 0
            previous = None
            segments = 0
            in_segment = False
            for identity in matched_ids:
                if identity is None:
                    in_segment = False
                    continue
                if not in_segment:
                    segments += 1
                    in_segment = True
                if previous is not None and identity != previous:
                    switches += 1
                previous = identity
            dominant = max(Counter(present_ids).values()) if present_ids else 0
            track_row = {
                "method": method,
                **row,
                "visible_frames": len(visible_frames),
                "matched_frames": len(present_ids),
                "dominant_id_frames": dominant,
                "id_switches": switches,
                "fragments": max(segments - 1, 0),
            }
            method_track_rows.append(track_row)
            per_track_rows.append(track_row)

        event_records = []
        for event in events:
            group = trajectory_groups.get((event.sequence, event.gt_id))
            if group is None:
                continue
            event_records.append(
                {
                    "group": group,
                    **evaluate_event(event, method_assignments[event.sequence], args.post_window),
                }
            )
        for group in GROUPS:
            tracks = [row for row in method_track_rows if row["group"] == group]
            grouped_events = [row for row in event_records if row["group"] == group]
            gt_frames = sum(int(row["visible_frames"]) for row in tracks)
            matched = sum(int(row["matched_frames"]) for row in tracks)
            dominant = sum(int(row["dominant_id_frames"]) for row in tracks)
            switches = sum(int(row["id_switches"]) for row in tracks)
            fragments = sum(int(row["fragments"]) for row in tracks)
            eligible = sum(int(row["eligible_before"]) for row in grouped_events)
            recovered = sum(int(row["identity_recovered"]) for row in grouped_events)
            summary_rows.append(
                {
                    "method": method,
                    "nonlinearity_group": group,
                    "track_count": len(tracks),
                    "gt_visible_frames": gt_frames,
                    "matched_frames": matched,
                    "coverage_pct": safe_rate(matched, gt_frames),
                    "dominant_id_consistency_pct": safe_rate(dominant, matched),
                    "id_switches": switches,
                    "id_switches_per_1000_gt_frames": round(1000.0 * switches / gt_frames, 3),
                    "fragments": fragments,
                    "fragments_per_1000_gt_frames": round(1000.0 * fragments / gt_frames, 3),
                    "reappearance_events": len(grouped_events),
                    "eligible_reappearance_events": eligible,
                    "identity_recovered": recovered,
                    "recovery_accuracy_pct": safe_rate(recovered, eligible),
                }
            )

    metadata = {
        "experiment_id": "DT-NONLINEAR-001",
        "score": "mean absolute wrapped change between consecutive GT velocity directions",
        "boundaries_rad": {"q33": low_boundary, "q67": high_boundary},
        "boundaries_deg": {
            "q33": math.degrees(low_boundary),
            "q67": math.degrees(high_boundary),
        },
        "visibility_threshold": args.visibility_threshold,
        "iou_threshold": args.iou_threshold,
        "post_window": args.post_window,
        "sequence_count": len(sequence_dirs),
        "eligible_trajectory_count": len(score_rows),
        "excluded_short_trajectories": len(identities) - len(score_rows),
        "methods": {name: str(path) for name, path in methods.items()},
        "metric_warning": (
            "These are transparent stratified diagnostics, not official HOTA/AssA/IDF1. "
            "Official metrics are not decomposed by post-hoc GT trajectory groups."
        ),
        "command": sys.argv,
    }
    write_csv(output / "nonlinearity_summary.csv", summary_rows)
    write_csv(output / "nonlinearity_trajectories.csv", per_track_rows)
    write_csv(output / "nonlinearity_scores.csv", score_rows)
    (output / "nonlinearity_results.json").write_text(
        json.dumps({"metadata": metadata, "summary": summary_rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# DanceTrack-val motion nonlinearity analysis",
        "",
        f"Boundaries (degrees): low <= {math.degrees(low_boundary):.3f}, "
        f"medium <= {math.degrees(high_boundary):.3f}, high above it.",
        "",
        "These values are stratified diagnostics, not official HOTA/AssA/IDF1.",
        "",
        "| Method | Group | Tracks | Coverage % | ID consistency % | IDSW/1000 GT frames | Recovery % |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        fmt = lambda value: "NA" if value is None else f"{float(value):.3f}"
        lines.append(
            f"| {row['method']} | {row['nonlinearity_group']} | {row['track_count']} | "
            f"{fmt(row['coverage_pct'])} | {fmt(row['dominant_id_consistency_pct'])} | "
            f"{fmt(row['id_switches_per_1000_gt_frames'])} | {fmt(row['recovery_accuracy_pct'])} |"
        )
    (output / "nonlinearity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {output.relative_to(resolve_path('.'))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

