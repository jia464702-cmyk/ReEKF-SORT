#!/usr/bin/env python3
"""Evaluate identity recovery after GT-defined occlusion intervals on DanceTrack.

Occlusion is defined before seeing method results: a GT identity is considered
visible when it has a valid person annotation whose visibility is at least the
configured threshold (0.5 by default). Missing or lower-visibility frames form
an occlusion interval. A reappearance event is the first visible frame after
such an interval, excluding track starts and ends.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


BINS = ((1, 5, "1-5"), (6, 10, "6-10"), (11, 20, "11-20"), (21, None, ">20"))

DEFAULT_METHODS = {
    "ocsort": "runs/mot/dancetrack/yolox_x_dancetrack_noreid_ocsort",
    "speed_angle_ekf": "runs/mot/dancetrack/yolox_x_dancetrack_noreid_reekfsort_2",
    "revised_backbone": "runs/mot/dancetrack/yolox_x_dancetrack_noreid_reekfsort_3",
    "pseudo_only": "runs/mot/dancetrack/yolox_x_dancetrack_noreid_reekfsort_5",
    "paper_final": "runs/mot/dancetrack/yolox_x_dancetrack_noreid_reekfsort_27",
}


@dataclass(frozen=True)
class Event:
    sequence: str
    gt_id: int
    before_frame: int
    reappearance_frame: int
    occlusion_length: int
    bin_name: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root() / path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mot(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        return np.empty((0, 10), dtype=float)
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    if data.shape[1] < 6:
        raise ValueError(f"Expected at least 6 MOT columns in {path}, got {data.shape[1]}")
    return data


def valid_gt_rows(data: np.ndarray) -> np.ndarray:
    keep = np.ones(len(data), dtype=bool)
    if data.shape[1] > 6:
        keep &= data[:, 6] > 0
    if data.shape[1] > 7 and np.any(data[:, 7] == 1):
        keep &= data[:, 7] == 1
    return data[keep]


def visibility_of(row: np.ndarray) -> float:
    return float(row[8]) if row.shape[0] > 8 else 1.0


def bin_for(length: int) -> str:
    for lower, upper, name in BINS:
        if length >= lower and (upper is None or length <= upper):
            return name
    raise ValueError(f"Invalid occlusion length: {length}")


def build_gt_index(data: np.ndarray, visibility_threshold: float):
    rows = valid_gt_rows(data)
    by_frame: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    by_identity: dict[int, dict[int, tuple[np.ndarray, bool]]] = defaultdict(dict)
    for row in rows:
        frame, identity = int(row[0]), int(row[1])
        bbox = np.asarray(row[2:6], dtype=float)
        visible = visibility_of(row) >= visibility_threshold
        by_identity[identity][frame] = (bbox, visible)
        if visible:
            by_frame[frame].append((identity, bbox))
    return by_frame, by_identity


def find_events(sequence: str, by_identity) -> list[Event]:
    events: list[Event] = []
    for gt_id, observations in by_identity.items():
        frames = sorted(observations)
        if not frames:
            continue
        first, last = frames[0], frames[-1]
        previous_visible: int | None = None
        for frame in range(first, last + 1):
            current = observations.get(frame)
            visible = current is not None and current[1]
            if not visible:
                continue
            if previous_visible is not None and frame - previous_visible > 1:
                length = frame - previous_visible - 1
                events.append(
                    Event(sequence, gt_id, previous_visible, frame, length, bin_for(length))
                )
            previous_visible = frame
    return events


def xywh_iou(gt_boxes: np.ndarray, tracker_boxes: np.ndarray) -> np.ndarray:
    if len(gt_boxes) == 0 or len(tracker_boxes) == 0:
        return np.zeros((len(gt_boxes), len(tracker_boxes)), dtype=float)
    gt_xy2 = gt_boxes[:, :2] + np.maximum(gt_boxes[:, 2:4], 0.0)
    tr_xy2 = tracker_boxes[:, :2] + np.maximum(tracker_boxes[:, 2:4], 0.0)
    tl = np.maximum(gt_boxes[:, None, :2], tracker_boxes[None, :, :2])
    br = np.minimum(gt_xy2[:, None, :], tr_xy2[None, :, :])
    wh = np.maximum(br - tl, 0.0)
    inter = wh[..., 0] * wh[..., 1]
    gt_area = np.maximum(gt_boxes[:, 2], 0.0) * np.maximum(gt_boxes[:, 3], 0.0)
    tr_area = np.maximum(tracker_boxes[:, 2], 0.0) * np.maximum(tracker_boxes[:, 3], 0.0)
    union = gt_area[:, None] + tr_area[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def frame_assignments(gt_by_frame, tracker_data: np.ndarray, iou_threshold: float):
    tracker_by_frame: dict[int, list[np.ndarray]] = defaultdict(list)
    for row in tracker_data:
        tracker_by_frame[int(row[0])].append(row)
    assignments: dict[int, dict[int, int]] = {}
    for frame, gt_items in gt_by_frame.items():
        tr_rows = tracker_by_frame.get(frame, [])
        if not tr_rows:
            assignments[frame] = {}
            continue
        gt_ids = [item[0] for item in gt_items]
        gt_boxes = np.asarray([item[1] for item in gt_items], dtype=float)
        tracker_rows = np.asarray(tr_rows, dtype=float)
        similarities = xywh_iou(gt_boxes, tracker_rows[:, 2:6])
        gt_indices, tracker_indices = linear_sum_assignment(1.0 - similarities)
        matched = {}
        for gi, ti in zip(gt_indices, tracker_indices):
            if similarities[gi, ti] >= iou_threshold:
                matched[gt_ids[gi]] = int(tracker_rows[ti, 1])
        assignments[frame] = matched
    return assignments


def evaluate_event(event: Event, assignments, post_window: int) -> dict[str, object]:
    before_id = assignments.get(event.before_frame, {}).get(event.gt_id)
    first_id = assignments.get(event.reappearance_frame, {}).get(event.gt_id)
    eligible = before_id is not None
    recovered = eligible and first_id is not None and first_id == before_id
    previous = before_id
    switches = 0
    matched_frames = 0
    for frame in range(event.reappearance_frame, event.reappearance_frame + post_window):
        current = assignments.get(frame, {}).get(event.gt_id)
        if current is None:
            continue
        matched_frames += 1
        if previous is not None and current != previous:
            switches += 1
        previous = current
    return {
        **asdict(event),
        "eligible_before": int(eligible),
        "first_frame_matched": int(first_id is not None),
        "identity_recovered": int(recovered),
        "post_window_matched_frames": matched_frames,
        "post_window_id_switches": switches,
        "stable_recovery": int(recovered and switches == 0),
    }


def safe_rate(numerator: int, denominator: int) -> float | None:
    return round(100.0 * numerator / denominator, 3) if denominator else None


def aggregate(method: str, records: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for _, _, name in BINS:
        subset = [row for row in records if row["bin_name"] == name]
        event_count = len(subset)
        eligible = sum(int(row["eligible_before"]) for row in subset)
        first_matched = sum(int(row["first_frame_matched"]) for row in subset)
        recovered = sum(int(row["identity_recovered"]) for row in subset)
        stable = sum(int(row["stable_recovery"]) for row in subset)
        switches = sum(int(row["post_window_id_switches"]) for row in subset)
        output.append(
            {
                "method": method,
                "occlusion_bin": name,
                "event_count": event_count,
                "eligible_before": eligible,
                "first_frame_matched": first_matched,
                "first_frame_match_rate_pct": safe_rate(first_matched, event_count),
                "identity_recovered": recovered,
                "recovery_accuracy_pct": safe_rate(recovered, eligible),
                "stable_recovery": stable,
                "stable_recovery_rate_pct": safe_rate(stable, eligible),
                "post_window_id_switches": switches,
                "id_switches_per_100_events": round(100.0 * switches / event_count, 3)
                if event_count
                else None,
            }
        )
    return output


def parse_methods(values: list[str] | None) -> dict[str, Path]:
    raw = DEFAULT_METHODS if not values else dict(item.split("=", 1) for item in values)
    return {name: resolve_path(path) for name, path in raw.items()}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: list[dict[str, object]], metadata: dict) -> None:
    lines = [
        "# DanceTrack-val occlusion-length identity recovery",
        "",
        ("GT visibility threshold: `{}`; IoU matching threshold: `{}`; post-reappearance "
         "window: `{}` frames.").format(
            metadata["visibility_threshold"], metadata["iou_threshold"], metadata["post_window"]
        ),
        "",
        "| Method | Occlusion | Events | Eligible | First-match % | ID recovery % | Stable recovery % | ID changes / 100 events |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        fmt = lambda value: "NA" if value is None else f"{float(value):.3f}"
        lines.append(
            f"| {row['method']} | {row['occlusion_bin']} | {row['event_count']} | "
            f"{row['eligible_before']} | {fmt(row['first_frame_match_rate_pct'])} | "
            f"{fmt(row['recovery_accuracy_pct'])} | {fmt(row['stable_recovery_rate_pct'])} | "
            f"{fmt(row['id_switches_per_100_events'])} |"
        )
    lines.extend(
        [
            "",
            "`post_window_id_switches` is an explicitly defined event-window identity-change count, "
            "not TrackEval's global CLEAR IDSW. Prediction-state center error is omitted because the "
            "saved MOT files do not contain latent predicted states.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gt-root",
        default="boxmot/engine/eval/trackeval/data/test1/val",
        help="Directory containing DanceTrack sequence folders.",
    )
    parser.add_argument(
        "--method",
        action="append",
        help="Optional NAME=RESULT_DIRECTORY override; repeat for every method.",
    )
    parser.add_argument("--visibility-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--post-window", type=int, default=5)
    parser.add_argument("--output", default="runs/paper_records/DT-OCCLUSION-001")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gt_root = resolve_path(args.gt_root)
    output = resolve_path(args.output)
    methods = parse_methods(args.method)
    sequence_dirs = sorted(path for path in gt_root.iterdir() if (path / "gt" / "gt.txt").is_file())
    if not sequence_dirs:
        raise FileNotFoundError(f"No sequence GT files found under {gt_root}")

    gt_indices = {}
    all_events: list[Event] = []
    for sequence_dir in sequence_dirs:
        sequence = sequence_dir.name
        gt_data = load_mot(sequence_dir / "gt" / "gt.txt")
        gt_by_frame, by_identity = build_gt_index(gt_data, args.visibility_threshold)
        gt_indices[sequence] = gt_by_frame
        all_events.extend(find_events(sequence, by_identity))
    if not all_events:
        raise RuntimeError(
            "No occlusion events found. Inspect GT visibility values and rerun with an explicitly "
            "pre-registered --visibility-threshold."
        )
    print(
        f"Loaded {len(sequence_dirs)} sequences and {len(all_events)} GT reappearance events.",
        flush=True,
    )

    output.mkdir(parents=True, exist_ok=True)
    all_event_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    provenance = {}
    for method, result_dir in methods.items():
        missing = [seq.name for seq in sequence_dirs if not (result_dir / f"{seq.name}.txt").is_file()]
        if missing:
            raise FileNotFoundError(f"{method}: missing {len(missing)} result files, first={missing[0]}")
        provenance[method] = {"result_dir": str(result_dir), "sequences": {}}
        method_records = []
        print(f"[method] {method}: {result_dir}", flush=True)
        for sequence_index, sequence_dir in enumerate(sequence_dirs, start=1):
            sequence = sequence_dir.name
            print(
                f"  [{sequence_index:02d}/{len(sequence_dirs):02d}] {sequence}",
                flush=True,
            )
            result_file = result_dir / f"{sequence}.txt"
            tracker_data = load_mot(result_file)
            assignments = frame_assignments(gt_indices[sequence], tracker_data, args.iou_threshold)
            records = [
                {"method": method, **evaluate_event(event, assignments, args.post_window)}
                for event in all_events
                if event.sequence == sequence
            ]
            method_records.extend(records)
            provenance[method]["sequences"][sequence] = file_sha256(result_file)
        all_event_rows.extend(method_records)
        summary_rows.extend(aggregate(method, method_records))

    metadata = {
        "experiment_id": "DT-OCCLUSION-001",
        "gt_root": str(gt_root),
        "sequence_count": len(sequence_dirs),
        "event_count": len(all_events),
        "visibility_threshold": args.visibility_threshold,
        "iou_threshold": args.iou_threshold,
        "post_window": args.post_window,
        "bins": [name for _, _, name in BINS],
        "methods": provenance,
        "definitions": {
            "first_frame_match_rate": "reappearance GT box has any IoU-matched tracker output",
            "recovery_accuracy": "reappearance tracker ID equals the tracker ID matched before occlusion",
            "stable_recovery": "identity recovered and no matched-ID change in the post window",
            "post_window_id_switches": "matched tracker-ID changes after reappearance; not global CLEAR IDSW",
        },
        "command": sys.argv,
    }
    write_csv(output / "occlusion_summary.csv", summary_rows)
    write_csv(output / "occlusion_events.csv", all_event_rows)
    (output / "occlusion_results.json").write_text(
        json.dumps({"metadata": metadata, "summary": summary_rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(output / "occlusion_report.md", summary_rows, metadata)
    print(f"Sequences: {len(sequence_dirs)}; GT reappearance events: {len(all_events)}")
    print(f"Saved: {output.relative_to(repo_root())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

