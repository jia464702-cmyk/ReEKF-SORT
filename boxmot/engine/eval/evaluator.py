from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Optional

import boxmot.utils.rich.ui as ui
from boxmot.data.benchmark import (
    COCO_CLASSES,
    _ordered_benchmark_eval_class_names,
    build_gt_class_remap,
    configure_benchmark_runtime,
    eval_init,
    load_benchmark_cfg_from_args,
    prepare_aabb_eval_gt,
    resolve_eval_box_type,
)
from boxmot.data.cache import (
    AppendableNpyWriter,
    _collect_seq_info,
    _existing_cache_path,
    _existing_embedding_cache_path,
    _load_embedding_cache_array,
    _load_numeric_cache_array,
    _max_frame_id,
    _saved_detection_column_count,
)
from boxmot.detectors import get_runtime_detector_cfg
from boxmot.engine.eval.cache import generate_dets_embs_batched, run_generate_dets_embs
from boxmot.engine.eval.replay import process_sequence, run_generate_mot_results
from boxmot.engine.workflows.reporting import extract_summary, timing_summary_from_stats
from boxmot.engine.workflows.results import ValidationResult
from boxmot.utils import (
    BENCHMARK_CONFIGS,
)
from boxmot.utils import (
    logger as LOGGER,
)
from boxmot.configs.benchmark import (
    ensure_benchmark_detector_model,
    ensure_benchmark_reid_model,
    load_benchmark_cfg,
    should_use_benchmark_detector,
    should_use_benchmark_reid,
)
from boxmot.utils.checks import RequirementsChecker
from boxmot.engine.eval.metrics.results import (
    _filter_obb_trackeval_results,
    _known_trackeval_class_names,
    _select_plot_metrics_data,
    log_trackeval_report,
    parse_mot_results,
    render_trackeval_report,
)
from boxmot.engine.eval.metrics.trackeval import (
    _load_obb_gt_matrix,
    trackeval_aabb,
    trackeval_obb,
)
from boxmot.utils.misc import resolve_model_path, suppress_boxmot_logs
from boxmot.engine.eval.plots import MetricsPlotter
from boxmot.utils.rich.eval_reporting import (
    EVAL_EVALUATE_STEP,
    EVAL_GENERATE_STEP,
    EVAL_SETUP_STEP,
    EVAL_TRACK_STEP,
    EvalWorkflowReporter,
    _build_eval_workflow_fields,
)
from boxmot.utils.rich.pipeline import PipelineTracker
from boxmot.utils.timing import TimingStats

_EVAL_DEPENDENCIES_READY = False

__all__ = [
    "AppendableNpyWriter",
    "_configure_benchmark_runtime",
    "_ensure_eval_dependencies",
    "_existing_cache_path",
    "_existing_embedding_cache_path",
    "_load_benchmark_cfg",
    "_load_embedding_cache_array",
    "_load_numeric_cache_array",
    "_load_obb_gt_matrix",
    "_max_frame_id",
    "_ordered_benchmark_eval_class_names",
    "_saved_detection_column_count",
    "_select_plot_metrics_data",
    "apply_class_remap",
    "eval_setup",
    "generate_dets_embs_batched",
    "main",
    "parse_mot_results",
    "process_sequence",
    "run_eval",
    "run_generate_dets_embs",
    "run_generate_mot_results",
    "run_trackeval",
]


def _ensure_eval_dependencies() -> None:
    global _EVAL_DEPENDENCIES_READY
    if _EVAL_DEPENDENCIES_READY:
        return
    checker = RequirementsChecker()
    checker.check_packages(("ultralytics",))
    _EVAL_DEPENDENCIES_READY = True


def _load_benchmark_cfg(args: argparse.Namespace) -> dict:
    return load_benchmark_cfg_from_args(args)


def _resolve_eval_box_type(args: argparse.Namespace, bench_cfg: Optional[dict] = None) -> str:
    return resolve_eval_box_type(args, bench_cfg)


def _is_no_gt_split(args: argparse.Namespace, cfg: Optional[dict] = None) -> bool:
    """Return whether the selected benchmark split intentionally has no ground truth."""
    cfg = _load_benchmark_cfg(args) if cfg is None else cfg
    if not isinstance(cfg, dict):
        return False
    scopes = [cfg]
    for key in ("dataset", "benchmark"):
        nested = cfg.get(key)
        if isinstance(nested, dict):
            scopes.append(nested)
    no_gt_splits = {
        str(value).lower()
        for scope in scopes
        for value in (scope.get("no_gt_splits") or [])
    }
    return str(getattr(args, "split", "")).lower() in no_gt_splits


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_no_gt_submission(args: argparse.Namespace) -> dict:
    """Validate tracker outputs and create an official 10-column submission archive."""
    exp_dir = Path(args.exp_dir)
    _, seq_info = _collect_seq_info(args.source)
    expected = set(seq_info)
    actual = {path.stem for path in exp_dir.glob("*.txt")}
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise RuntimeError(f"Submission sequence mismatch; missing={missing}, unexpected={unexpected}")

    submission_dir = exp_dir / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)
    sequence_records = []
    total_rows = 0
    for seq_name in sorted(expected):
        source_path = exp_dir / f"{seq_name}.txt"
        target_path = submission_dir / source_path.name
        row_count = 0
        min_frame = None
        max_frame = None
        with source_path.open("r", encoding="utf-8") as source, target_path.open("w", encoding="utf-8", newline="") as target:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                fields = line.strip().split(",")
                if len(fields) not in (9, 10):
                    raise RuntimeError(f"{source_path}:{line_number}: expected 9 or 10 columns, got {len(fields)}")
                try:
                    values = [float(value) for value in fields]
                except ValueError as exc:
                    raise RuntimeError(f"{source_path}:{line_number}: non-numeric value") from exc
                if not all(math.isfinite(value) for value in values):
                    raise RuntimeError(f"{source_path}:{line_number}: non-finite value")
                frame_id, track_id = int(values[0]), int(values[1])
                if frame_id < 1 or frame_id > int(seq_info[seq_name]):
                    raise RuntimeError(f"{source_path}:{line_number}: frame {frame_id} outside sequence range")
                if track_id < 1 or values[4] <= 0 or values[5] <= 0:
                    raise RuntimeError(f"{source_path}:{line_number}: invalid track id or box dimensions")
                target.write(
                    f"{frame_id},{track_id},{values[2]:.3f},{values[3]:.3f},"
                    f"{values[4]:.3f},{values[5]:.3f},{values[6]:.6f},-1,-1,-1\n"
                )
                row_count += 1
                min_frame = frame_id if min_frame is None else min(min_frame, frame_id)
                max_frame = frame_id if max_frame is None else max(max_frame, frame_id)
        if row_count == 0:
            raise RuntimeError(f"Submission file is empty: {source_path}")
        total_rows += row_count
        sequence_records.append(
            {
                "sequence": seq_name,
                "rows": row_count,
                "min_frame": min_frame,
                "max_frame": max_frame,
                "seq_length": int(seq_info[seq_name]),
                "sha256": _sha256(target_path),
            }
        )

    archive_path = exp_dir / f"{getattr(args, 'benchmark', 'benchmark')}_{getattr(args, 'split', 'test')}_submission.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for seq_name in sorted(expected):
            txt_path = submission_dir / f"{seq_name}.txt"
            info = zipfile.ZipInfo(txt_path.name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(
                info,
                txt_path.read_bytes(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            )

    summary = {
        "status": "ready",
        "format": "MOTChallenge 10-column",
        "sequence_count": len(sequence_records),
        "total_rows": total_rows,
        "archive": str(archive_path),
        "archive_sha256": _sha256(archive_path),
        "sequences": sequence_records,
    }
    (exp_dir / "submission_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def _configure_benchmark_runtime(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    return configure_benchmark_runtime(
        args,
        load_benchmark_cfg_fn=_load_benchmark_cfg,
        should_use_benchmark_detector_fn=should_use_benchmark_detector,
        should_use_benchmark_reid_fn=should_use_benchmark_reid,
        ensure_benchmark_detector_model_fn=ensure_benchmark_detector_model,
        ensure_benchmark_reid_model_fn=ensure_benchmark_reid_model,
    )


def run_trackeval(args: argparse.Namespace, verbose: bool = True) -> dict:
    """
    Evaluate tracking results via TrackEval and print a summary.
    """
    seq_paths, seq_info = _collect_seq_info(args.source)
    annotations_dir = args.source.parent / "annotations"
    gt_folder = annotations_dir if annotations_dir.exists() else args.source

    if not seq_paths:
        raise ValueError(f"No sequences with images found under {args.source}")

    if annotations_dir.exists():
        for seq_name in list(seq_info.keys()):
            ann_file = annotations_dir / f"{seq_name}.txt"
            if not ann_file.exists():
                continue
            try:
                with open(ann_file, "r") as handle:
                    max_frame = 0
                    for line in handle:
                        if not line.strip():
                            continue
                        frame_id = int(float(line.split(",", 1)[0]))
                        if frame_id > max_frame:
                            max_frame = frame_id
                    if max_frame:
                        seq_info[seq_name] = max(seq_info.get(seq_name, 0) or 0, max_frame)
            except (ValueError, OSError) as exc:
                LOGGER.warning(f"Failed to read annotation file {ann_file} for sequence length inference: {exc}")

    if getattr(args, "benchmark", None):
        save_dir = Path(args.project) / args.benchmark / args.name
    else:
        save_dir = Path(args.project) / args.name

    cfg = _load_benchmark_cfg(args)
    if not cfg:
        cfg_name = (
            getattr(args, "benchmark_id", None)
            or getattr(args, "dataset_id", None)
            or getattr(args, "benchmark", str(args.source.parent.name))
        )
        try:
            cfg = load_benchmark_cfg(cfg_name)
        except FileNotFoundError:
            found = False
            for config_file in BENCHMARK_CONFIGS.glob("*.yaml"):
                if config_file.stem in str(args.source):
                    cfg = load_benchmark_cfg(config_file.stem)
                    found = True
                    break
            if not found:
                LOGGER.warning(f"Could not find benchmark config for {cfg_name}. Class filtering might be incorrect.")
                cfg = {}

    if _resolve_eval_box_type(args, cfg) == "obb":
        trackeval_results = trackeval_obb(args, seq_paths, save_dir, gt_folder, seq_info=seq_info)
    else:
        gt_folder = prepare_aabb_eval_gt(args, gt_folder, seq_info)
        trackeval_results = trackeval_aabb(args, seq_paths, save_dir, gt_folder, seq_info=seq_info)

    parsed_results = parse_mot_results(
        trackeval_results,
        seq_names=set(seq_info.keys()),
        known_classes=_known_trackeval_class_names(args, cfg),
    )
    eval_box_type = _resolve_eval_box_type(args, cfg)

    single_class_mode = False
    if eval_box_type == "obb":
        parsed_results, single_class_mode = _filter_obb_trackeval_results(parsed_results, args, cfg.get("benchmark", {}))
    elif getattr(args, "remapped_class_names", None):
        remapped_lower = {name.lower() for name in args.remapped_class_names}
        parsed_results = {key: value for key, value in parsed_results.items() if key.lower() in remapped_lower}
        if len(args.remapped_class_names) == 1:
            single_class_mode = True
    elif "benchmark" in cfg:
        bench_cfg = cfg["benchmark"]
        bench_classes = _ordered_benchmark_eval_class_names(bench_cfg)
        if bench_classes:
            parsed_results = {key: value for key, value in parsed_results.items() if key in bench_classes}
            if len(bench_classes) == 1:
                single_class_mode = True
    elif hasattr(args, "classes") and args.classes is not None:
        class_indices = args.classes if isinstance(args.classes, list) else [args.classes]
        user_classes = [COCO_CLASSES[int(index)] for index in class_indices]
        parsed_results = {key: value for key, value in parsed_results.items() if key in user_classes}
        if len(user_classes) == 1:
            single_class_mode = True

    final_results = list(parsed_results.values())[0] if single_class_mode and parsed_results else parsed_results

    if verbose:
        log_trackeval_report(
            render_trackeval_report(
                parsed_results,
                args,
                cfg,
                title="📊 RESULTS SUMMARY",
                include_sequences=True,
                colorize=False,
            )
        )

    if getattr(args, "ci", False):
        with open(args.tracker + "_output.json", "w") as outfile:
            outfile.write(json.dumps(final_results))

    return final_results


def eval_setup(args, pipeline: PipelineTracker | None = None) -> None:
    """
    Common setup for eval and tune pipelines.
    """
    _ensure_eval_dependencies()
    status_fn = pipeline.callback() if pipeline is not None else None
    eval_init(args, status_fn=status_fn)
    _, _, dataset_detector_cfg = _configure_benchmark_runtime(args)
    det_cfg = get_runtime_detector_cfg(args.detector[0], dataset_detector_cfg)
    apply_class_remap(args, det_cfg)


def apply_class_remap(args, det_cfg: dict) -> None:
    """
    Remap GT class IDs to match detector output.
    """
    bench_cfg: dict = {}
    benchmark_id = (
        getattr(args, "benchmark_id", None)
        or getattr(args, "dataset_id", None)
        or getattr(args, "benchmark", None)
    )
    if benchmark_id:
        try:
            bench_cfg = (load_benchmark_cfg(benchmark_id) or {}).get("benchmark", {})
        except (FileNotFoundError, KeyError, ValueError) as exc:
            LOGGER.debug(f"Could not load benchmark config for class remap: {exc}")
            pass

    if str(bench_cfg.get("box_type", "")).lower() == "obb":
        return

    remap_result = build_gt_class_remap(
        bench_cfg,
        det_cfg,
        benchmark_name=getattr(args, "benchmark", ""),
        model_stem=args.detector[0].stem,
    )
    if remap_result is not None:
        remap_dict, new_class_ids, new_class_names = remap_result
        distractor_ids = [int(key) for key in bench_cfg.get("distractor_classes", {}).keys()]
        args.gt_class_remap = remap_dict
        args.gt_class_distractor_ids = distractor_ids
        args.remapped_class_ids = new_class_ids
        args.remapped_class_names = [name.lower() for name in new_class_names]


def _normalize_eval_models(args: argparse.Namespace) -> None:
    args.detector = [resolve_model_path(model) for model in args.detector]
    args.reid = [resolve_model_path(model) for model in args.reid]


def log_eval_pipeline_intro(args: argparse.Namespace) -> ui.WorkflowProgress:
    _normalize_eval_models(args)
    return EvalWorkflowReporter(args).create()


def run_eval(
    args: argparse.Namespace,
    *,
    evolve_config: dict | None = None,
    setup: bool = True,
    prepare_cache: bool = True,
    verbose: bool | None = None,
    show_progress: bool | None = None,
    pipeline: PipelineTracker | None = None,
) -> ValidationResult:
    _ensure_eval_dependencies()
    _normalize_eval_models(args)
    if verbose is None:
        verbose = bool(getattr(args, "verbose", False))
    if show_progress is None:
        show_progress = bool(getattr(args, "show_progress", True))
    args.show_progress = bool(show_progress)

    timing_stats = TimingStats()
    has_pipeline = pipeline is not None
    suppress = (not verbose) or has_pipeline

    # -- Setup --
    if setup:
        eval_setup(args, pipeline=pipeline)
        if pipeline is not None:
            pipeline.refresh_fields(_build_eval_workflow_fields(args))

    # -- Generate detections & embeddings --
    if prepare_cache:
        from boxmot.engine.workflows.support import REID_TRACKERS
        tracker_name = str(getattr(args, "tracker", "")).lower()
        if tracker_name not in REID_TRACKERS:
            args.reid = []
        if pipeline is not None:
            pipeline.advance("Generating detections & embeddings...")
        with suppress_boxmot_logs(suppress, level="WARNING"):
            run_generate_dets_embs(
                args,
                timing_stats=timing_stats,
                progress_callback=pipeline.callback() if pipeline and show_progress else None,
            )
    if pipeline is not None:
        pipeline.advance("Starting tracker...")

    # -- Track --
    with suppress_boxmot_logs(suppress, level="WARNING"):
        run_generate_mot_results(
            args,
            evolve_config=evolve_config,
            timing_stats=timing_stats,
            quiet=not bool(show_progress),
            progress_callback=pipeline.callback() if pipeline and show_progress else None,
        )
    no_gt_split = _is_no_gt_split(args)
    if pipeline is not None:
        pipeline.advance("Preparing submission..." if no_gt_split else "Computing metrics...")

    # -- Evaluate or package a no-GT test submission --
    if no_gt_split:
        raw_results = {}
        summary_label = "submission"
        summary = _prepare_no_gt_submission(args)
    else:
        raw_results = run_trackeval(args, verbose=verbose and not has_pipeline)
        summary_label, summary = extract_summary(raw_results)
    result = ValidationResult(
        benchmark=str(getattr(args, "benchmark", getattr(args, "data", ""))),
        raw=raw_results,
        summary_label=summary_label,
        summary=summary,
        exp_dir=getattr(args, "exp_dir", None),
        timings=timing_summary_from_stats(timing_stats),
        args=args,
        workflow_rendered=has_pipeline,
    )
    if pipeline is not None:
        include_timings = bool(getattr(args, "show_timing", False))
        pipeline.complete_step()
        pipeline.set_detail_renderable(
            pipeline.current_step,
            result.renderable(include_timings=include_timings),
        )

    return result


def main(args):
    _normalize_eval_models(args)
    pipeline = EvalWorkflowReporter(args).pipeline()
    with pipeline:
        result = run_eval(args, verbose=False, pipeline=pipeline)

    if result.exp_dir is not None:
        result_name = "submission_result.json" if result.summary_label == "submission" else "experiment_results.json"
        result_path = Path(result.exp_dir) / result_name
        result_path.write_text(
            json.dumps(result.to_dict(include_raw=True), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if result.summary_label == "submission":
            manifest_path = Path(result.exp_dir) / "submission_manifest.json"
            manifest = dict(result.summary)
            manifest["timings"] = result.timings
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_class, metrics_data = _select_plot_metrics_data(result.raw)
    if metrics_data:
        plotter = MetricsPlotter(result.exp_dir)
        plot_metrics = ["HOTA", "MOTA", "IDF1"]
        plot_values = [metrics_data.get(metric, 0) for metric in plot_metrics]

        plotter.plot_radar_chart(
            {args.tracker: plot_values},
            plot_metrics,
            title=f"MOT metrics radar Chart ({plot_class})",
            ylim=(0, 100),
            yticks=[20, 40, 60, 80, 100],
            ytick_labels=["20", "40", "60", "80", "100"],
        )
    return result


if __name__ == "__main__":
    main()
