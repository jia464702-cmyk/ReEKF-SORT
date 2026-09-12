from argparse import Namespace
from pathlib import Path
from zipfile import ZipFile

import pytest

from boxmot.engine.eval import evaluator
from boxmot.engine.workflows.results import ValidationResult


def _make_sequence(root: Path, name: str, frames: int) -> None:
    image_dir = root / name / "img1"
    image_dir.mkdir(parents=True)
    for frame in range(1, frames + 1):
        (image_dir / f"{frame:06d}.jpg").touch()


@pytest.mark.parametrize(
    "config",
    [
        {"no_gt_splits": ["test"], "benchmark": {"source": "data"}},
        {"dataset": {"no_gt_splits": ["test"]}},
        {"benchmark": {"no_gt_splits": ["test"]}},
    ],
)
def test_is_no_gt_split(monkeypatch, config):
    args = Namespace(split="test")
    monkeypatch.setattr(evaluator, "_load_benchmark_cfg", lambda _: config)
    assert evaluator._is_no_gt_split(args)


def test_prepare_no_gt_submission_writes_valid_archive(tmp_path):
    source = tmp_path / "data" / "test"
    _make_sequence(source, "SEQ-01", 2)
    _make_sequence(source, "SEQ-02", 1)
    exp_dir = tmp_path / "run"
    exp_dir.mkdir()
    (exp_dir / "SEQ-01.txt").write_text("1,1,10,20,30,40,0.9,1,0\n2,1,11,20,30,40,0.8,1,0\n")
    (exp_dir / "SEQ-02.txt").write_text("1,2,1,2,3,4,0.7,1,0\n")
    args = Namespace(source=source, exp_dir=exp_dir, benchmark="mot17", split="test")

    summary = evaluator._prepare_no_gt_submission(args)

    assert summary["status"] == "ready"
    assert summary["sequence_count"] == 2
    assert summary["total_rows"] == 3
    archive_path = Path(summary["archive"])
    assert archive_path.is_file()
    with ZipFile(archive_path) as archive:
        assert archive.namelist() == ["SEQ-01.txt", "SEQ-02.txt"]
        first_line = archive.read("SEQ-01.txt").decode().splitlines()[0]
    assert len(first_line.split(",")) == 10
    assert first_line.endswith(",-1,-1,-1")


def test_prepare_no_gt_submission_rejects_missing_sequence(tmp_path):
    source = tmp_path / "data" / "test"
    _make_sequence(source, "SEQ-01", 1)
    exp_dir = tmp_path / "run"
    exp_dir.mkdir()
    args = Namespace(source=source, exp_dir=exp_dir, benchmark="mot17", split="test")
    with pytest.raises(RuntimeError, match="missing"):
        evaluator._prepare_no_gt_submission(args)


def test_prepare_no_gt_submission_is_repeatable(tmp_path):
    source = tmp_path / "data" / "test"
    _make_sequence(source, "SEQ-01", 1)
    exp_dir = tmp_path / "run"
    exp_dir.mkdir()
    (exp_dir / "SEQ-01.txt").write_text("1,1,10,20,30,40,0.9,1,0\n")
    args = Namespace(source=source, exp_dir=exp_dir, benchmark="mot17", split="test")

    first = evaluator._prepare_no_gt_submission(args)
    second = evaluator._prepare_no_gt_submission(args)

    assert first["archive_sha256"] == second["archive_sha256"]
    assert second["sequence_count"] == 1
    assert second["total_rows"] == 1


def test_submission_result_report():
    result = ValidationResult(
        benchmark="mot17",
        raw={},
        summary_label="submission",
        summary={
            "sequence_count": 21,
            "total_rows": 123,
            "format": "MOTChallenge 10-column",
            "archive": "submission.zip",
            "archive_sha256": "abc",
        },
    )
    report = result.render()
    assert "SUBMISSION READY" in report
    assert "Sequences: 21" in report
    assert "SHA256: abc" in report
