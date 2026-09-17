from pathlib import Path

import numpy as np
import pytest
import yaml

from boxmot.trackers.bbox.reekfsort.reekfsort import (
    _k_previous_observation,
    _speed_direction,
    associate_reekf,
    confidence_continuity_cost,
    rematch_last_observations,
)


def test_normalized_confidence_gap_matches_ablation_equation():
    result = confidence_continuity_cost(
        np.array([0.8]), np.array([0.6]), mode="normalized", eps=1e-6
    )
    expected = abs(0.8 - 0.6) / (0.8 + 0.6 + 1e-6)
    assert result.shape == (1, 1)
    assert result[0, 0] == pytest.approx(expected)


def test_confidence_ablation_modes():
    detections = np.array([0.8, 0.2])
    tracks = np.array([0.6, 0.1])
    np.testing.assert_allclose(
        confidence_continuity_cost(detections, tracks, mode="absolute"),
        [[0.2, 0.7], [0.4, 0.1]],
    )
    np.testing.assert_allclose(
        confidence_continuity_cost(detections, tracks, mode="raw"),
        [[0.52, 0.92], [0.88, 0.98]],
    )
    np.testing.assert_allclose(
        confidence_continuity_cost(detections, tracks, mode="none"),
        np.zeros((2, 2)),
    )


def test_default_confidence_gap_is_validation_selected_absolute():
    result = confidence_continuity_cost(np.array([0.8]), np.array([0.6]))
    np.testing.assert_allclose(result, [[0.2]])


def test_unknown_confidence_cost_mode_is_rejected():
    with pytest.raises(ValueError, match="confidence_cost_mode"):
        confidence_continuity_cost(np.array([0.5]), np.array([0.5]), mode="invalid")


def test_paper_final_configuration_is_frozen():
    config_path = (
        Path(__file__).resolve().parents[2]
        / "boxmot"
        / "configs"
        / "trackers"
        / "reekfsort.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected = {
        "confidence_cost_mode": "absolute",
        "lambda_conf": 1.2,
        "virtual_update_interval": 6,
        "virtual_obs_noise_scale": 10.0,
        "use_virtual_observation": True,
        "use_confidence_cost": True,
        "use_angle_cost": False,
        "lambda_angle": 0.0,
    }
    actual = {key: config[key]["default"] for key in expected}
    assert actual == expected


def test_disabled_angle_cost_skips_angle_computation():
    class Track:
        predicted_confidence = 0.8

        @property
        def last_observation(self):
            raise AssertionError(
                "angle state should not be read when lambda_angle is zero"
            )

        @property
        def predicted_angle(self):
            raise AssertionError(
                "angle state should not be read when lambda_angle is zero"
            )

    matches, unmatched_dets, unmatched_trks = associate_reekf(
        detections=np.array([[0.0, 0.0, 2.0, 2.0, 0.9]]),
        trackers=np.array([[0.0, 0.0, 2.0, 2.0]]),
        tracks=[Track()],
        asso_func=lambda dets, trks: np.array([[0.9]]),
        iou_threshold=0.3,
        lambda_conf=1.2,
        lambda_angle=0.0,
        eps=1e-6,
    )

    np.testing.assert_array_equal(matches, [[0, 0]])
    assert unmatched_dets.size == 0
    assert unmatched_trks.size == 0


def test_observation_centric_helpers_use_requested_history_window():
    observations = {
        2: np.array([0.0, 0.0, 2.0, 2.0, 0.8]),
        4: np.array([2.0, 0.0, 4.0, 2.0, 0.8]),
    }
    previous = _k_previous_observation(observations, current_age=5, delta_t=3)
    np.testing.assert_array_equal(previous, observations[2])
    np.testing.assert_allclose(
        _speed_direction(previous, observations[4]),
        [0.0, 1.0],
        atol=1e-6,
    )


def test_ocr_rematches_detection_to_last_real_observation(monkeypatch):
    monkeypatch.setattr(
        "boxmot.trackers.bbox.reekfsort.reekfsort.linear_assignment",
        lambda cost: np.array([[0, 0]], dtype=int),
    )
    detections = np.array([[10.0, 10.0, 20.0, 20.0, 0.9]])
    last_observations = np.array([[10.0, 10.0, 20.0, 20.0, 0.8]])
    matches, unmatched_dets, unmatched_trks = rematch_last_observations(
        detections,
        last_observations,
        np.array([0]),
        np.array([0]),
        lambda dets, trks: np.array([[0.9]]),
        0.3,
    )
    np.testing.assert_array_equal(matches, [[0, 0]])
    assert unmatched_dets.size == 0
    assert unmatched_trks.size == 0
