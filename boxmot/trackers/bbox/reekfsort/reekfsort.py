from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from boxmot.trackers.association import linear_assignment, speed_direction_batch
from boxmot.trackers.basetracker import BaseTracker
from boxmot.trackers.ops import xyxy2xysr


def _wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    wrapped = (np.asarray(angle, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi
    if np.isscalar(angle):
        return float(wrapped)
    return wrapped


def _bbox_center(bbox: np.ndarray) -> tuple[float, float]:
    return (
        float((bbox[0] + bbox[2]) * 0.5),
        float((bbox[1] + bbox[3]) * 0.5),
    )


def _k_previous_observation(
    observations: dict[int, np.ndarray], current_age: int, delta_t: int
) -> np.ndarray:
    """Return the latest observation up to ``delta_t`` frames in the past."""
    if not observations:
        return np.full(5, -1.0, dtype=float)
    for offset in range(delta_t, 0, -1):
        if current_age - offset in observations:
            return observations[current_age - offset]
    return observations[max(observations)]


def _speed_direction(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    """Return normalized image-plane motion as ``[dy, dx]``."""
    center1 = _bbox_center(bbox1)
    center2 = _bbox_center(bbox2)
    delta = np.array([center2[1] - center1[1], center2[0] - center1[0]])
    return delta / (np.linalg.norm(delta) + 1e-6)


CONFIDENCE_COST_MODES = ("absolute", "normalized", "raw", "none")


def confidence_continuity_cost(
    detection_confidences: np.ndarray,
    track_confidences: np.ndarray,
    *,
    mode: str = "absolute",
    eps: float = 1e-6,
) -> np.ndarray:
    """Return a detection-to-track confidence cost matrix.

    ``absolute`` is the validation-selected paper formulation. The other modes
    are exposed for controlled ablation. In particular, ``raw`` is a joint-score
    baseline and must not be interpreted as a calibrated match probability.
    """
    selected_mode = str(mode).lower()
    if selected_mode not in CONFIDENCE_COST_MODES:
        raise ValueError(
            "confidence_cost_mode must be one of: normalized, absolute, raw, none"
        )

    det_confs = np.asarray(detection_confidences, dtype=float).reshape(-1, 1)
    trk_confs = np.asarray(track_confidences, dtype=float).reshape(1, -1)
    shape = (det_confs.shape[0], trk_confs.shape[1])
    if selected_mode == "normalized":
        return np.abs(det_confs - trk_confs) / (det_confs + trk_confs + eps)
    if selected_mode == "absolute":
        return np.abs(det_confs - trk_confs)
    if selected_mode == "raw":
        return 1.0 - det_confs * trk_confs
    return np.zeros(shape, dtype=float)


def _bbox_to_measurement(
    bbox: np.ndarray,
    *,
    reference_bbox: np.ndarray | None,
    fallback_angle: float,
) -> np.ndarray:
    xysr = xyxy2xysr(bbox).reshape(-1)
    if reference_bbox is None or np.asarray(reference_bbox).sum() < 0:
        angle = fallback_angle
    else:
        prev_u, prev_v = _bbox_center(reference_bbox)
        curr_u, curr_v = _bbox_center(bbox)
        angle = np.arctan2(curr_v - prev_v, curr_u - prev_u)
    return np.array([xysr[0], xysr[1], xysr[2], xysr[3], angle, bbox[4]], dtype=float)


def _state_to_bbox(x: np.ndarray, score: float | None = None) -> np.ndarray:
    """Convert EKF state [u, v, s, r, ...] to [x1, y1, x2, y2, optional score]."""
    state = np.asarray(x, dtype=float).reshape(-1)
    s = max(float(state[2]), 1e-12)
    r = max(float(state[3]), 1e-12)
    w = np.sqrt(s * r)
    h = s / max(w, 1e-6)
    box = np.array(
        [
            state[0] - w * 0.5,
            state[1] - h * 0.5,
            state[0] + w * 0.5,
            state[1] + h * 0.5,
        ],
        dtype=float,
    )
    if score is None:
        return box.reshape(1, 4)
    return np.concatenate([box, [float(score)]]).reshape(1, 5)


class SpeedAngleEkf:
    """Stateful EKF for the ReEKF-SORT speed-angle motion model.

    State layout:
        [u, v, s, r, rho, angle, s_dot, angle_dot, conf, conf_dot]

    The nonlinear part is the image-plane position update:
        u' = u + rho * cos(angle)
        v' = v + rho * sin(angle)

    The remaining components use constant-rate propagation. The Jacobian below
    linearizes only the speed-angle transition needed by the EKF covariance
    prediction step.
    """

    dim_x = 10
    dim_z = 6

    def __init__(
        self,
        bbox: np.ndarray,
        *,
        q_xy_scaling: float = 0.01,
        q_s_scaling: float = 0.0001,
        q_angle_scaling: float = 0.001,
        q_conf_scaling: float = 0.001,
    ) -> None:
        xysr = xyxy2xysr(bbox).reshape(-1)
        conf = float(np.clip(bbox[4], 0.0, 1.0))

        self.x = np.zeros((self.dim_x, 1), dtype=float)
        self.x[0:4, 0] = xysr
        self.x[8, 0] = conf

        self.P = np.eye(self.dim_x, dtype=float) * 10.0
        self.P[4:, 4:] *= 100.0

        self.Q = np.diag(
            [
                q_xy_scaling,
                q_xy_scaling,
                q_s_scaling,
                1e-4,
                q_xy_scaling,
                q_angle_scaling,
                q_s_scaling,
                q_angle_scaling,
                q_conf_scaling,
                q_conf_scaling,
            ]
        ).astype(float)

        self.R = np.diag([1.0, 1.0, 10.0, 10.0, 0.1, 0.05]).astype(float)
        self.H = np.zeros((self.dim_z, self.dim_x), dtype=float)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0
        self.H[4, 5] = 1.0
        self.H[5, 8] = 1.0
        self._I = np.eye(self.dim_x, dtype=float)

    def _transition_jacobian(self) -> np.ndarray:
        """Build the local first-order approximation of the nonlinear transition."""
        state = self.x[:, 0]
        rho = float(state[4])
        angle = float(state[5])

        F = np.eye(self.dim_x, dtype=float)
        F[0, 4] = np.cos(angle)
        F[0, 5] = -rho * np.sin(angle)
        F[1, 4] = np.sin(angle)
        F[1, 5] = rho * np.cos(angle)
        F[2, 6] = 1.0
        F[5, 7] = 1.0
        F[8, 9] = 1.0
        return F

    def predict(self) -> None:
        """Run the EKF prediction step with the nonlinear speed-angle model."""
        state = self.x[:, 0].copy()
        rho = float(state[4])
        angle = float(state[5])

        predicted = state.copy()
        predicted[0] = state[0] + rho * np.cos(angle)
        predicted[1] = state[1] + rho * np.sin(angle)
        predicted[2] = max(state[2] + state[6], 1e-6)
        predicted[3] = max(state[3], 1e-6)
        predicted[5] = _wrap_angle(state[5] + state[7])
        predicted[8] = float(np.clip(state[8] + state[9], 0.0, 1.0))

        F = self._transition_jacobian()
        self.x[:, 0] = predicted
        self.P = F @ self.P @ F.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)

    def update(self, z: np.ndarray, R: np.ndarray | None = None) -> None:
        """Correct the EKF state with a real or virtual observation.

        The observation layout is [u, v, s, r, angle, confidence]. Angle
        residuals are wrapped before correction so crossing the -pi/pi boundary
        does not create a false large innovation.
        """
        measurement = np.asarray(z, dtype=float).reshape((self.dim_z, 1))
        measurement[2, 0] = max(float(measurement[2, 0]), 1e-6)
        measurement[3, 0] = max(float(measurement[3, 0]), 1e-6)
        measurement[4, 0] = self.x[5, 0] + _wrap_angle(
            float(measurement[4, 0] - self.x[5, 0])
        )
        measurement[5, 0] = float(np.clip(measurement[5, 0], 0.0, 1.0))

        obs_noise = self.R if R is None else np.asarray(R, dtype=float)
        projected = self.H @ self.x
        innovation = measurement - projected
        innovation[4, 0] = _wrap_angle(float(innovation[4, 0]))

        S = self.H @ self.P @ self.H.T + obs_noise
        K = self.P @ self.H.T @ np.linalg.pinv(S)
        self.x = self.x + K @ innovation
        # Joseph-form covariance update is more numerically stable than the
        # abbreviated (I-KH)P form and better preserves positive semidefiniteness.
        ikh = self._I - K @ self.H
        self.P = ikh @ self.P @ ikh.T + K @ obs_noise @ K.T

        self.x[2, 0] = max(float(self.x[2, 0]), 1e-6)
        self.x[3, 0] = max(float(self.x[3, 0]), 1e-6)
        self.x[5, 0] = _wrap_angle(float(self.x[5, 0]))
        self.x[8, 0] = float(np.clip(self.x[8, 0], 0.0, 1.0))
        self.P = 0.5 * (self.P + self.P.T)


class ReEkfTrack:
    """Single target state for ReEKF-SORT.

    Each track owns one EKF plus the historical motion summaries needed by the
    causal trajectory prediction module. The summaries are updated only when a
    real detection is matched, so virtual observations during occlusion never
    use future re-observations.
    """

    count = 0

    def __init__(
        self,
        bbox: np.ndarray,
        cls: float,
        det_ind: float,
        *,
        max_obs: int = 50,
        angle_smoothing: float = 0.8,
        angular_velocity_smoothing: float = 0.8,
        confidence_decay: float = 0.95,
        virtual_update_interval: int = 6,
        virtual_obs_noise_scale: float = 10.0,
        delta_t: int = 3,
        Q_xy_scaling: float = 0.01,
        Q_s_scaling: float = 0.0001,
        Q_angle_scaling: float = 0.001,
        Q_conf_scaling: float = 0.001,
    ) -> None:
        self.kf = SpeedAngleEkf(
            bbox,
            q_xy_scaling=Q_xy_scaling,
            q_s_scaling=Q_s_scaling,
            q_angle_scaling=Q_angle_scaling,
            q_conf_scaling=Q_conf_scaling,
        )
        self.id = ReEkfTrack.count
        ReEkfTrack.count += 1

        self.max_obs = max_obs
        self.history = deque([], maxlen=self.max_obs)
        self.history_observations = deque([], maxlen=self.max_obs)
        self.observations: dict[int, np.ndarray] = {}
        self.last_observation = np.asarray(bbox, dtype=float).copy()

        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.conf = float(bbox[-1])
        self.cls = cls
        self.det_ind = det_ind

        self.angle_smoothing = float(angle_smoothing)
        self.angular_velocity_smoothing = float(angular_velocity_smoothing)
        self.confidence_decay = float(confidence_decay)
        self.virtual_update_interval = max(1, int(virtual_update_interval))
        self.virtual_obs_noise_scale = float(virtual_obs_noise_scale)
        self.delta_t = max(1, int(delta_t))
        self.velocity: np.ndarray | None = None

        self.smoothed_angle = float(self.kf.x[5, 0])
        self.smoothed_angular_velocity = 0.0
        self.occlusion_age = 0
        self._last_reliable_center = _bbox_center(bbox)
        self._last_reliable_scale = float(self.kf.x[2, 0])
        self._last_reliable_ratio = float(self.kf.x[3, 0])
        self._last_reliable_speed = 0.0
        self._last_reliable_conf = self.conf

    def _store_reliable_motion(self, bbox: np.ndarray) -> None:
        """Update pre-occlusion motion summaries from a matched detection."""
        curr_center = _bbox_center(bbox)
        prev_center = self._last_reliable_center
        dx = curr_center[0] - prev_center[0]
        dy = curr_center[1] - prev_center[1]
        speed = float(np.hypot(dx, dy))

        if speed > 1e-6:
            angle = float(np.arctan2(dy, dx))
            angular_velocity = _wrap_angle(angle - self.smoothed_angle)
            self.smoothed_angle = _wrap_angle(
                self.smoothed_angle
                + (1.0 - self.angle_smoothing)
                * _wrap_angle(angle - self.smoothed_angle)
            )
            self.smoothed_angular_velocity = (
                self.angular_velocity_smoothing * self.smoothed_angular_velocity
                + (1.0 - self.angular_velocity_smoothing) * angular_velocity
            )
            self.kf.x[4, 0] = speed
            self.kf.x[5, 0] = self.smoothed_angle
            self.kf.x[7, 0] = self.smoothed_angular_velocity
            self._last_reliable_speed = speed

        xysr = xyxy2xysr(bbox).reshape(-1)
        self._last_reliable_center = curr_center
        self._last_reliable_scale = float(xysr[2])
        self._last_reliable_ratio = float(xysr[3])
        self._last_reliable_conf = float(bbox[4])

    def update(self, bbox: np.ndarray, cls: float, det_ind: float) -> None:
        self.det_ind = det_ind
        self.conf = float(bbox[-1])
        self.cls = cls

        reference_bbox = None if self.last_observation.sum() < 0 else self.last_observation
        if self.observations:
            previous_bbox = _k_previous_observation(
                self.observations, self.age, self.delta_t
            )
            self.velocity = _speed_direction(previous_bbox, bbox)
        self._store_reliable_motion(bbox)
        z = _bbox_to_measurement(
            bbox,
            reference_bbox=reference_bbox,
            fallback_angle=float(self.kf.x[5, 0]),
        )
        self.kf.update(z)

        self.last_observation = np.asarray(bbox, dtype=float).copy()
        self.observations[self.age] = self.last_observation.copy()
        self.history_observations.append(self.last_observation.copy())
        self.time_since_update = 0
        self.occlusion_age = 0
        self.hits += 1
        self.hit_streak += 1

    def _virtual_observation(self) -> np.ndarray:
        """Generate a causal pseudo-observation for the current occlusion step.

        The pseudo state is rolled forward from the last reliable center using
        the smoothed pre-occlusion angle and angular velocity. Confidence decays
        over occlusion time, and the caller applies enlarged observation noise
        so this synthetic measurement remains a weak correction.
        """
        k = max(1, self.occlusion_age)
        center_u, center_v = self._last_reliable_center
        angle = self.smoothed_angle
        for step in range(1, k + 1):
            angle = _wrap_angle(
                self.smoothed_angle + step * self.smoothed_angular_velocity
            )
            center_u += self._last_reliable_speed * np.cos(angle)
            center_v += self._last_reliable_speed * np.sin(angle)
        conf = float((self.confidence_decay ** k) * self._last_reliable_conf)
        return np.array(
            [
                center_u,
                center_v,
                max(float(self.kf.x[2, 0]), 1e-6),
                max(float(self.kf.x[3, 0]), 1e-6),
                angle,
                np.clip(conf, 0.0, 1.0),
            ],
            dtype=float,
        )

    def mark_missed(self) -> None:
        """Handle one unmatched frame and optionally inject a pseudo-observation."""
        self.occlusion_age += 1
        if self.occlusion_age % self.virtual_update_interval != 0:
            return
        # Pseudo-observations are intentionally noisy so they guide the EKF
        # without overpowering future real detections.
        self.kf.update(
            self._virtual_observation(),
            R=self.kf.R * self.virtual_obs_noise_scale,
        )

    def predict(self) -> np.ndarray:
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(_state_to_bbox(self.kf.x, score=float(self.kf.x[8, 0])))
        return self.history[-1]

    def get_state(self) -> np.ndarray:
        return _state_to_bbox(self.kf.x)

    @property
    def predicted_confidence(self) -> float:
        return float(np.clip(self.kf.x[8, 0], 0.0, 1.0))

    @property
    def predicted_angle(self) -> float:
        return float(self.kf.x[5, 0])


def associate_reekf(
    detections: np.ndarray,
    trackers: np.ndarray,
    tracks: list[ReEkfTrack],
    asso_func,
    iou_threshold: float,
    lambda_conf: float,
    lambda_angle: float,
    eps: float,
    confidence_cost_mode: str = "absolute",
    velocities: np.ndarray | None = None,
    previous_observations: np.ndarray | None = None,
    inertia: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Associate detections and tracks with confidence and OC motion cues.

    The final matrix minimizes:
        (1 - IoU) - observation_centric_direction_reward
                  + lambda_conf * confidence_continuity
                  + lambda_angle * EKF_angle_consistency

    The paper configuration uses the absolute difference between the current
    detection confidence and the track's predicted confidence. This penalizes
    abrupt confidence changes without treating either score as a calibrated
    match probability.
    """
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0,), dtype=int),
        )

    if len(detections) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.empty((0,), dtype=int),
            np.arange(len(trackers)),
        )

    iou_matrix = np.asarray(asso_func(detections, trackers), dtype=float)
    iou_cost = 1.0 - iou_matrix

    conf_cost = confidence_continuity_cost(
        detections[:, 4],
        [trk.predicted_confidence for trk in tracks],
        mode=confidence_cost_mode,
        eps=eps,
    )

    angle_cost = np.zeros_like(iou_cost)
    for det_idx, det in enumerate(detections):
        det_u, det_v = _bbox_center(det)
        for trk_idx, trk in enumerate(tracks):
            ref_u, ref_v = _bbox_center(trk.last_observation)
            det_angle = np.arctan2(det_v - ref_v, det_u - ref_u)
            angle_cost[det_idx, trk_idx] = 1.0 - np.cos(
                _wrap_angle(det_angle - trk.predicted_angle)
            )

    direction_reward = np.zeros_like(iou_cost)
    if (
        inertia > 0.0
        and velocities is not None
        and previous_observations is not None
        and len(previous_observations) == len(trackers)
    ):
        direction_y, direction_x = speed_direction_batch(
            detections, previous_observations
        )
        track_velocities = np.asarray(velocities, dtype=float)
        velocity_y = np.repeat(
            track_velocities[:, 0, np.newaxis], direction_y.shape[1], axis=1
        )
        velocity_x = np.repeat(
            track_velocities[:, 1, np.newaxis], direction_x.shape[1], axis=1
        )
        cosine = np.clip(
            velocity_x * direction_x + velocity_y * direction_y, -1.0, 1.0
        )
        angular_reward = (np.pi / 2.0 - np.abs(np.arccos(cosine))) / np.pi
        valid = (np.asarray(previous_observations)[:, 4] >= 0).astype(float)
        angular_reward *= valid[:, np.newaxis]
        angular_reward *= detections[:, 4][np.newaxis, :]
        direction_reward = (float(inertia) * angular_reward).T

    cost_matrix = (
        iou_cost
        - direction_reward
        + lambda_conf * conf_cost
        + lambda_angle * angle_cost
    )
    matched_indices = linear_assignment(cost_matrix)
    if matched_indices.size == 0:
        matched_indices = np.empty((0, 2), dtype=int)

    unmatched_detections = []
    for det_idx in range(len(detections)):
        if det_idx not in matched_indices[:, 0]:
            unmatched_detections.append(det_idx)

    unmatched_trackers = []
    for trk_idx in range(len(trackers)):
        if trk_idx not in matched_indices[:, 1]:
            unmatched_trackers.append(trk_idx)

    matches = []
    for det_idx, trk_idx in matched_indices:
        if iou_matrix[det_idx, trk_idx] < iou_threshold:
            unmatched_detections.append(det_idx)
            unmatched_trackers.append(trk_idx)
            continue
        matches.append(np.array([[det_idx, trk_idx]], dtype=int))

    if not matches:
        return (
            np.empty((0, 2), dtype=int),
            np.array(unmatched_detections, dtype=int),
            np.array(unmatched_trackers, dtype=int),
        )

    return (
        np.concatenate(matches, axis=0),
        np.array(unmatched_detections, dtype=int),
        np.array(unmatched_trackers, dtype=int),
    )


def rematch_last_observations(
    detections: np.ndarray,
    last_observations: np.ndarray,
    unmatched_detections: np.ndarray,
    unmatched_trackers: np.ndarray,
    asso_func,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run OC-SORT's OCR pass against the last real observations."""
    if len(unmatched_detections) == 0 or len(unmatched_trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            unmatched_detections,
            unmatched_trackers,
        )

    iou_matrix = np.asarray(
        asso_func(
            detections[unmatched_detections],
            last_observations[unmatched_trackers],
        ),
        dtype=float,
    )
    if iou_matrix.size == 0 or iou_matrix.max() <= iou_threshold:
        return (
            np.empty((0, 2), dtype=int),
            unmatched_detections,
            unmatched_trackers,
        )

    assignments = linear_assignment(-iou_matrix)
    matches = []
    for det_pos, trk_pos in assignments:
        if iou_matrix[det_pos, trk_pos] < iou_threshold:
            continue
        matches.append(
            [unmatched_detections[det_pos], unmatched_trackers[trk_pos]]
        )

    if not matches:
        return (
            np.empty((0, 2), dtype=int),
            unmatched_detections,
            unmatched_trackers,
        )

    matches_array = np.asarray(matches, dtype=int)
    return (
        matches_array,
        np.setdiff1d(unmatched_detections, matches_array[:, 0]),
        np.setdiff1d(unmatched_trackers, matches_array[:, 1]),
    )


class ReEkfSort(BaseTracker):
    """ReEKF-SORT tracker based on causal EKF prediction and confidence continuity."""

    supports_obb = False

    def __init__(
        self,
        min_conf: float = 0.1,
        lambda_conf: float = 1.2,
        lambda_angle: float = 0.0,
        confidence_eps: float = 1e-6,
        confidence_cost_mode: str = "absolute",
        angle_smoothing: float = 0.8,
        angular_velocity_smoothing: float = 0.8,
        confidence_decay: float = 0.95,
        virtual_update_interval: int = 6,
        virtual_obs_noise_scale: float = 10.0,
        delta_t: int = 3,
        inertia: float = 0.1,
        use_observation_centric: bool = True,
        use_ocr: bool = True,
        use_virtual_observation: bool = True,
        use_confidence_cost: bool = True,
        use_angle_cost: bool = False,
        Q_xy_scaling: float = 0.01,
        Q_s_scaling: float = 0.0001,
        Q_angle_scaling: float = 0.001,
        Q_conf_scaling: float = 0.001,
        **kwargs: Any,
    ) -> None:
        init_args = {k: v for k, v in locals().items() if k not in ("self", "kwargs")}
        super().__init__(**init_args, _tracker_name="ReEkfSort", **kwargs)

        if self.is_obb:
            raise AssertionError(
                "ReEkfSort currently supports axis-aligned detections only."
            )

        self.min_conf = float(min_conf)
        self.asso_threshold = float(self.iou_threshold)
        self.lambda_conf = float(lambda_conf)
        self.lambda_angle = float(lambda_angle)
        self.confidence_eps = float(confidence_eps)
        self.confidence_cost_mode = str(confidence_cost_mode).lower()
        if self.confidence_cost_mode not in CONFIDENCE_COST_MODES:
            raise ValueError(
                "confidence_cost_mode must be one of: normalized, absolute, raw, none"
            )
        self.angle_smoothing = float(angle_smoothing)
        self.angular_velocity_smoothing = float(angular_velocity_smoothing)
        self.confidence_decay = float(confidence_decay)
        self.virtual_update_interval = int(virtual_update_interval)
        self.virtual_obs_noise_scale = float(virtual_obs_noise_scale)
        self.delta_t = max(1, int(delta_t))
        self.inertia = float(inertia)
        self.use_observation_centric = bool(use_observation_centric)
        self.use_ocr = bool(use_ocr)
        self.use_virtual_observation = bool(use_virtual_observation)
        self.use_confidence_cost = bool(use_confidence_cost)
        self.use_angle_cost = bool(use_angle_cost)
        self.Q_xy_scaling = float(Q_xy_scaling)
        self.Q_s_scaling = float(Q_s_scaling)
        self.Q_angle_scaling = float(Q_angle_scaling)
        self.Q_conf_scaling = float(Q_conf_scaling)
        self.frame_count = 0
        ReEkfTrack.count = 0
        self.active_tracks: list[ReEkfTrack] = []

    def _update_impl(
        self,
        dets: np.ndarray,
        img: np.ndarray,
        embs: np.ndarray = None,
        masks: np.ndarray = None,
    ) -> np.ndarray:
        self.check_inputs(dets, img)
        self.frame_count += 1

        dets = self.detection_layout.with_detection_indices(dets)
        confs = self.detection_layout.confidences(dets)
        dets = dets[confs > max(self.det_thresh, self.min_conf)]

        trks = np.zeros(
            (len(self.active_tracks), self.detection_layout.box_with_conf_cols)
        )
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.active_tracks[t].predict()[0]
            trk[:] = [pos[i] for i in range(self.detection_layout.box_cols)] + [pos[4]]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.active_tracks.pop(t)

        velocities = np.array(
            [
                trk.velocity if trk.velocity is not None else np.zeros(2)
                for trk in self.active_tracks
            ]
        )
        previous_observations = np.array(
            [
                _k_previous_observation(
                    trk.observations, trk.age, self.delta_t
                )
                for trk in self.active_tracks
            ]
        )
        last_observations = np.array(
            [trk.last_observation for trk in self.active_tracks]
        )

        matched, unmatched_dets, unmatched_trks = associate_reekf(
            dets[:, : self.detection_layout.box_with_conf_cols],
            trks,
            self.active_tracks,
            self.asso_func,
            self.asso_threshold,
            self.lambda_conf if self.use_confidence_cost else 0.0,
            self.lambda_angle if self.use_angle_cost else 0.0,
            self.confidence_eps,
            self.confidence_cost_mode,
            velocities=velocities,
            previous_observations=previous_observations,
            inertia=self.inertia if self.use_observation_centric else 0.0,
        )

        for det_idx, trk_idx in matched:
            self.active_tracks[trk_idx].update(
                dets[det_idx, :-2],
                dets[det_idx, self.detection_layout.cls_idx],
                dets[det_idx, self.detection_layout.det_cols],
            )

        if self.use_ocr:
            rematched, unmatched_dets, unmatched_trks = rematch_last_observations(
                dets[:, : self.detection_layout.box_with_conf_cols],
                last_observations,
                unmatched_dets,
                unmatched_trks,
                self.asso_func,
                self.asso_threshold,
            )
            for det_idx, trk_idx in rematched:
                self.active_tracks[trk_idx].update(
                    dets[det_idx, :-2],
                    dets[det_idx, self.detection_layout.cls_idx],
                    dets[det_idx, self.detection_layout.det_cols],
                )

        for trk_idx in unmatched_trks:
            if self.use_virtual_observation:
                self.active_tracks[trk_idx].mark_missed()

        for det_idx in unmatched_dets:
            trk = ReEkfTrack(
                dets[det_idx, : self.detection_layout.box_with_conf_cols],
                dets[det_idx, self.detection_layout.cls_idx],
                dets[det_idx, self.detection_layout.det_cols],
                max_obs=self.max_obs,
                angle_smoothing=self.angle_smoothing,
                angular_velocity_smoothing=self.angular_velocity_smoothing,
                confidence_decay=self.confidence_decay,
                virtual_update_interval=self.virtual_update_interval,
                virtual_obs_noise_scale=self.virtual_obs_noise_scale,
                delta_t=self.delta_t,
                Q_xy_scaling=self.Q_xy_scaling,
                Q_s_scaling=self.Q_s_scaling,
                Q_angle_scaling=self.Q_angle_scaling,
                Q_conf_scaling=self.Q_conf_scaling,
            )
            self.active_tracks.append(trk)

        i = len(self.active_tracks)
        for trk in reversed(self.active_tracks):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                d = trk.last_observation[: self.detection_layout.box_cols]

            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                ret.append(
                    np.concatenate(
                        (d, [trk.id + 1], [trk.conf], [trk.cls], [trk.det_ind])
                    ).reshape(1, -1)
                )
            i -= 1
            if trk.time_since_update > self.max_age:
                self.active_tracks.pop(i)

        if ret:
            return np.concatenate(ret)
        return self.empty_output()
