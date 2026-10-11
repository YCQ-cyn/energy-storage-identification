import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


TARGET_PIPELINE_VERSION = "teacher-storage-behavior-v5-pq-visible-role-normalized"
PHYSICAL_TARGET_NAMES = (
    "pq_alignment",
    "edge_pair_match",
    "visible_segment_coherence",
    "activity_strength",
    "storage_evidence",
)
RESPONSE_TARGET_NAMES = (
    "charge_visible",
    "discharge_visible",
    "response_confidence",
)
BEHAVIOR_TARGET_NAMES = (
    "charge_role_fit",
    "discharge_role_fit",
    "arbitrage_order",
    "policy_pattern",
    "storage_behavior",
)


@dataclass(frozen=True)
class TeacherTargetConfig:
    charge_sign: int = 1
    activity_ratio: float = 0.02
    min_activity_to_load_ratio: float = 0.002
    alignment_max_lag: int = 2
    background_smooth_radius: int = 2
    noise_smooth_radius: int = 4
    response_noise_floor_ratio: float = 0.03
    min_run_points: float = 2.0
    edge_window_points: int = 2
    strength_reference_ratio: float = 0.03
    role_smooth_radius: int = 2
    lower_quantile: float = 0.05
    upper_quantile: float = 0.95
    eps: float = 1e-6

    def __post_init__(self):
        if self.charge_sign not in {-1, 1}:
            raise ValueError("charge_sign 必须为 -1 或 1")
        if not 0.0 < self.activity_ratio < 1.0:
            raise ValueError("activity_ratio 必须位于 (0, 1)")
        if not 0.0 <= self.min_activity_to_load_ratio < 1.0:
            raise ValueError("min_activity_to_load_ratio 必须位于 [0, 1)")
        if self.alignment_max_lag < 0 or self.alignment_max_lag > 8:
            raise ValueError("alignment_max_lag 必须位于 [0, 8]")
        if self.background_smooth_radius < 0 or self.noise_smooth_radius < 0:
            raise ValueError("平滑半径不能为负数")
        if self.min_run_points <= 0 or self.strength_reference_ratio <= 0:
            raise ValueError("运行长度和强度参考值必须为正数")
        if self.edge_window_points <= 0 or self.edge_window_points > 8:
            raise ValueError("边界窗口长度必须位于 [1, 8]")
        if not 0.0 <= self.lower_quantile < self.upper_quantile <= 1.0:
            raise ValueError("目标分位数配置无效")


def _as_numpy(value, dtype=np.float32):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _validate_curves(p_raw, q_raw, price_role_seq=None):
    p_raw = _as_numpy(p_raw)
    q_raw = _as_numpy(q_raw)
    if p_raw.ndim != 2 or p_raw.shape[1] != 96:
        raise ValueError(f"P_raw 形状应为 N×96，实际为 {p_raw.shape}")
    if q_raw.shape != p_raw.shape:
        raise ValueError(f"Q_raw 与 P_raw 形状不一致：{q_raw.shape} != {p_raw.shape}")
    if not np.isfinite(p_raw).all() or not np.isfinite(q_raw).all():
        raise ValueError("原始 P/Q 中存在非有限值")

    if price_role_seq is None:
        return p_raw, q_raw, None
    price_role_seq = _as_numpy(price_role_seq)
    if price_role_seq.ndim == 1:
        price_role_seq = np.broadcast_to(price_role_seq, p_raw.shape)
    if price_role_seq.shape != p_raw.shape:
        raise ValueError("price_role_seq 与 P_raw 形状不一致")
    return p_raw, q_raw, price_role_seq


def _moving_average(values, radius):
    values = _as_numpy(values)
    radius = int(radius)
    if radius <= 0:
        return values.copy()
    width = 2 * radius + 1
    padded = np.pad(values, ((0, 0), (radius, radius)), mode="edge")
    prefix = np.pad(np.cumsum(padded, axis=1, dtype=np.float64), ((0, 0), (1, 0)))
    return ((prefix[:, width:] - prefix[:, :-width]) / width).astype(np.float32)


def _weighted_mean(values, weights, eps):
    numerator = (values * weights).sum(axis=1)
    denominator = weights.sum(axis=1)
    return np.divide(
        numerator,
        denominator + eps,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=denominator > eps,
    ).astype(np.float32)


def _weighted_mean_or(values, weights, default, eps):
    result = _weighted_mean(values, weights, eps)
    return np.where(weights.sum(axis=1) > eps, result, default).astype(np.float32)


def _shift_without_wrap(values, lag):
    shifted = np.zeros_like(values)
    if lag > 0:
        shifted[:, lag:] = values[:, :-lag]
    elif lag < 0:
        shifted[:, :lag] = values[:, -lag:]
    else:
        shifted[:] = values
    return shifted


def _activity_thresholds(p_raw, q_raw, config):
    q_reference = np.percentile(np.abs(q_raw), 95, axis=1)
    p_reference = np.percentile(np.abs(p_raw), 95, axis=1)
    return np.maximum.reduce([
        config.activity_ratio * q_reference,
        config.min_activity_to_load_ratio * p_reference,
        np.full_like(q_reference, config.eps),
    ]).astype(np.float32)


def _activity_components(p_raw, q_raw, config):
    threshold = _activity_thresholds(p_raw, q_raw, config)
    q_effect = q_raw * float(config.charge_sign)
    charge = np.where(q_effect > threshold[:, None], q_effect, 0.0).astype(np.float32)
    discharge = np.where(q_effect < -threshold[:, None], -q_effect, 0.0).astype(np.float32)
    return charge, discharge, threshold


def _signed_runs(q_effect, active_mask):
    active_indices = np.flatnonzero(active_mask)
    if active_indices.size == 0:
        return []

    runs = []
    start = int(active_indices[0])
    previous = start
    sign = float(np.sign(q_effect[start]))
    for point_index in active_indices[1:]:
        point_index = int(point_index)
        current_sign = float(np.sign(q_effect[point_index]))
        if point_index != previous + 1 or current_sign != sign:
            runs.append((start, previous + 1, sign))
            start = point_index
            sign = current_sign
        previous = point_index
    runs.append((start, previous + 1, sign))
    return runs


def _window_change(values, boundary, width):
    if boundary <= 0 or boundary >= len(values):
        return None
    before = values[max(0, boundary - width):boundary]
    after = values[boundary:min(len(values), boundary + width)]
    if before.size == 0 or after.size == 0:
        return None
    return float(np.median(after) - np.median(before))


def _edge_match(delta_p, delta_q, noise, eps):
    magnitude = abs(delta_q)
    sign_match = np.clip(
        delta_p * np.sign(delta_q) / (magnitude + noise + eps),
        0.0,
        1.0,
    )
    amplitude_match = np.exp(-abs(delta_p - delta_q) / (magnitude + noise + eps))
    snr = magnitude / (magnitude + noise + eps)
    return float(np.sqrt(np.clip(snr * (0.4 * sign_match + 0.6 * amplitude_match), 0.0, 1.0)))


def _segment_response_scores(p_raw, evidence, config):
    q_aligned = evidence["q_aligned_raw"]
    q_effect = evidence["q_effect"]
    active = evidence["active_mask"]
    local_noise = evidence["local_noise_raw"]
    point_alignment = evidence["point_alignment"]

    point_scores = np.zeros_like(q_effect, dtype=np.float32)
    point_edge_scores = np.zeros_like(q_effect, dtype=np.float32)
    segment_edge_support = np.zeros_like(q_effect, dtype=np.float32)
    visible_scores = np.zeros(q_effect.shape[0], dtype=np.float32)
    edge_scores = np.zeros(q_effect.shape[0], dtype=np.float32)
    edge_weights = np.zeros(q_effect.shape[0], dtype=np.float32)
    switch_counts = np.zeros(q_effect.shape[0], dtype=np.float32)
    run_counts = np.zeros(q_effect.shape[0], dtype=np.float32)

    for sample_index in range(q_effect.shape[0]):
        runs = _signed_runs(q_effect[sample_index], active[sample_index])
        if not runs:
            continue
        run_counts[sample_index] = len(runs)
        signs = [run[2] for run in runs]
        switch_count = sum(left != right for left, right in zip(signs[:-1], signs[1:]))
        switch_counts[sample_index] = switch_count
        switch_factor = np.exp(-max(switch_count - 3, 0) / 4.0)

        visible_numerator = 0.0
        visible_denominator = 0.0
        edge_numerator = 0.0
        edge_denominator = 0.0
        for start, end, _ in runs:
            run_length = end - start
            run_score = (1.0 - np.exp(-run_length / config.min_run_points)) * switch_factor
            point_scores[sample_index, start:end] = run_score

            segment_weights = np.abs(q_effect[sample_index, start:end])
            segment_energy = float(segment_weights.sum())
            interior_alignment = float(
                np.sum(point_alignment[sample_index, start:end] * segment_weights)
                / (segment_energy + config.eps)
            )

            best_pair_score = 0.0
            best_boundary_scores = []
            best_boundaries = []
            for lag in range(-config.alignment_max_lag, config.alignment_max_lag + 1):
                boundary_scores = []
                boundary_points = []
                for boundary in (start, end):
                    delta_q = _window_change(
                        q_aligned[sample_index], boundary, config.edge_window_points
                    )
                    p_boundary = boundary + lag
                    delta_p = _window_change(
                        p_raw[sample_index], p_boundary, config.edge_window_points
                    )
                    if delta_q is None or delta_p is None:
                        continue
                    noise_start = max(0, p_boundary - config.edge_window_points)
                    noise_end = min(
                        p_raw.shape[1], p_boundary + config.edge_window_points
                    )
                    noise = float(np.median(local_noise[sample_index, noise_start:noise_end]))
                    boundary_scores.append(
                        _edge_match(delta_p, delta_q, noise, config.eps)
                    )
                    boundary_points.append(p_boundary)
                if not boundary_scores:
                    continue
                pair_score = float(
                    np.exp(np.mean(np.log(np.asarray(boundary_scores) + config.eps)))
                )
                if pair_score > best_pair_score:
                    best_pair_score = pair_score
                    best_boundary_scores = boundary_scores
                    best_boundaries = boundary_points

            edge_reliability = min(len(best_boundary_scores) / 2.0, 1.0)
            supportedness = (
                0.70 * interior_alignment
                + 0.30 * best_pair_score * edge_reliability
            ) / (0.70 + 0.30 * edge_reliability + config.eps)
            visible_score = np.sqrt(np.clip(run_score * supportedness, 0.0, 1.0))
            visible_numerator += visible_score * segment_energy
            visible_denominator += segment_energy

            if edge_reliability > 0.0:
                edge_numerator += best_pair_score * segment_energy * edge_reliability
                edge_denominator += segment_energy * edge_reliability
                segment_edge_support[sample_index, start:end] = best_pair_score
                for boundary, boundary_score in zip(best_boundaries, best_boundary_scores):
                    point_edge_scores[sample_index, boundary] = max(
                        point_edge_scores[sample_index, boundary], boundary_score
                    )

        visible_scores[sample_index] = visible_numerator / (
            visible_denominator + config.eps
        )
        if edge_denominator > config.eps:
            edge_scores[sample_index] = edge_numerator / edge_denominator
        edge_weights[sample_index] = edge_denominator / (
            np.abs(q_effect[sample_index]).sum() + config.eps
        )

    return {
        "point_run_score": point_scores,
        "point_edge_score": point_edge_scores,
        "segment_edge_support": segment_edge_support,
        "visible_segment_coherence": visible_scores,
        "edge_pair_match": edge_scores,
        "edge_weight": edge_weights,
        "run_count": run_counts,
        "switch_count": switch_counts,
    }


def _relative_preference(observed, baseline, eps):
    observed = np.clip(observed, 0.0, 1.0)
    baseline = np.clip(baseline, 0.0, 1.0)
    result = np.full_like(observed, 0.5, dtype=np.float32)
    higher = observed >= baseline
    valid_higher = higher & ((1.0 - baseline) > eps)
    valid_lower = (~higher) & (baseline > eps)
    result[valid_higher] = 0.5 + 0.5 * (
        observed[valid_higher] - baseline[valid_higher]
    ) / (1.0 - baseline[valid_higher])
    result[valid_lower] = 0.5 * observed[valid_lower] / baseline[valid_lower]
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def _lag_tolerant_physical_evidence(p_raw, q_raw, config):
    thresholds = _activity_thresholds(p_raw, q_raw, config)
    lags = np.arange(-config.alignment_max_lag, config.alignment_max_lag + 1)
    candidate_scores = []
    candidate_data = []

    for lag in lags:
        q_aligned = _shift_without_wrap(q_raw, int(lag))
        q_effect = q_aligned * float(config.charge_sign)
        active = np.abs(q_effect) > thresholds[:, None]
        background = p_raw - q_aligned
        background_smooth = _moving_average(background, config.background_smooth_radius)
        background_deviation = np.abs(background - background_smooth)
        background_delta = np.abs(
            np.diff(background_smooth, axis=1, prepend=background_smooth[:, :1])
        )
        local_noise = _moving_average(
            background_deviation + background_delta,
            config.noise_smooth_radius,
        )
        background_iqr = np.percentile(background_smooth, 75, axis=1) - np.percentile(
            background_smooth, 25, axis=1
        )
        p_reference = np.percentile(np.abs(p_raw), 95, axis=1)
        noise_floor = np.maximum.reduce([
            config.response_noise_floor_ratio * background_iqr,
            0.002 * p_reference,
            np.full_like(background_iqr, config.eps),
        ])
        local_noise = np.maximum(local_noise, noise_floor[:, None])

        residual = p_raw - background_smooth
        magnitude = np.abs(q_aligned)
        sign_match = np.clip(
            residual * np.sign(q_aligned) / (magnitude + local_noise + config.eps),
            0.0,
            1.0,
        )
        amplitude_match = np.exp(
            -np.abs(residual - q_aligned) / (magnitude + local_noise + config.eps)
        )
        snr = magnitude / (magnitude + local_noise + config.eps)
        point_alignment = np.sqrt(
            np.clip(snr * (0.4 * sign_match + 0.6 * amplitude_match), 0.0, 1.0)
        )
        point_alignment *= active
        score = _weighted_mean(point_alignment, magnitude * active, config.eps)
        candidate_scores.append(score)
        candidate_data.append((
            q_aligned,
            q_effect,
            active,
            background,
            background_smooth,
            local_noise,
            point_alignment,
        ))

    score_matrix = np.stack(candidate_scores, axis=0)
    best_indices = np.argmax(score_matrix, axis=0)
    sample_indices = np.arange(p_raw.shape[0])

    def select(candidate_index):
        stacked = np.stack([values[candidate_index] for values in candidate_data], axis=0)
        return stacked[best_indices, sample_indices]

    return {
        "best_lag": lags[best_indices].astype(np.int8),
        "pq_alignment": score_matrix[best_indices, sample_indices].astype(np.float32),
        "q_aligned_raw": select(0).astype(np.float32),
        "q_effect": select(1).astype(np.float32),
        "active_mask": select(2).astype(bool),
        "background_raw": select(3).astype(np.float32),
        "background_smooth_raw": select(4).astype(np.float32),
        "local_noise_raw": select(5).astype(np.float32),
        "point_alignment": select(6).astype(np.float32),
        "activity_threshold": thresholds.astype(np.float32),
    }


def build_base_load_raw(p_raw, q_raw):
    p_raw, q_raw, _ = _validate_curves(p_raw, q_raw)
    return (p_raw - q_raw).astype(np.float32)


def compute_physical_components(p_raw, q_raw, price_role_seq, config=None):
    config = config or TeacherTargetConfig()
    p_raw, q_raw, price_role_seq = _validate_curves(p_raw, q_raw, price_role_seq)
    charge, discharge, _ = _activity_components(p_raw, q_raw, config)
    base_load = p_raw - q_raw

    charge_energy = charge.sum(axis=1) * 0.25
    discharge_energy = discharge.sum(axis=1) * 0.25
    storage_energy = charge_energy + discharge_energy
    base_energy = np.abs(base_load).sum(axis=1) * 0.25
    energy_balance = 1.0 - np.abs(charge_energy - discharge_energy) / (
        storage_energy + config.eps
    )
    valley_mask = price_role_seq < -0.5
    peak_mask = price_role_seq > 0.5
    charge_in_valley = (charge * valley_mask).sum(axis=1) / (charge.sum(axis=1) + config.eps)
    discharge_in_peak = (discharge * peak_mask).sum(axis=1) / (
        discharge.sum(axis=1) + config.eps
    )
    return {
        "storage_to_base_ratio": (storage_energy / (base_energy + config.eps)).astype(np.float32),
        "charge_energy": charge_energy.astype(np.float32),
        "discharge_energy": discharge_energy.astype(np.float32),
        "energy_balance": np.clip(energy_balance, 0.0, 1.0).astype(np.float32),
        "charge_in_valley_ratio": np.clip(charge_in_valley, 0.0, 1.0).astype(np.float32),
        "discharge_in_peak_ratio": np.clip(discharge_in_peak, 0.0, 1.0).astype(np.float32),
        "base_negative_ratio": (base_load < 0.0).mean(axis=1).astype(np.float32),
    }


def compute_stage2_targets(p_raw, q_raw, price_role_seq, config=None, include_pointwise=False):
    config = config or TeacherTargetConfig()
    p_raw, q_raw, price_role_seq = _validate_curves(p_raw, q_raw, price_role_seq)
    evidence = _lag_tolerant_physical_evidence(p_raw, q_raw, config)
    q_effect = evidence["q_effect"]
    active = evidence["active_mask"]
    charge = np.where((q_effect > 0.0) & active, q_effect, 0.0).astype(np.float32)
    discharge = np.where((q_effect < 0.0) & active, -q_effect, 0.0).astype(np.float32)
    eps = config.eps

    charge_energy = charge.sum(axis=1)
    discharge_energy = discharge.sum(axis=1)
    storage_energy = charge_energy + discharge_energy
    charge_count = (charge > 0.0).sum(axis=1)
    discharge_count = (discharge > 0.0).sum(axis=1)
    active_count = active.sum(axis=1)
    charge_weight = (1.0 - np.exp(-charge_count / 4.0)).astype(np.float32)
    discharge_weight = (1.0 - np.exp(-discharge_count / 4.0)).astype(np.float32)
    activity_weight = (1.0 - np.exp(-active_count / 4.0)).astype(np.float32)

    point_alignment = evidence["point_alignment"]
    segment_data = _segment_response_scores(p_raw, evidence, config)
    point_run_score = segment_data["point_run_score"]
    edge_pair_match = segment_data["edge_pair_match"]
    edge_weight = segment_data["edge_weight"]
    visible_segment_coherence = segment_data["visible_segment_coherence"]
    charge_visible = _weighted_mean(point_alignment, charge, eps)
    discharge_visible = _weighted_mean(point_alignment, discharge, eps)
    response_visibility = (
        charge_visible * charge_weight + discharge_visible * discharge_weight
    ) / (charge_weight + discharge_weight + eps)
    response_confidence = np.clip(
        response_visibility * (0.65 + 0.35 * visible_segment_coherence),
        0.0,
        1.0,
    )

    background = evidence["background_raw"]
    storage_to_base_ratio = storage_energy / (np.abs(background).sum(axis=1) + eps)
    activity_strength = storage_to_base_ratio / (
        storage_to_base_ratio + config.strength_reference_ratio
    )
    storage_evidence_numerator = (
        0.35 * evidence["pq_alignment"]
        + 0.25 * visible_segment_coherence
        + 0.15 * activity_strength
        + 0.25 * edge_pair_match * edge_weight
    )
    storage_evidence_denominator = 0.75 + 0.25 * edge_weight
    storage_evidence = np.clip(
        storage_evidence_numerator / (storage_evidence_denominator + eps),
        0.0,
        1.0,
    )

    physical_targets = np.column_stack([
        evidence["pq_alignment"],
        edge_pair_match,
        visible_segment_coherence,
        activity_strength,
        storage_evidence,
    ]).astype(np.float32)
    physical_weights = np.column_stack([
        activity_weight,
        activity_weight * edge_weight,
        activity_weight,
        activity_weight,
        activity_weight,
    ]).astype(np.float32)

    response_targets = np.column_stack([
        charge_visible,
        discharge_visible,
        response_confidence,
    ]).astype(np.float32)
    response_weights = np.column_stack([
        charge_weight,
        discharge_weight,
        activity_weight,
    ]).astype(np.float32)

    price_rank = np.clip((price_role_seq + 1.0) * 0.5, 0.0, 1.0)
    price_rank = _moving_average(price_rank, config.role_smooth_radius)
    point_evidence = point_alignment * (0.5 + 0.5 * point_run_score)
    supported_charge = charge * point_evidence
    supported_discharge = discharge * point_evidence
    cheapness = 1.0 - price_rank
    expensiveness = price_rank
    charge_preference = _weighted_mean_or(cheapness, supported_charge, 0.5, eps)
    discharge_preference = _weighted_mean_or(
        expensiveness, supported_discharge, 0.5, eps
    )
    cheapness_exposure = cheapness.mean(axis=1).astype(np.float32)
    expensiveness_exposure = expensiveness.mean(axis=1).astype(np.float32)
    charge_role_fit = _relative_preference(
        charge_preference, cheapness_exposure, eps
    )
    discharge_role_fit = _relative_preference(
        discharge_preference, expensiveness_exposure, eps
    )
    average_charge_rank = _weighted_mean_or(
        price_rank, supported_charge, 0.5, eps
    )
    average_discharge_rank = _weighted_mean_or(
        price_rank, supported_discharge, 0.5, eps
    )
    arbitrage_order = np.clip(
        0.5 + 0.5 * (average_discharge_rank - average_charge_rank),
        0.0,
        1.0,
    )
    supported_charge_energy = supported_charge.sum(axis=1)
    supported_discharge_energy = supported_discharge.sum(axis=1)
    supported_storage_energy = supported_charge_energy + supported_discharge_energy
    cycle_completeness = 2.0 * np.minimum(
        supported_charge_energy, supported_discharge_energy
    ) / (
        supported_storage_energy + eps
    )
    role_informativeness = np.ptp(price_rank, axis=1).astype(np.float32)
    charge_behavior_weight = (
        charge_weight * np.sqrt(np.clip(charge_visible, 0.0, 1.0)) * role_informativeness
    )
    discharge_behavior_weight = (
        discharge_weight
        * np.sqrt(np.clip(discharge_visible, 0.0, 1.0))
        * role_informativeness
    )
    role_weight_sum = charge_behavior_weight + discharge_behavior_weight
    role_fit = (
        charge_role_fit * charge_behavior_weight
        + discharge_role_fit * discharge_behavior_weight
    ) / (role_weight_sum + eps)
    role_confidence = 1.0 - (
        1.0 - charge_behavior_weight
    ) * (1.0 - discharge_behavior_weight)
    paired_weight = np.sqrt(
        charge_behavior_weight * discharge_behavior_weight
    ) * cycle_completeness
    policy_numerator = (
        0.70 * role_fit * role_confidence
        + 0.30 * arbitrage_order * paired_weight
    )
    policy_denominator = 0.70 * role_confidence + 0.30 * paired_weight
    policy_pattern = np.divide(
        policy_numerator,
        policy_denominator + eps,
        out=np.full_like(policy_numerator, 0.5, dtype=np.float32),
        where=policy_denominator > eps,
    )
    storage_behavior = np.clip(
        0.50 * storage_evidence
        + 0.25 * response_confidence
        + 0.25 * policy_pattern,
        0.0,
        1.0,
    )
    behavior_targets = np.column_stack([
        charge_role_fit,
        discharge_role_fit,
        arbitrage_order,
        policy_pattern,
        storage_behavior,
    ]).astype(np.float32)
    behavior_weights = np.column_stack([
        charge_behavior_weight,
        discharge_behavior_weight,
        paired_weight,
        role_confidence,
        activity_weight,
    ]).astype(np.float32)

    result = {
        "physical_targets": np.clip(physical_targets, 0.0, 1.0),
        "physical_weights": np.clip(physical_weights, 0.0, 1.0),
        "response_targets": np.clip(response_targets, 0.0, 1.0),
        "response_weights": np.clip(response_weights, 0.0, 1.0),
        "behavior_targets": np.clip(behavior_targets, 0.0, 1.0),
        "behavior_weights": np.clip(behavior_weights, 0.0, 1.0),
        "auxiliary": {
            "storage_to_base_ratio": storage_to_base_ratio.astype(np.float32),
            "charge_energy": charge_energy.astype(np.float32),
            "discharge_energy": discharge_energy.astype(np.float32),
            "cycle_completeness": np.clip(cycle_completeness, 0.0, 1.0).astype(np.float32),
            "response_visibility": response_visibility.astype(np.float32),
            "edge_pair_match": edge_pair_match.astype(np.float32),
            "edge_weight": edge_weight.astype(np.float32),
            "visible_segment_coherence": visible_segment_coherence.astype(np.float32),
            "best_lag": evidence["best_lag"],
            "activity_threshold": evidence["activity_threshold"],
            "run_count": segment_data["run_count"],
            "switch_count": segment_data["switch_count"],
            "average_charge_rank": average_charge_rank.astype(np.float32),
            "average_discharge_rank": average_discharge_rank.astype(np.float32),
            "cheapness_exposure": cheapness_exposure,
            "expensiveness_exposure": expensiveness_exposure,
            "role_fit": role_fit.astype(np.float32),
            "policy_pattern": policy_pattern.astype(np.float32),
            "charge_activity_ratio": (charge > 0.0).mean(axis=1).astype(np.float32),
            "discharge_activity_ratio": (discharge > 0.0).mean(axis=1).astype(np.float32),
        },
    }
    if include_pointwise:
        point_role_fit = np.where(q_effect >= 0.0, 1.0 - price_rank, price_rank)
        point_behavior_evidence = point_evidence * point_role_fit * active
        observed_response = p_raw - evidence["background_smooth_raw"]
        response_error = observed_response - evidence["q_aligned_raw"]
        response_center = evidence["background_smooth_raw"] + evidence["q_aligned_raw"]
        result["pointwise"] = {
            "q_aligned_raw": evidence["q_aligned_raw"],
            "background_raw": evidence["background_raw"],
            "background_smooth_raw": evidence["background_smooth_raw"],
            "local_noise_raw": evidence["local_noise_raw"],
            "observed_response_raw": observed_response.astype(np.float32),
            "response_error_raw": response_error.astype(np.float32),
            "response_delta_band_lower_raw": (
                evidence["q_aligned_raw"] - evidence["local_noise_raw"]
            ).astype(np.float32),
            "response_delta_band_upper_raw": (
                evidence["q_aligned_raw"] + evidence["local_noise_raw"]
            ).astype(np.float32),
            "response_band_hit": (
                (np.abs(response_error) <= evidence["local_noise_raw"]) & active
            ),
            "response_center_raw": response_center.astype(np.float32),
            "response_band_lower_raw": (response_center - evidence["local_noise_raw"]).astype(np.float32),
            "response_band_upper_raw": (response_center + evidence["local_noise_raw"]).astype(np.float32),
            "point_alignment": point_alignment,
            "point_run_score": point_run_score,
            "point_edge_score": segment_data["point_edge_score"],
            "segment_edge_support": segment_data["segment_edge_support"],
            "point_evidence": point_evidence.astype(np.float32),
            "point_role_fit": point_role_fit.astype(np.float32),
            "point_behavior_evidence": point_behavior_evidence.astype(np.float32),
            "supported_charge_raw": supported_charge.astype(np.float32),
            "supported_discharge_raw": supported_discharge.astype(np.float32),
            "charge_mask": (charge > 0.0),
            "discharge_mask": (discharge > 0.0),
            "active_mask": active,
        }
    return result


def summarize_stage2_targets(target_data):
    rows = []
    groups = (
        ("physical", PHYSICAL_TARGET_NAMES, target_data["physical_targets"], target_data["physical_weights"]),
        ("response", RESPONSE_TARGET_NAMES, target_data["response_targets"], target_data["response_weights"]),
        (
            "behavior",
            BEHAVIOR_TARGET_NAMES,
            target_data["behavior_targets"],
            target_data["behavior_weights"],
        ),
    )
    for group, names, targets, weights in groups:
        for index, name in enumerate(names):
            values = targets[:, index]
            target_weights = weights[:, index]
            active = target_weights > 1e-6
            observed = values[active] if active.any() else values
            rows.append({
                "group": group,
                "name": name,
                "mean": float(observed.mean()),
                "std": float(observed.std()),
                "q05": float(np.quantile(observed, 0.05)),
                "q50": float(np.quantile(observed, 0.50)),
                "q95": float(np.quantile(observed, 0.95)),
                "weight_mean": float(target_weights.mean()),
                "active_ratio": float(active.mean()),
            })
    return rows


@dataclass
class FixedQuantileScaler:
    names: list
    lower: np.ndarray
    upper: np.ndarray
    directions: list
    eps: float = 1e-6

    @classmethod
    def fit(cls, targets, train_indices, lower_quantile=0.05, upper_quantile=0.95,
            directions=None, eps=1e-6):
        names = list(targets)
        if not names:
            raise ValueError("没有可拟合的目标")
        train_indices = np.asarray(train_indices, dtype=np.int64)
        if train_indices.size == 0:
            raise ValueError("训练索引为空")
        matrix = np.column_stack([_as_numpy(targets[name]).reshape(-1) for name in names])
        directions = directions or {name: "higher" for name in names}
        direction_list = [directions.get(name, "higher") for name in names]
        if any(direction not in {"higher", "lower"} for direction in direction_list):
            raise ValueError("方向仅支持 higher 或 lower")
        train_matrix = matrix[train_indices]
        lower = np.quantile(train_matrix, lower_quantile, axis=0).astype(np.float32)
        upper = np.quantile(train_matrix, upper_quantile, axis=0).astype(np.float32)
        upper = np.maximum(upper, lower + eps)
        return cls(names, lower, upper, direction_list, eps=eps)

    def transform(self, targets):
        matrix = np.column_stack([_as_numpy(targets[name]).reshape(-1) for name in self.names])
        scaled = np.clip((matrix - self.lower) / (self.upper - self.lower + self.eps), 0.0, 1.0)
        for index, direction in enumerate(self.directions):
            if direction == "lower":
                scaled[:, index] = 1.0 - scaled[:, index]
        return {name: scaled[:, index].astype(np.float32) for index, name in enumerate(self.names)}

    def to_dict(self):
        return {
            "names": self.names,
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
            "directions": self.directions,
            "eps": self.eps,
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(
            names=list(payload["names"]),
            lower=np.asarray(payload["lower"], dtype=np.float32),
            upper=np.asarray(payload["upper"], dtype=np.float32),
            directions=list(payload["directions"]),
            eps=float(payload.get("eps", 1e-6)),
        )


def _stage2_cache_metadata(data, config):
    return {
        "target_pipeline_version": TARGET_PIPELINE_VERSION,
        "data_pipeline_version": data.get("pipeline_version", ""),
        "data_sha256": data.get("data_sha256", ""),
        "split_manifest_sha256": data.get("split_manifest_sha256", ""),
        "sample_count": int(len(data["meta"])),
        "config": asdict(config),
        "physical_target_names": list(PHYSICAL_TARGET_NAMES),
        "response_target_names": list(RESPONSE_TARGET_NAMES),
        "behavior_target_names": list(BEHAVIOR_TARGET_NAMES),
    }


def save_stage2_target_cache(path, target_data, metadata):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "physical_targets": target_data["physical_targets"],
        "physical_weights": target_data["physical_weights"],
        "response_targets": target_data["response_targets"],
        "response_weights": target_data["response_weights"],
        "behavior_targets": target_data["behavior_targets"],
        "behavior_weights": target_data["behavior_weights"],
        "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False)),
    }
    arrays.update({f"aux__{name}": values for name, values in target_data["auxiliary"].items()})
    np.savez_compressed(output_path, **arrays)


def load_stage2_target_cache(path):
    with np.load(path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata_json"].item()))
        auxiliary = {
            key.removeprefix("aux__"): cache[key].copy()
            for key in cache.files if key.startswith("aux__")
        }
        return {
            "physical_targets": cache["physical_targets"].copy(),
            "physical_weights": cache["physical_weights"].copy(),
            "response_targets": cache["response_targets"].copy(),
            "response_weights": cache["response_weights"].copy(),
            "behavior_targets": cache["behavior_targets"].copy(),
            "behavior_weights": cache["behavior_weights"].copy(),
            "auxiliary": auxiliary,
            "metadata": metadata,
        }


def build_stage2_target_cache(data, output_path, config=None, force=False):
    config = config or TeacherTargetConfig()
    expected_metadata = _stage2_cache_metadata(data, config)
    output_path = Path(output_path)
    if output_path.is_file() and not force:
        cached = load_stage2_target_cache(output_path)
        metadata = cached["metadata"]
        comparable = (
            "target_pipeline_version",
            "data_pipeline_version",
            "data_sha256",
            "split_manifest_sha256",
            "sample_count",
            "config",
        )
        if all(metadata.get(key) == expected_metadata.get(key) for key in comparable):
            return cached

    price_role_seq = _as_numpy(data["X"][:, 3, :])
    target_data = compute_stage2_targets(
        data["P_raw"],
        data["Q_raw"],
        price_role_seq,
        config=config,
    )
    save_stage2_target_cache(output_path, target_data, expected_metadata)
    target_data["metadata"] = expected_metadata
    return target_data
