import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

try:
    from storage.data import (
        build_price_features,
        build_price_jump_seq,
        scale_pq,
    )
except ImportError:
    try:
        from .data import build_price_features, build_price_jump_seq, scale_pq
    except ImportError:
        from data import build_price_features, build_price_jump_seq, scale_pq


POLICY_TRANSFER_VERSION = "policy-relocation-v3-exposure-normalized"


@dataclass(frozen=True)
class PolicyTransferConfig:
    charge_sign: int = 1
    activity_ratio: float = 0.02
    min_activity_to_load_ratio: float = 0.002
    min_active_points: int = 2
    min_activity_coverage: float = 0.70
    min_energy_retention: float = 0.70
    min_target_pattern: float = 0.60
    random_shift_probability: float = 0.50
    shift_step: int = 4
    min_shift_points: int = 8
    eps: float = 1e-6

    def __post_init__(self):
        if self.charge_sign not in {-1, 1}:
            raise ValueError("charge_sign 必须为 -1 或 1")
        if self.min_active_points < 1:
            raise ValueError("min_active_points 必须大于 0")
        if self.shift_step < 1 or self.min_shift_points < 1:
            raise ValueError("政策位移步长必须大于 0")
        for name in (
            "min_activity_coverage",
            "min_energy_retention",
            "min_target_pattern",
            "random_shift_probability",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} 必须位于 [0, 1]")


def _as_curve(values, name):
    curve = np.asarray(values, dtype=np.float32).reshape(-1)
    if curve.size != 96:
        raise ValueError(f"{name} 必须是 96 点曲线")
    if not np.isfinite(curve).all():
        raise ValueError(f"{name} 包含非有限值")
    return curve


def _activity_threshold(p_raw, q_raw, config):
    q_reference = float(np.percentile(np.abs(q_raw), 95))
    p_reference = float(np.percentile(np.abs(p_raw), 95))
    return max(
        config.activity_ratio * q_reference,
        config.min_activity_to_load_ratio * p_reference,
        config.eps,
    )


def _contiguous_runs(mask):
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    edges = np.diff(padded)
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _relative_preference(observed, baseline, eps):
    observed = float(np.clip(observed, 0.0, 1.0))
    baseline = float(np.clip(baseline, 0.0, 1.0))
    if observed >= baseline:
        if 1.0 - baseline <= eps:
            return 0.5
        return 0.5 + 0.5 * (observed - baseline) / (1.0 - baseline)
    if baseline <= eps:
        return 0.5
    return 0.5 * observed / baseline


def _role_pattern_proxy(q_raw, role_seq, threshold, charge_sign):
    q_effect = q_raw * float(charge_sign)
    charge = np.where(q_effect > threshold, q_effect, 0.0)
    discharge = np.where(q_effect < -threshold, -q_effect, 0.0)
    charge_energy = float(charge.sum())
    discharge_energy = float(discharge.sum())
    rank = np.clip((role_seq + 1.0) * 0.5, 0.0, 1.0)

    scores = []
    weights = []
    if charge_energy > 0.0:
        observed = float(((1.0 - rank) * charge).sum() / charge_energy)
        scores.append(_relative_preference(observed, float((1.0 - rank).mean()), 1e-6))
        weights.append(charge_energy)
    if discharge_energy > 0.0:
        observed = float((rank * discharge).sum() / discharge_energy)
        scores.append(_relative_preference(observed, float(rank.mean()), 1e-6))
        weights.append(discharge_energy)
    if not scores:
        return 0.0

    role_fit = float(np.average(scores, weights=weights))
    if charge_energy <= 0.0 or discharge_energy <= 0.0:
        return role_fit
    charge_rank = float((rank * charge).sum() / charge_energy)
    discharge_rank = float((rank * discharge).sum() / discharge_energy)
    arbitrage_order = np.clip(0.5 + 0.5 * (discharge_rank - charge_rank), 0.0, 1.0)
    return float(0.70 * role_fit + 0.30 * arbitrage_order)


def _place_runs(component, preferred_score, base_load, occupied, discharge, rng):
    source_mask = np.abs(component) > 0.0
    runs = _contiguous_runs(source_mask)
    runs.sort(key=lambda bounds: -float(np.abs(component[bounds[0]:bounds[1]]).sum()))
    mapped = np.zeros(96, dtype=np.float32)
    source_energy = float(np.abs(component).sum())
    mapped_energy = 0.0

    for start, end in runs:
        values = component[start:end]
        offsets = np.arange(len(values), dtype=np.int64)
        best = None
        for candidate_start in range(96):
            indices = (candidate_start + offsets) % 96
            if occupied[indices].any():
                continue
            retention = 1.0
            if discharge:
                magnitude = np.abs(values)
                capacity = np.maximum(base_load[indices], 0.0)
                retention = float(np.min(capacity / (magnitude + 1e-8)))
                retention = float(np.clip(retention, 0.0, 1.0))
            role_score = float(
                np.average(preferred_score[indices], weights=np.abs(values) + 1e-8)
            )
            score = role_score + 0.15 * retention + float(rng.uniform(0.0, 1e-5))
            if best is None or score > best[0]:
                best = (score, indices, retention)
        if best is None:
            continue
        _, indices, retention = best
        placed = values * retention
        mapped[indices] = placed
        occupied[indices] = True
        mapped_energy += float(np.abs(placed).sum())

    return mapped, source_energy, mapped_energy


def relocate_storage_response(p_raw, q_raw, target_role_seq, config=None, rng=None):
    config = config or PolicyTransferConfig()
    rng = rng or np.random.default_rng()
    p_raw = _as_curve(p_raw, "P_raw")
    q_raw = _as_curve(q_raw, "Q_raw")
    target_role_seq = _as_curve(target_role_seq, "target_role_seq")
    threshold = _activity_threshold(p_raw, q_raw, config)
    q_effect = q_raw * float(config.charge_sign)
    active = np.abs(q_effect) > threshold
    charge = np.where(q_effect > threshold, q_raw, 0.0).astype(np.float32)
    discharge = np.where(q_effect < -threshold, q_raw, 0.0).astype(np.float32)
    base_load = (p_raw - q_raw).astype(np.float32)
    rank = np.clip((target_role_seq + 1.0) * 0.5, 0.0, 1.0)
    occupied = np.zeros(96, dtype=bool)

    charge_new, charge_source, charge_target = _place_runs(
        charge,
        1.0 - rank,
        base_load,
        occupied,
        discharge=False,
        rng=rng,
    )
    discharge_new, discharge_source, discharge_target = _place_runs(
        discharge,
        rank,
        base_load,
        occupied,
        discharge=True,
        rng=rng,
    )
    q_new = (charge_new + discharge_new).astype(np.float32)
    p_new = (base_load + q_new).astype(np.float32)

    total_energy = float(np.abs(q_raw).sum())
    active_energy = float(np.abs(q_raw[active]).sum())
    source_energy = charge_source + discharge_source
    target_energy = charge_target + discharge_target
    activity_coverage = active_energy / max(total_energy, config.eps)
    energy_retention = target_energy / max(source_energy, config.eps)
    target_pattern = _role_pattern_proxy(
        q_new,
        target_role_seq,
        threshold,
        config.charge_sign,
    )
    active_points = int(active.sum())
    accepted = bool(
        active_points >= config.min_active_points
        and activity_coverage >= config.min_activity_coverage
        and energy_retention >= config.min_energy_retention
        and target_pattern >= config.min_target_pattern
        and float(p_new.min()) >= -1e-5
        and np.isfinite(p_new).all()
        and np.isfinite(q_new).all()
    )
    quality = float(
        np.sqrt(np.clip(activity_coverage * energy_retention, 0.0, 1.0))
        * np.clip((target_pattern - 0.5) / 0.5, 0.0, 1.0)
    )
    if not accepted:
        quality = 0.0

    return {
        "P_base_raw": base_load,
        "P_new_raw": np.maximum(p_new, 0.0).astype(np.float32),
        "Q_new_raw": q_new,
        "accepted": accepted,
        "quality": quality,
        "active_points": active_points,
        "activity_coverage": float(activity_coverage),
        "energy_retention": float(energy_retention),
        "target_pattern": float(target_pattern),
    }


def _target_policy_features(month, config, rng):
    features = build_price_features(month=int(month), policy_name="new")
    role = features["price_role_seq"].copy()
    shift = 0
    if rng.random() < config.random_shift_probability:
        candidates = np.arange(
            config.min_shift_points,
            96 - config.min_shift_points + 1,
            config.shift_step,
            dtype=np.int64,
        )
        shift = int(rng.choice(candidates))
        role = np.roll(role, shift)
    return {
        "season_flag": features["season_flag"].astype(np.float32),
        "price_role_seq": role.astype(np.float32),
        "price_role_jump": build_price_jump_seq(role),
        "policy_shift": shift,
    }


def _mismatched_role(q_raw, target_role, threshold, config, rng):
    candidates = []
    for shift in range(config.min_shift_points, 96, config.shift_step):
        role = np.roll(target_role, shift)
        if np.array_equal(role, target_role):
            continue
        score = _role_pattern_proxy(q_raw, role, threshold, config.charge_sign)
        candidates.append((score, role))
    if not candidates:
        return np.roll(target_role, 24).astype(np.float32)
    candidates.sort(key=lambda item: item[0])
    choice_count = min(3, len(candidates))
    return candidates[int(rng.integers(choice_count))][1].astype(np.float32)


def build_policy_transfer_sample(p_raw, q_raw, month, q_global_scale, config=None, rng=None):
    config = config or PolicyTransferConfig()
    rng = rng or np.random.default_rng()
    target = _target_policy_features(month, config, rng)
    transfer = relocate_storage_response(
        p_raw,
        q_raw,
        target["price_role_seq"],
        config=config,
        rng=rng,
    )
    if transfer["accepted"]:
        p_new_raw = transfer["P_new_raw"]
        q_new_raw = transfer["Q_new_raw"]
        role = target["price_role_seq"]
        jump = target["price_role_jump"]
        season = target["season_flag"]
    else:
        p_new_raw = _as_curve(p_raw, "P_raw")
        q_new_raw = _as_curve(q_raw, "Q_raw")
        fallback = build_price_features(month=int(month), policy_name="old")
        role = fallback["price_role_seq"]
        jump = fallback["price_role_jump"]
        season = fallback["season_flag"]

    p_scaled, q_scaled, _ = scale_pq(
        p_new_raw,
        q_new_raw,
        q_global_scale=float(q_global_scale),
    )
    threshold = _activity_threshold(p_new_raw, q_new_raw, config)
    mismatch_role = _mismatched_role(q_new_raw, role, threshold, config, rng)
    positive = np.stack([p_scaled, q_scaled, season, role, jump], axis=0).astype(np.float32)
    return {
        **transfer,
        "P_new_raw": p_new_raw.astype(np.float32),
        "Q_new_raw": q_new_raw.astype(np.float32),
        "positive_inputs": positive,
        "target_role_seq": role.astype(np.float32),
        "mismatch_role_seq": mismatch_role,
        "mismatch_role_jump": build_price_jump_seq(mismatch_role),
        "policy_shift": int(target["policy_shift"]),
    }


def _cache_metadata(data, config):
    return {
        "policy_transfer_version": POLICY_TRANSFER_VERSION,
        "data_pipeline_version": data.get("pipeline_version", ""),
        "data_sha256": data.get("data_sha256", ""),
        "split_manifest_sha256": data.get("split_manifest_sha256", ""),
        "sample_count": int(len(data["meta"])),
        "q_global_scale": float(data["normalization"]["q_global_scale"]),
        "config": asdict(config),
    }


def load_policy_transfer_cache(path):
    with np.load(path, allow_pickle=False) as cache:
        return {
            key: cache[key].copy()
            for key in cache.files
            if key != "metadata_json"
        } | {"metadata": json.loads(str(cache["metadata_json"].item()))}


def build_policy_transfer_cache(data, output_path, config=None, force=False, seed=1111):
    config = config or PolicyTransferConfig()
    expected = _cache_metadata(data, config)
    output_path = Path(output_path)
    if output_path.is_file() and not force:
        cached = load_policy_transfer_cache(output_path)
        comparable = tuple(expected)
        if all(cached["metadata"].get(key) == expected[key] for key in comparable):
            return cached

    sample_count = len(data["meta"])
    positive_inputs = np.empty((sample_count, 5, 96), dtype=np.float32)
    positive_p_raw = np.empty((sample_count, 96), dtype=np.float32)
    positive_q_raw = np.empty((sample_count, 96), dtype=np.float32)
    target_role_seq = np.empty((sample_count, 96), dtype=np.float32)
    mismatch_role_seq = np.empty((sample_count, 96), dtype=np.float32)
    mismatch_role_jump = np.empty((sample_count, 96), dtype=np.float32)
    transfer_weight = np.zeros(sample_count, dtype=np.float32)
    accepted = np.zeros(sample_count, dtype=np.uint8)
    policy_shift = np.zeros(sample_count, dtype=np.int16)
    activity_coverage = np.zeros(sample_count, dtype=np.float32)
    energy_retention = np.zeros(sample_count, dtype=np.float32)
    target_pattern = np.zeros(sample_count, dtype=np.float32)
    rng = np.random.default_rng(seed)
    q_global_scale = float(data["normalization"]["q_global_scale"])

    for index in range(sample_count):
        sample = build_policy_transfer_sample(
            data["P_raw"][index],
            data["Q_raw"][index],
            int(data["meta"].iloc[index]["month"]),
            q_global_scale,
            config=config,
            rng=rng,
        )
        positive_inputs[index] = sample["positive_inputs"]
        positive_p_raw[index] = sample["P_new_raw"]
        positive_q_raw[index] = sample["Q_new_raw"]
        target_role_seq[index] = sample["target_role_seq"]
        mismatch_role_seq[index] = sample["mismatch_role_seq"]
        mismatch_role_jump[index] = sample["mismatch_role_jump"]
        transfer_weight[index] = float(sample["quality"])
        accepted[index] = int(sample["accepted"])
        policy_shift[index] = int(sample["policy_shift"])
        activity_coverage[index] = float(sample["activity_coverage"])
        energy_retention[index] = float(sample["energy_retention"])
        target_pattern[index] = float(sample["target_pattern"])

    result = {
        "positive_inputs": positive_inputs,
        "positive_p_raw": positive_p_raw,
        "positive_q_raw": positive_q_raw,
        "target_role_seq": target_role_seq,
        "mismatch_role_seq": mismatch_role_seq,
        "mismatch_role_jump": mismatch_role_jump,
        "transfer_weight": transfer_weight,
        "accepted": accepted,
        "policy_shift": policy_shift,
        "activity_coverage": activity_coverage,
        "energy_retention": energy_retention,
        "target_pattern": target_pattern,
        "metadata": expected,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in result.items() if key != "metadata"}
    arrays["metadata_json"] = np.asarray(json.dumps(expected, ensure_ascii=False))
    np.savez_compressed(output_path, **arrays)
    return result


def build_policy_transfer_pair(
    P_raw,
    Q_raw,
    month,
    source_policy="old",
    target_policy="new",
    day_type="workday",
    user_type="normal",
    charge_sign=1,
    max_power_scale=1.25,
    min_energy_retention=0.70,
    min_source_role_coverage=0.0,
    require_both=False,
):
    del max_power_scale
    source = build_price_features(
        month=int(month),
        policy_name=source_policy,
        day_type=day_type,
        user_type=user_type,
    )
    target = build_price_features(
        month=int(month),
        policy_name=target_policy,
        day_type=day_type,
        user_type=user_type,
    )
    config = PolicyTransferConfig(
        charge_sign=charge_sign,
        min_energy_retention=min_energy_retention,
        min_activity_coverage=min_source_role_coverage,
        random_shift_probability=0.0,
    )
    transfer = relocate_storage_response(
        P_raw,
        Q_raw,
        target["price_role_seq"],
        config=config,
        rng=np.random.default_rng(0),
    )
    q_effect = _as_curve(Q_raw, "Q_raw") * float(charge_sign)
    threshold = _activity_threshold(
        _as_curve(P_raw, "P_raw"), _as_curve(Q_raw, "Q_raw"), config
    )
    charge_source = float(np.where(q_effect > threshold, q_effect, 0.0).sum())
    discharge_source = float(np.where(q_effect < -threshold, -q_effect, 0.0).sum())
    q_new_effect = transfer["Q_new_raw"] * float(charge_sign)
    charge_target = float(np.clip(q_new_effect, 0.0, None).sum())
    discharge_target = float(np.clip(-q_new_effect, 0.0, None).sum())
    if require_both and (charge_source <= 0.0 or discharge_source <= 0.0):
        transfer["accepted"] = False
        transfer["quality"] = 0.0

    p_reference = max(float(np.percentile(np.abs(P_raw), 95)), config.eps)
    q_global_scale = max(
        float(np.percentile(np.abs(Q_raw) / p_reference, 95)),
        config.eps,
    )
    p_new, q_new, _ = scale_pq(
        transfer["P_new_raw"], transfer["Q_new_raw"], q_global_scale
    )
    p_base, q_zero, _ = scale_pq(
        transfer["P_base_raw"], np.zeros(96, dtype=np.float32), q_global_scale
    )
    positive = np.stack([
        p_new,
        q_new,
        target["season_flag"],
        target["price_role_seq"],
        target["price_role_jump"],
    ]).astype(np.float32)
    pseudo_negative = np.stack([
        p_base,
        q_zero,
        target["season_flag"],
        target["price_role_seq"],
        target["price_role_jump"],
    ]).astype(np.float32)
    source_selected = charge_source + discharge_source
    total_source = float(np.abs(Q_raw).sum())
    transfer.update({
        "positive_5ch": positive,
        "pseudo_negative_5ch": pseudo_negative,
        "source_season": source["policy_cfg"]["season_name"],
        "target_season": target["policy_cfg"]["season_name"],
        "mapping_quality": float(transfer["energy_retention"]),
        "total_source_energy": total_source,
        "selected_source_energy": source_selected,
        "source_role_coverage": source_selected / max(total_source, config.eps),
        "charge_source_energy": charge_source,
        "charge_target_energy": charge_target,
        "charge_energy_retention": charge_target / max(charge_source, config.eps),
        "discharge_source_energy": discharge_source,
        "discharge_target_energy": discharge_target,
        "discharge_energy_retention": discharge_target / max(discharge_source, config.eps),
    })
    return transfer
