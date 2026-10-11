"""
Stage-2 教师模型完整诊断（增强版）

适配：
- add_test_st2.py 中的 teacher-multitask-v2-floor-cov-policy-calibrated
- teacher_targets.py 中的 teacher-storage-behavior-v5-pq-visible-role-normalized
- policy_transfer.py 中的 policy-relocation-v3-exposure-normalized

主要输出：
1. diagnostics_full.json / diagnostics_summary.md
2. integrity_checks.csv
3. target_distributions.csv / target_metrics.csv / calibration_bins.csv
4. per_sample_predictions.csv / user_metrics.csv / user_level_summary.csv
5. pseudo_a_pairs.csv / pseudo_a_summary.csv
6. policy_counterfactual_pairs.csv / policy_counterfactual_summary.csv /
   policy_counterfactual_strata.csv
7. representation_health.json / representation_spectrum.csv
8. core_head_metrics.csv
9. p_only_distillability.json（默认执行，可用 --skip_p_only_probe 关闭）
10. plots/ 下的目标拟合、反事实、表征谱、多头与用户级图

说明：
- A 类错配与 B 类反事实都不是现实负类；相关 AUC/accuracy 只作反事实可分性诊断。
- val 用于正式验收；test 仅报告，不参与阈值调整。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
    SKLEARN_AVAILABLE = True
except Exception:
    HistGradientBoostingClassifier = None
    HistGradientBoostingRegressor = None
    mean_absolute_error = None
    r2_score = None
    roc_auc_score = None
    SKLEARN_AVAILABLE = False

try:
    from TCN.adding_problem import add_test_st2 as st2
    from TCN.adding_problem.policy_transfer import (
        POLICY_TRANSFER_VERSION,
        PolicyTransferConfig,
        build_policy_transfer_cache,
    )
    from TCN.adding_problem.teacher_targets import (
        BEHAVIOR_TARGET_NAMES,
        PHYSICAL_TARGET_NAMES,
        RESPONSE_TARGET_NAMES,
        TARGET_PIPELINE_VERSION,
        TeacherTargetConfig,
        build_stage2_target_cache,
        compute_stage2_targets,
    )
    from TCN.adding_problem.utils import load_storage_data
except ImportError:
    import add_test_st2 as st2
    from policy_transfer import (
        POLICY_TRANSFER_VERSION,
        PolicyTransferConfig,
        build_policy_transfer_cache,
    )
    from teacher_targets import (
        BEHAVIOR_TARGET_NAMES,
        PHYSICAL_TARGET_NAMES,
        RESPONSE_TARGET_NAMES,
        TARGET_PIPELINE_VERSION,
        TeacherTargetConfig,
        build_stage2_target_cache,
        compute_stage2_targets,
    )
    from utils import load_storage_data


SCRIPT_DIR = Path(__file__).resolve().parent
EPS = 1e-12
CORE_SPECS = (
    ("physical", "storage_evidence"),
    ("response", "response_confidence"),
    ("behavior", "storage_behavior"),
)
GROUP_SPECS = (
    ("physical", PHYSICAL_TARGET_NAMES, "physical_targets", "physical_weights"),
    ("response", RESPONSE_TARGET_NAMES, "response_targets", "response_weights"),
    ("behavior", BEHAVIOR_TARGET_NAMES, "behavior_targets", "behavior_weights"),
)


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return SCRIPT_DIR / candidate


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"无法 JSON 序列化：{type(value)!r}")


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )


def save_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = [dict(row) for row in rows]
    if not rows:
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_or_nan(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not valid.any():
        return 0.0
    weight_sum = float(weights[valid].sum())
    return float(np.sum(values[valid] * weights[valid]) / max(weight_sum, EPS))


def weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not valid.any():
        return 0.0
    mean = weighted_mean(values[valid], weights[valid])
    variance = weighted_mean(np.square(values[valid] - mean), weights[valid])
    return float(np.sqrt(max(variance, 0.0)))


def weighted_corr(values: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = (
        np.isfinite(values)
        & np.isfinite(targets)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    if valid.sum() < 2:
        return 0.0
    values = values[valid]
    targets = targets[valid]
    weights = weights[valid]
    weight_sum = float(weights.sum())
    value_mean = float(np.sum(weights * values) / max(weight_sum, EPS))
    target_mean = float(np.sum(weights * targets) / max(weight_sum, EPS))
    value_centered = values - value_mean
    target_centered = targets - target_mean
    covariance = float(np.sum(weights * value_centered * target_centered) / max(weight_sum, EPS))
    value_var = float(np.sum(weights * np.square(value_centered)) / max(weight_sum, EPS))
    target_var = float(np.sum(weights * np.square(target_centered)) / max(weight_sum, EPS))
    if value_var <= EPS or target_var <= EPS:
        return 0.0
    return float(covariance / np.sqrt(value_var * target_var))


def average_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average
        start = end
    return ranks


def weighted_spearman(values: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = (
        np.isfinite(values)
        & np.isfinite(targets)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    if valid.sum() < 2:
        return 0.0
    return weighted_corr(
        average_rank(values[valid]),
        average_rank(targets[valid]),
        weights[valid],
    )


def weighted_error_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> Dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = (
        np.isfinite(prediction)
        & np.isfinite(target)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    if not valid.any():
        return {
            "pearson": 0.0,
            "spearman": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "pred_std": 0.0,
            "target_std": 0.0,
            "std_ratio": 0.0,
            "slope": 0.0,
            "intercept": 0.0,
            "effective_n": 0.0,
            "weight_sum": 0.0,
        }
    p = prediction[valid]
    t = target[valid]
    w = weights[valid]
    w_sum = float(w.sum())
    absolute_error = np.abs(p - t)
    squared_error = np.square(p - t)
    p_mean = weighted_mean(p, w)
    t_mean = weighted_mean(t, w)
    target_variance = weighted_mean(np.square(t - t_mean), w)
    covariance = weighted_mean((t - t_mean) * (p - p_mean), w)
    slope = covariance / max(target_variance, EPS)
    intercept = p_mean - slope * t_mean
    effective_n = w_sum * w_sum / max(float(np.square(w).sum()), EPS)
    target_std = weighted_std(t, w)
    pred_std = weighted_std(p, w)
    return {
        "pearson": weighted_corr(p, t, w),
        "spearman": weighted_spearman(p, t, w),
        "mae": weighted_mean(absolute_error, w),
        "rmse": float(np.sqrt(weighted_mean(squared_error, w))),
        "pred_std": pred_std,
        "target_std": target_std,
        "std_ratio": pred_std / max(target_std, EPS),
        "slope": float(slope),
        "intercept": float(intercept),
        "effective_n": float(effective_n),
        "weight_sum": w_sum,
    }


def weighted_quantile(values: np.ndarray, quantiles: Sequence[float], weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    quantiles = np.asarray(quantiles, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not valid.any():
        return np.zeros(len(quantiles), dtype=np.float64)
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= max(float(weights.sum()), EPS)
    return np.interp(quantiles, cumulative, values)


def effective_rank(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2:
        return 0.0
    centered = values - values.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    energy = np.square(singular_values)
    probability = energy / max(float(energy.sum()), EPS)
    probability = np.clip(probability, 1e-12, 1.0)
    return float(np.exp(-np.sum(probability * np.log(probability))))


def representation_statistics(
    values: np.ndarray,
    target_std: float = 0.08,
    covariance_weight: float = 5.0,
) -> Tuple[Dict[str, float], np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2:
        return {}, np.zeros(0, dtype=np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    dimension_std = values.std(axis=0)
    normalized = values / (np.linalg.norm(values, axis=1, keepdims=True) + EPS)
    normalized_centered = normalized - normalized.mean(axis=0, keepdims=True)
    normalized_std = normalized.std(axis=0)
    covariance = normalized_centered.T @ normalized_centered / max(len(values) - 1, 1)
    offdiag = covariance - np.diag(np.diag(covariance))
    correlation = covariance / (
        np.sqrt(np.diag(covariance)[:, None] * np.diag(covariance)[None, :]) + 1e-6
    )
    correlation_offdiag = correlation - np.diag(np.diag(correlation))
    singular_values = np.linalg.svd(centered, compute_uv=False)
    energy = np.square(singular_values)
    energy_total = max(float(energy.sum()), EPS)
    probability = energy / energy_total
    cumulative = np.cumsum(probability)
    k90 = int(np.searchsorted(cumulative, 0.90) + 1)
    k99 = int(np.searchsorted(cumulative, 0.99) + 1)
    variance_floor = float(np.maximum(target_std - normalized_std, 0.0).mean())
    covariance_loss = float(np.square(offdiag).sum() / max(values.shape[1], 1))
    stats = {
        "samples": int(len(values)),
        "dimensions": int(values.shape[1]),
        "raw_std_min": float(dimension_std.min()),
        "raw_std_mean": float(dimension_std.mean()),
        "raw_std_max": float(dimension_std.max()),
        "normalized_std_min": float(normalized_std.min()),
        "normalized_std_mean": float(normalized_std.mean()),
        "normalized_std_max": float(normalized_std.max()),
        "effective_rank": effective_rank(values),
        "k90": k90,
        "k99": k99,
        "top1_energy_ratio": float(probability[:1].sum()),
        "top5_energy_ratio": float(probability[:5].sum()),
        "top10_energy_ratio": float(probability[:10].sum()),
        "variance_floor_loss": variance_floor,
        "covariance_loss": covariance_loss,
        "current_regularization": variance_floor + covariance_weight * covariance_loss,
        "correlation_offdiag_loss": float(np.square(correlation_offdiag).mean()),
    }
    return stats, energy


def calibration_rows(
    split: str,
    group: str,
    name: str,
    prediction: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    bins: int,
) -> Tuple[List[Dict[str, Any]], float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = (
        np.isfinite(prediction)
        & np.isfinite(target)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    if valid.sum() < 2:
        return [], 0.0
    p = prediction[valid]
    t = target[valid]
    w = weights[valid]
    edges = np.quantile(t, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return [], 0.0
    assignments = np.digitize(t, edges[1:-1], right=True)
    rows: List[Dict[str, Any]] = []
    means = []
    counts = []
    for bin_index in range(len(edges) - 1):
        mask = assignments == bin_index
        if not mask.any():
            continue
        bin_weight = w[mask]
        rows.append({
            "split": split,
            "group": group,
            "name": name,
            "bin": bin_index,
            "left": float(edges[bin_index]),
            "right": float(edges[bin_index + 1]),
            "count": int(mask.sum()),
            "weight_sum": float(bin_weight.sum()),
            "target_mean": weighted_mean(t[mask], bin_weight),
            "prediction_mean": weighted_mean(p[mask], bin_weight),
            "prediction_std": weighted_std(p[mask], bin_weight),
            "mae": weighted_mean(np.abs(p[mask] - t[mask]), bin_weight),
        })
        means.append(rows[-1]["prediction_mean"])
        counts.append(rows[-1]["weight_sum"])
    if len(means) < 2:
        monotonicity = 0.0
    else:
        monotonicity = weighted_spearman(
            np.asarray(means),
            np.arange(len(means), dtype=np.float64),
            np.asarray(counts),
        )
    return rows, monotonicity


def compare_nested(expected: Any, actual: Any, path: str = "") -> List[str]:
    errors: List[str] = []
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        for key in expected:
            child = f"{path}.{key}" if path else str(key)
            if key not in actual:
                errors.append(f"缺少 {child}")
            else:
                errors.extend(compare_nested(expected[key], actual[key], child))
        return errors
    if isinstance(expected, (float, np.floating)) or isinstance(actual, (float, np.floating)):
        try:
            if not math.isclose(float(expected), float(actual), rel_tol=1e-6, abs_tol=1e-8):
                errors.append(f"{path}: {actual!r} != {expected!r}")
            return errors
        except (TypeError, ValueError):
            pass
    if expected != actual:
        errors.append(f"{path}: {actual!r} != {expected!r}")
    return errors


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = st2.load_torch_checkpoint(checkpoint_path, map_location=device)
    model_kwargs = checkpoint.get("model_kwargs")
    if not model_kwargs:
        saved_args = checkpoint.get("args", {}) or {}
        model_kwargs = {
            "input_channels": 5,
            "nhid": int(saved_args.get("nhid", 64)),
            "levels": int(saved_args.get("levels", st2.DEFAULT_TCN_LEVELS)),
            "ksize": int(saved_args.get("ksize", 7)),
            "dropout": float(saved_args.get("dropout", 0.2)),
            "lstm_hidden": int(saved_args.get("lstm_hidden", 64)),
            "lstm_layers": int(saved_args.get("lstm_layers", 1)),
            "embedding_dim": int(saved_args.get("embedding_dim", 64)),
            "proj_dim": int(saved_args.get("proj_dim", 128)),
        }
    model = st2.EncoderWithHead(**model_kwargs).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    model.eval()
    return model, checkpoint, model_kwargs


def load_stage1_anchor_model(
    checkpoint: Mapping[str, Any],
    model_kwargs: Mapping[str, Any],
    explicit_path: Optional[str],
    device: torch.device,
):
    candidate = explicit_path
    if not candidate:
        candidate = (checkpoint.get("args", {}) or {}).get("phase1_ckpt")
    if not candidate:
        return None, "checkpoint 未记录 phase1_ckpt"
    path = resolve_path(candidate)
    if not path.is_file():
        return None, f"Stage-1 checkpoint 不存在：{path}"
    model = st2.EncoderWithHead(**dict(model_kwargs)).to(device)
    try:
        st2.load_stage1_model(model, path, device)
    except Exception as exc:
        return None, f"Stage-1 加载失败：{exc}"
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, ""


def validate_integrity(
    checkpoint: Mapping[str, Any],
    model: torch.nn.Module,
    data: Mapping[str, Any],
    targets: Mapping[str, Any],
    transfer: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    rows: List[Dict[str, Any]] = []
    failures: List[str] = []
    warnings_list: List[str] = []

    def record(name: str, passed: bool, actual: Any = "", expected: Any = "", severity: str = "error"):
        rows.append({
            "check": name,
            "passed": bool(passed),
            "actual": actual,
            "expected": expected,
            "severity": severity,
        })
        if not passed:
            message = f"{name}: actual={actual!r}, expected={expected!r}"
            if severity == "error":
                failures.append(message)
            else:
                warnings_list.append(message)

    record(
        "target_pipeline_version",
        checkpoint.get("target_pipeline_version") == TARGET_PIPELINE_VERSION,
        checkpoint.get("target_pipeline_version"),
        TARGET_PIPELINE_VERSION,
    )
    record(
        "stage2_objective_version",
        checkpoint.get("stage2_objective_version") == st2.STAGE2_OBJECTIVE_VERSION,
        checkpoint.get("stage2_objective_version"),
        st2.STAGE2_OBJECTIVE_VERSION,
    )
    expected_arch = getattr(st2, "TCN_LSTM_ARCH_VERSION", None)
    if expected_arch is not None:
        record(
            "model_arch_version",
            checkpoint.get("model_arch_version") == expected_arch,
            checkpoint.get("model_arch_version"),
            expected_arch,
        )
    record(
        "physical_head_dimensions",
        model.physical_head.out_features == len(PHYSICAL_TARGET_NAMES),
        model.physical_head.out_features,
        len(PHYSICAL_TARGET_NAMES),
    )
    record(
        "response_head_dimensions",
        model.response_head.out_features == len(RESPONSE_TARGET_NAMES),
        model.response_head.out_features,
        len(RESPONSE_TARGET_NAMES),
    )
    record(
        "behavior_head_dimensions",
        model.behavior_head.out_features == len(BEHAVIOR_TARGET_NAMES),
        model.behavior_head.out_features,
        len(BEHAVIOR_TARGET_NAMES),
    )
    checkpoint_args = checkpoint.get("args", {}) or {}
    semantics = {
        "physical_head_semantics": "policy-invariant storage evidence",
        "behavior_head_semantics": "role-relative storage behavior under current policy",
        "physical_policy_masked": True,
        "physical_season_masked": True,
        "behavior_season_masked": True,
        "behavior_policy_role_retained": True,
        "policy_counterfactual_training": True,
        "hierarchy_consistency_training": float(
            checkpoint_args.get("lambda_hierarchy", 0.0)
        ) > 0.0,
        "p_attenuation_training": float(
            checkpoint_args.get("lambda_p_attenuation", 0.0)
        ) > 0.0,
        "p_attenuation_role_invariance": float(
            checkpoint_args.get("lambda_role_invariance", 0.0)
        ) > 0.0,
    }
    for key, expected in semantics.items():
        record(key, checkpoint.get(key) == expected, checkpoint.get(key), expected)

    data_info = checkpoint.get("data_info", {}) or {}
    for key, default in (
        ("lambda_hierarchy", 0.03),
        ("lambda_p_attenuation", 0.05),
        ("p_attenuation_margin", 0.015),
        ("lambda_role_invariance", 0.01),
    ):
        actual = checkpoint_args.get(key)
        record(
            f"args.{key}",
            actual is not None and float(actual) >= 0.0,
            actual,
            f">= 0 (default={default})",
        )
    record(
        "data_info.p_attenuation_cache_version",
        data_info.get("p_attenuation_cache_version")
        == st2.P_ATTENUATION_CACHE_VERSION,
        data_info.get("p_attenuation_cache_version"),
        st2.P_ATTENUATION_CACHE_VERSION,
    )
    for key in ("pipeline_version", "data_sha256", "split_manifest_sha256", "split_strategy"):
        actual = data.get(key, "")
        expected = data_info.get(key, "")
        record(f"data_info.{key}", actual == expected, actual, expected)
    normalization_errors = compare_nested(data_info.get("normalization", {}), data.get("normalization", {}))
    record(
        "normalization_match",
        not normalization_errors,
        "; ".join(normalization_errors[:5]),
        "checkpoint normalization",
    )

    meta = data["meta"]
    user_sets = {
        split: set(meta.iloc[data["split_indices"][split]]["user_id"].astype(str))
        for split in ("train", "val", "test")
    }
    overlap = {
        "train_val": sorted(user_sets["train"] & user_sets["val"]),
        "train_test": sorted(user_sets["train"] & user_sets["test"]),
        "val_test": sorted(user_sets["val"] & user_sets["test"]),
    }
    overlap_count = sum(len(values) for values in overlap.values())
    record("user_split_disjoint", overlap_count == 0, overlap_count, 0)
    record("all_splits_nonempty", all(len(data["split_indices"][s]) > 0 for s in ("train", "val", "test")),
           {s: len(data["split_indices"][s]) for s in ("train", "val", "test")}, "all > 0")

    finite_sources = {
        "X": np.asarray(data["X"]),
        "P_raw": data["P_raw"],
        "Q_raw": data["Q_raw"],
        "physical_targets": targets["physical_targets"],
        "response_targets": targets["response_targets"],
        "behavior_targets": targets["behavior_targets"],
        "transfer_positive_inputs": transfer["positive_inputs"],
        "transfer_weight": transfer["transfer_weight"],
    }
    for name, values in finite_sources.items():
        values = np.asarray(values)
        finite = bool(np.isfinite(values).all())
        record(f"finite.{name}", finite, int(np.size(values) - np.isfinite(values).sum()), 0)

    target_metadata = targets.get("metadata", {}) or {}
    record(
        "target_cache_data_sha",
        target_metadata.get("data_sha256") == data.get("data_sha256"),
        target_metadata.get("data_sha256"),
        data.get("data_sha256"),
    )
    transfer_metadata = transfer.get("metadata", {}) or {}
    record(
        "policy_transfer_version",
        transfer_metadata.get("policy_transfer_version") == POLICY_TRANSFER_VERSION,
        transfer_metadata.get("policy_transfer_version"),
        POLICY_TRANSFER_VERSION,
    )
    record(
        "transfer_cache_data_sha",
        transfer_metadata.get("data_sha256") == data.get("data_sha256"),
        transfer_metadata.get("data_sha256"),
        data.get("data_sha256"),
    )
    acceptance = float(np.asarray(transfer["accepted"]).mean())
    record("policy_transfer_acceptance", acceptance >= 0.30, acceptance, ">=0.30", severity="warning")
    return rows, failures, warnings_list


def select_indices(
    indices: Sequence[int],
    meta,
    max_samples: int,
    max_per_user: int,
    seed: int,
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    rng = np.random.default_rng(seed)
    if max_per_user > 0:
        selected_parts = []
        selected_meta = meta.iloc[indices]
        for _, group in selected_meta.groupby("user_id", sort=False):
            group_indices = group.index.to_numpy(dtype=np.int64)
            if len(group_indices) > max_per_user:
                group_indices = rng.choice(group_indices, size=max_per_user, replace=False)
            selected_parts.append(np.asarray(group_indices, dtype=np.int64))
        indices = np.concatenate(selected_parts) if selected_parts else np.zeros(0, dtype=np.int64)
    if max_samples > 0 and len(indices) > max_samples:
        indices = rng.choice(indices, size=max_samples, replace=False)
    return np.sort(indices.astype(np.int64))


@torch.inference_mode()
def collect_predictions(
    model,
    stage1_model,
    clean_inputs: torch.Tensor,
    transfer_inputs: torch.Tensor,
    mismatch_inputs: torch.Tensor,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    pseudo_shift_points: int,
) -> Dict[str, np.ndarray]:
    index_tensor = torch.as_tensor(indices, dtype=torch.long)
    clean_selected = clean_inputs.index_select(0, index_tensor)
    transfer_selected = transfer_inputs.index_select(0, index_tensor)
    mismatch_selected = mismatch_inputs.index_select(0, index_tensor)
    outputs: Dict[str, List[np.ndarray]] = defaultdict(list)

    model.eval()
    if stage1_model is not None:
        stage1_model.eval()

    def head_predict(head, embedding):
        return torch.sigmoid(head(F.normalize(embedding, dim=1)))

    for start in range(0, len(indices), batch_size):
        clean = clean_selected[start:start + batch_size].to(device)
        transfer = transfer_selected[start:start + batch_size].to(device)
        mismatch = mismatch_selected[start:start + batch_size].to(device)
        pseudo_shift = st2.make_pseudo_a_variant(
            clean, shift_points=pseudo_shift_points, sign_flip=False
        )
        pseudo_shift_sign = st2.make_pseudo_a_variant(
            clean, shift_points=pseudo_shift_points, sign_flip=True
        )

        physical_h = model.encode(st2.mask_physical_context(clean))
        context_h = model.encode(st2.mask_behavior_season(clean))
        transfer_physical_h = model.encode(st2.mask_physical_context(transfer))
        mismatch_physical_h = model.encode(st2.mask_physical_context(mismatch))
        transfer_context_h = model.encode(st2.mask_behavior_season(transfer))
        mismatch_context_h = model.encode(st2.mask_behavior_season(mismatch))
        pseudo_shift_h = model.encode(st2.mask_physical_context(pseudo_shift))
        pseudo_shift_sign_h = model.encode(st2.mask_physical_context(pseudo_shift_sign))

        tensors = {
            "physical": head_predict(model.physical_head, physical_h),
            "response": head_predict(model.response_head, physical_h),
            "behavior": head_predict(model.behavior_head, context_h),
            "transfer_physical": head_predict(model.physical_head, transfer_physical_h),
            "transfer_response": head_predict(model.response_head, transfer_physical_h),
            "transfer_behavior": head_predict(model.behavior_head, transfer_context_h),
            "mismatch_physical": head_predict(model.physical_head, mismatch_physical_h),
            "mismatch_response": head_predict(model.response_head, mismatch_physical_h),
            "mismatch_behavior": head_predict(model.behavior_head, mismatch_context_h),
            "pseudo_shift_physical": head_predict(model.physical_head, pseudo_shift_h),
            "pseudo_shift_response": head_predict(model.response_head, pseudo_shift_h),
            "pseudo_shift_sign_physical": head_predict(model.physical_head, pseudo_shift_sign_h),
            "pseudo_shift_sign_response": head_predict(model.response_head, pseudo_shift_sign_h),
            "embedding_physical": physical_h,
            "embedding_context": context_h,
            "embedding_transfer_context": transfer_context_h,
            "embedding_mismatch_context": mismatch_context_h,
        }
        if stage1_model is not None:
            tensors["embedding_stage1"] = stage1_model.encode(clean)
        for name, tensor in tensors.items():
            outputs[name].append(tensor.detach().float().cpu().numpy())

    return {name: np.concatenate(parts, axis=0) for name, parts in outputs.items()}


def build_target_diagnostics(
    split: str,
    predictions: Mapping[str, np.ndarray],
    targets: Mapping[str, Any],
    indices: np.ndarray,
    sample_weight: np.ndarray,
    calibration_bins_count: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    metric_rows: List[Dict[str, Any]] = []
    distribution_rows: List[Dict[str, Any]] = []
    all_calibration_rows: List[Dict[str, Any]] = []
    selected_sample_weight = sample_weight[indices]
    for group, names, target_key, weight_key in GROUP_SPECS:
        prediction = predictions[group]
        target_matrix = targets[target_key][indices]
        weight_matrix = targets[weight_key][indices]
        for target_index, name in enumerate(names):
            target = target_matrix[:, target_index]
            target_weight = weight_matrix[:, target_index]
            effective_weight = target_weight * selected_sample_weight
            metrics = weighted_error_metrics(
                prediction[:, target_index], target, effective_weight
            )
            unweighted = np.ones_like(effective_weight)
            unweighted_metrics = weighted_error_metrics(
                prediction[:, target_index], target, unweighted
            )
            calibration, monotonicity = calibration_rows(
                split,
                group,
                name,
                prediction[:, target_index],
                target,
                effective_weight,
                calibration_bins_count,
            )
            all_calibration_rows.extend(calibration)
            metric_rows.append({
                "split": split,
                "group": group,
                "name": name,
                **metrics,
                "unweighted_pearson": unweighted_metrics["pearson"],
                "unweighted_spearman": unweighted_metrics["spearman"],
                "unweighted_mae": unweighted_metrics["mae"],
                "calibration_monotonicity": monotonicity,
                "target_weight_mean": float(target_weight.mean()),
                "active_ratio": float((target_weight > 1e-6).mean()),
            })
            active = target_weight > 1e-6
            observed = target[active] if active.any() else target
            observed_weights = effective_weight[active] if active.any() else selected_sample_weight
            quantiles = weighted_quantile(
                observed,
                (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99),
                observed_weights,
            )
            distribution_rows.append({
                "split": split,
                "group": group,
                "name": name,
                "count": int(len(target)),
                "active_count": int(active.sum()),
                "active_ratio": float(active.mean()),
                "mean": weighted_mean(observed, observed_weights),
                "std": weighted_std(observed, observed_weights),
                "q01": float(quantiles[0]),
                "q05": float(quantiles[1]),
                "q25": float(quantiles[2]),
                "q50": float(quantiles[3]),
                "q75": float(quantiles[4]),
                "q95": float(quantiles[5]),
                "q99": float(quantiles[6]),
                "zero_ratio": float((observed <= 1e-6).mean()),
                "one_ratio": float((observed >= 1.0 - 1e-6).mean()),
                "weight_mean": float(target_weight.mean()),
            })
    return metric_rows, distribution_rows, all_calibration_rows


def make_per_sample_rows(
    split: str,
    indices: np.ndarray,
    meta,
    predictions: Mapping[str, np.ndarray],
    targets: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    selected_meta = meta.iloc[indices].reset_index(drop=True)
    for local_index, global_index in enumerate(indices):
        meta_row = selected_meta.iloc[local_index]
        row: Dict[str, Any] = {
            "split": split,
            "global_index": int(global_index),
            "sample_id": int(meta_row.get("sample_id", global_index)),
            "source_row": int(meta_row.get("source_row", -1)),
            "user_id": str(meta_row.get("user_id", "")),
            "date": str(meta_row.get("date", "")),
            "month": int(meta_row.get("month", 0)),
            "industry": str(meta_row.get("industry", "")),
            "sample_weight": float(meta_row.get("sample_weight", 1.0)),
        }
        for group, names, target_key, weight_key in GROUP_SPECS:
            for target_index, name in enumerate(names):
                row[f"{group}_{name}_pred"] = float(predictions[group][local_index, target_index])
                row[f"{group}_{name}_target"] = float(targets[target_key][global_index, target_index])
                row[f"{group}_{name}_weight"] = float(targets[weight_key][global_index, target_index])
        rows.append(row)
    return rows


def pseudo_score(physical: np.ndarray, response: np.ndarray) -> np.ndarray:
    storage_index = PHYSICAL_TARGET_NAMES.index("storage_evidence")
    confidence_index = RESPONSE_TARGET_NAMES.index("response_confidence")
    return 0.70 * physical[:, storage_index] + 0.30 * response[:, confidence_index]


def pseudo_a_diagnostics(
    split: str,
    indices: np.ndarray,
    meta,
    predictions: Mapping[str, np.ndarray],
    targets: Mapping[str, Any],
    sample_weight: np.ndarray,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    real = pseudo_score(predictions["physical"], predictions["response"])
    variant_scores = {
        "shift": pseudo_score(
            predictions["pseudo_shift_physical"],
            predictions["pseudo_shift_response"],
        ),
        "shift_sign": pseudo_score(
            predictions["pseudo_shift_sign_physical"],
            predictions["pseudo_shift_sign_response"],
        ),
    }
    pair_rows: List[Dict[str, Any]] = []
    selected_meta = meta.iloc[indices].reset_index(drop=True)
    activity_index = PHYSICAL_TARGET_NAMES.index("activity_strength")
    activity = targets["physical_targets"][indices, activity_index]
    weights = sample_weight[indices]
    summary_rows: List[Dict[str, Any]] = []
    combined_drops = []
    combined_weights = []
    for variant, pseudo in variant_scores.items():
        drop = real - pseudo
        combined_drops.append(drop)
        combined_weights.append(weights)
        q05, q50, q95 = weighted_quantile(drop, (0.05, 0.50, 0.95), weights)
        summary_rows.append({
            "split": split,
            "variant": variant,
            "count": int(len(drop)),
            "score_drop_mean": weighted_mean(drop, weights),
            "score_drop_q05": float(q05),
            "score_drop_q50": float(q50),
            "score_drop_q95": float(q95),
            "pair_accuracy": weighted_mean((drop > 0.0).astype(np.float64), weights),
            "negative_drop_ratio": weighted_mean((drop < 0.0).astype(np.float64), weights),
        })
        for local_index, global_index in enumerate(indices):
            meta_row = selected_meta.iloc[local_index]
            pair_rows.append({
                "split": split,
                "variant": variant,
                "global_index": int(global_index),
                "user_id": str(meta_row.get("user_id", "")),
                "date": str(meta_row.get("date", "")),
                "month": int(meta_row.get("month", 0)),
                "sample_weight": float(weights[local_index]),
                "activity_strength": float(activity[local_index]),
                "real_score": float(real[local_index]),
                "pseudo_score": float(pseudo[local_index]),
                "score_drop": float(drop[local_index]),
                "pair_correct": int(drop[local_index] > 0.0),
            })
    all_drop = np.concatenate(combined_drops)
    all_weight = np.concatenate(combined_weights)
    q05, q50, q95 = weighted_quantile(all_drop, (0.05, 0.50, 0.95), all_weight)
    summary_rows.append({
        "split": split,
        "variant": "all",
        "count": int(len(all_drop)),
        "score_drop_mean": weighted_mean(all_drop, all_weight),
        "score_drop_q05": float(q05),
        "score_drop_q50": float(q50),
        "score_drop_q95": float(q95),
        "pair_accuracy": weighted_mean((all_drop > 0.0).astype(np.float64), all_weight),
        "negative_drop_ratio": weighted_mean((all_drop < 0.0).astype(np.float64), all_weight),
    })
    return pair_rows, summary_rows


def policy_shift_group(shift: int) -> str:
    shift = abs(int(shift))
    if shift == 0:
        return "0"
    if shift <= 24:
        return "8-24"
    if shift <= 48:
        return "28-48"
    return "52-88"


def policy_diagnostics(
    split: str,
    indices: np.ndarray,
    meta,
    predictions: Mapping[str, np.ndarray],
    transfer_targets: Mapping[str, Any],
    mismatch_targets: Mapping[str, Any],
    transfer: Mapping[str, Any],
    sample_weight: np.ndarray,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, float]]:
    selected_meta = meta.iloc[indices].reset_index(drop=True)
    transfer_weight = np.asarray(transfer["transfer_weight"])[indices]
    valid = transfer_weight > 0.0
    pair_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    strata_rows: List[Dict[str, Any]] = []
    invariance = {
        "physical_max_error": 0.0,
        "physical_mean_error": 0.0,
        "response_max_error": 0.0,
        "response_mean_error": 0.0,
    }
    if not valid.any():
        return pair_rows, summary_rows, strata_rows, invariance

    valid_local = np.flatnonzero(valid)
    valid_global = indices[valid]
    base_weight = sample_weight[valid_global] * transfer_weight[valid]
    transfer_behavior_target = transfer_targets["behavior_targets"][valid_global]
    mismatch_behavior_target = mismatch_targets["behavior_targets"][valid_global]
    transfer_behavior_weight = transfer_targets["behavior_weights"][valid_global]
    mismatch_behavior_weight = mismatch_targets["behavior_weights"][valid_global]
    transfer_behavior_prediction = predictions["transfer_behavior"][valid]
    mismatch_behavior_prediction = predictions["mismatch_behavior"][valid]

    for target_index, name in enumerate(BEHAVIOR_TARGET_NAMES):
        dimension_weight = base_weight * np.maximum(
            transfer_behavior_weight[:, target_index],
            mismatch_behavior_weight[:, target_index],
        )
        predicted_drop = (
            transfer_behavior_prediction[:, target_index]
            - mismatch_behavior_prediction[:, target_index]
        )
        target_drop = (
            transfer_behavior_target[:, target_index]
            - mismatch_behavior_target[:, target_index]
        )
        transfer_fit = weighted_error_metrics(
            transfer_behavior_prediction[:, target_index],
            transfer_behavior_target[:, target_index],
            base_weight * transfer_behavior_weight[:, target_index],
        )
        mismatch_fit = weighted_error_metrics(
            mismatch_behavior_prediction[:, target_index],
            mismatch_behavior_target[:, target_index],
            base_weight * mismatch_behavior_weight[:, target_index],
        )
        summary_rows.append({
            "split": split,
            "name": name,
            "count": int(valid.sum()),
            "predicted_drop_mean": weighted_mean(predicted_drop, dimension_weight),
            "target_drop_mean": weighted_mean(target_drop, dimension_weight),
            "drop_calibration_mae": abs(
                weighted_mean(predicted_drop, dimension_weight)
                - weighted_mean(target_drop, dimension_weight)
            ),
            "pair_accuracy": weighted_mean((predicted_drop > 0.0).astype(np.float64), dimension_weight),
            "drop_pearson": weighted_corr(predicted_drop, target_drop, dimension_weight),
            "drop_spearman": weighted_spearman(predicted_drop, target_drop, dimension_weight),
            "transfer_pearson": transfer_fit["pearson"],
            "transfer_mae": transfer_fit["mae"],
            "mismatch_pearson": mismatch_fit["pearson"],
            "mismatch_mae": mismatch_fit["mae"],
        })

    storage_index = BEHAVIOR_TARGET_NAMES.index("storage_behavior")
    storage_pred_drop = (
        transfer_behavior_prediction[:, storage_index]
        - mismatch_behavior_prediction[:, storage_index]
    )
    storage_target_drop = (
        transfer_behavior_target[:, storage_index]
        - mismatch_behavior_target[:, storage_index]
    )
    for row_index, local_index in enumerate(valid_local):
        global_index = int(indices[local_index])
        meta_row = selected_meta.iloc[local_index]
        row: Dict[str, Any] = {
            "split": split,
            "global_index": global_index,
            "user_id": str(meta_row.get("user_id", "")),
            "date": str(meta_row.get("date", "")),
            "month": int(meta_row.get("month", 0)),
            "sample_weight": float(sample_weight[global_index]),
            "transfer_weight": float(transfer_weight[local_index]),
            "accepted": int(np.asarray(transfer["accepted"])[global_index]),
            "policy_shift": int(np.asarray(transfer["policy_shift"])[global_index]),
            "shift_group": policy_shift_group(np.asarray(transfer["policy_shift"])[global_index]),
            "activity_coverage": float(np.asarray(transfer["activity_coverage"])[global_index]),
            "energy_retention": float(np.asarray(transfer["energy_retention"])[global_index]),
            "target_pattern_quality": float(np.asarray(transfer["target_pattern"])[global_index]),
            "storage_behavior_predicted_drop": float(storage_pred_drop[row_index]),
            "storage_behavior_target_drop": float(storage_target_drop[row_index]),
            "storage_behavior_pair_correct": int(storage_pred_drop[row_index] > 0.0),
        }
        for target_index, name in enumerate(BEHAVIOR_TARGET_NAMES):
            row[f"{name}_transfer_pred"] = float(transfer_behavior_prediction[row_index, target_index])
            row[f"{name}_mismatch_pred"] = float(mismatch_behavior_prediction[row_index, target_index])
            row[f"{name}_transfer_target"] = float(transfer_behavior_target[row_index, target_index])
            row[f"{name}_mismatch_target"] = float(mismatch_behavior_target[row_index, target_index])
        pair_rows.append(row)

    if pair_rows:
        quality_values = np.asarray([row["transfer_weight"] for row in pair_rows])
        q1, q2 = np.quantile(quality_values, [1 / 3, 2 / 3])
        for row in pair_rows:
            quality = row["transfer_weight"]
            row["quality_group"] = "low" if quality <= q1 else ("mid" if quality <= q2 else "high")

        def summarize_stratum(stratum_type: str, stratum_value: Any, chosen: List[Dict[str, Any]]):
            if not chosen:
                return
            weights = np.asarray([
                float(row["sample_weight"]) * float(row["transfer_weight"])
                for row in chosen
            ])
            pred_drop = np.asarray([row["storage_behavior_predicted_drop"] for row in chosen])
            target_drop = np.asarray([row["storage_behavior_target_drop"] for row in chosen])
            strata_rows.append({
                "split": split,
                "stratum_type": stratum_type,
                "stratum": str(stratum_value),
                "count": int(len(chosen)),
                "predicted_drop_mean": weighted_mean(pred_drop, weights),
                "target_drop_mean": weighted_mean(target_drop, weights),
                "pair_accuracy": weighted_mean((pred_drop > 0.0).astype(float), weights),
                "drop_pearson": weighted_corr(pred_drop, target_drop, weights),
            })

        for stratum_type in ("shift_group", "quality_group", "month"):
            values = sorted({row[stratum_type] for row in pair_rows}, key=str)
            for value in values:
                summarize_stratum(
                    stratum_type,
                    value,
                    [row for row in pair_rows if row[stratum_type] == value],
                )

    physical_error = np.abs(
        predictions["transfer_physical"][valid]
        - predictions["mismatch_physical"][valid]
    )
    response_error = np.abs(
        predictions["transfer_response"][valid]
        - predictions["mismatch_response"][valid]
    )
    invariance = {
        "physical_max_error": float(physical_error.max()),
        "physical_mean_error": float(physical_error.mean()),
        "response_max_error": float(response_error.max()),
        "response_mean_error": float(response_error.mean()),
    }
    return pair_rows, summary_rows, strata_rows, invariance


def representation_diagnostics(
    split: str,
    predictions: Mapping[str, np.ndarray],
    target_std: float,
    covariance_weight: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    views = {
        "physical": predictions.get("embedding_physical"),
        "context": predictions.get("embedding_context"),
        "transfer_context": predictions.get("embedding_transfer_context"),
        "mismatch_context": predictions.get("embedding_mismatch_context"),
        "stage1_anchor": predictions.get("embedding_stage1"),
    }
    output: Dict[str, Any] = {}
    spectrum_rows: List[Dict[str, Any]] = []
    for name, values in views.items():
        if values is None:
            continue
        stats, energy = representation_statistics(values, target_std, covariance_weight)
        output[name] = stats
        probability = energy / max(float(energy.sum()), EPS)
        for index, (eigenvalue, ratio) in enumerate(zip(energy, probability)):
            spectrum_rows.append({
                "split": split,
                "view": name,
                "index": index + 1,
                "eigenvalue": float(eigenvalue),
                "energy_ratio": float(ratio),
            })
    if "physical" in output and "stage1_anchor" in output:
        output["rank_retention"] = (
            output["physical"]["effective_rank"]
            / max(output["stage1_anchor"]["effective_rank"], EPS)
        )
    return output, spectrum_rows


def core_head_diagnostics(
    split: str,
    predictions: Mapping[str, np.ndarray],
    targets: Mapping[str, Any],
    indices: np.ndarray,
) -> List[Dict[str, Any]]:
    p_idx = PHYSICAL_TARGET_NAMES.index("storage_evidence")
    r_idx = RESPONSE_TARGET_NAMES.index("response_confidence")
    b_idx = BEHAVIOR_TARGET_NAMES.index("storage_behavior")
    pred = np.column_stack([
        predictions["physical"][:, p_idx],
        predictions["response"][:, r_idx],
        predictions["behavior"][:, b_idx],
    ])
    target = np.column_stack([
        targets["physical_targets"][indices, p_idx],
        targets["response_targets"][indices, r_idx],
        targets["behavior_targets"][indices, b_idx],
    ])
    names = ("physical", "response", "behavior")
    rows: List[Dict[str, Any]] = []
    for left, right in ((0, 1), (0, 2), (1, 2)):
        pred_corr = weighted_corr(pred[:, left], pred[:, right], np.ones(len(pred)))
        target_corr = weighted_corr(target[:, left], target[:, right], np.ones(len(target)))
        rows.append({
            "split": split,
            "metric": "pair_correlation",
            "pair": f"{names[left]}_{names[right]}",
            "prediction": pred_corr,
            "target": target_corr,
            "excess": abs(pred_corr) - abs(target_corr),
        })
    rows.extend([
        {
            "split": split,
            "metric": "effective_rank",
            "pair": "all_three",
            "prediction": effective_rank(pred),
            "target": effective_rank(target),
            "excess": "",
        },
        {
            "split": split,
            "metric": "minimum_output_std",
            "pair": "all_three",
            "prediction": float(pred.std(axis=0).min()),
            "target": float(target.std(axis=0).min()),
            "excess": "",
        },
    ])
    return rows


def build_user_metrics(
    split: str,
    indices: np.ndarray,
    meta,
    predictions: Mapping[str, np.ndarray],
    targets: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    selected_meta = meta.iloc[indices].reset_index(drop=True)
    groups = selected_meta.groupby("user_id", sort=False).indices
    for user_id, local_indices in groups.items():
        local = np.asarray(local_indices, dtype=np.int64)
        global_indices = indices[local]
        sample_weights = selected_meta.iloc[local]["sample_weight"].to_numpy(np.float64)
        for group, names, target_key, weight_key in GROUP_SPECS:
            for target_index, name in enumerate(names):
                prediction = predictions[group][local, target_index]
                target = targets[target_key][global_indices, target_index]
                weights = targets[weight_key][global_indices, target_index] * sample_weights
                metrics = weighted_error_metrics(prediction, target, weights)
                rows.append({
                    "split": split,
                    "user_id": str(user_id),
                    "sample_count": int(len(local)),
                    "group": group,
                    "name": name,
                    "prediction_mean": weighted_mean(prediction, weights),
                    "target_mean": weighted_mean(target, weights),
                    "mae": metrics["mae"],
                    "pearson_within_user": metrics["pearson"] if len(local) >= 3 else "",
                    "spearman_within_user": metrics["spearman"] if len(local) >= 3 else "",
                })

    summary_rows: List[Dict[str, Any]] = []
    for group, name in CORE_SPECS:
        relevant = [row for row in rows if row["group"] == group and row["name"] == name]
        if len(relevant) < 2:
            continue
        pred = np.asarray([row["prediction_mean"] for row in relevant], dtype=np.float64)
        target = np.asarray([row["target_mean"] for row in relevant], dtype=np.float64)
        user_weights = np.ones(len(relevant), dtype=np.float64)
        summary_rows.append({
            "split": split,
            "group": group,
            "name": name,
            "user_count": int(len(relevant)),
            "user_mean_pearson": weighted_corr(pred, target, user_weights),
            "user_mean_spearman": weighted_spearman(pred, target, user_weights),
            "user_mean_mae": float(np.mean(np.abs(pred - target))),
            "worst_user_mae": float(max(float(row["mae"]) for row in relevant)),
        })
    return rows, summary_rows


def scale_power(power: np.ndarray, quantile: float = 95.0, clip_max: float = 2.0) -> np.ndarray:
    reference = np.percentile(power, quantile, axis=1)
    maximum = np.max(power, axis=1)
    reference = np.where(reference > 1e-6, reference, maximum)
    reference = np.where(reference > 1e-6, reference, 1.0)
    return np.clip(power / reference[:, None], 0.0, clip_max).astype(np.float32)


def masked_statistics(values: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    count = np.maximum(mask.sum(axis=1), 1)
    mean = (values * mask).sum(axis=1) / count
    centered = (values - mean[:, None]) * mask
    std = np.sqrt((centered * centered).sum(axis=1) / count)
    return mean, std


def build_policy_relative_features(power: np.ndarray, role: np.ndarray, role_jump: np.ndarray) -> np.ndarray:
    difference = np.diff(power, axis=1, prepend=power[:, :1])
    absolute_difference = np.abs(difference)
    features = [
        power.mean(axis=1),
        power.std(axis=1),
        *[np.quantile(power, q, axis=1) for q in (0.05, 0.25, 0.50, 0.75, 0.95)],
        absolute_difference.mean(axis=1),
        np.quantile(absolute_difference, 0.95, axis=1),
        np.clip(difference, 0.0, None).mean(axis=1),
        np.clip(-difference, 0.0, None).mean(axis=1),
    ]
    for mask in (role < -0.25, np.abs(role) <= 0.25, role > 0.25):
        p_mean, p_std = masked_statistics(power, mask)
        positive_mean, _ = masked_statistics(np.clip(difference, 0.0, None), mask)
        negative_mean, _ = masked_statistics(np.clip(-difference, 0.0, None), mask)
        features.extend((p_mean, p_std, positive_mean, negative_mean))
    role_centered = role - role.mean(axis=1, keepdims=True)
    power_centered = power - power.mean(axis=1, keepdims=True)
    correlation = (role_centered * power_centered).mean(axis=1) / (
        role_centered.std(axis=1) * power_centered.std(axis=1) + 1e-6
    )
    transition = np.abs(role_jump) > 1e-6
    transition_change, transition_change_std = masked_statistics(absolute_difference, transition)
    features.extend((correlation, transition_change, transition_change_std))
    return np.column_stack(features).astype(np.float32)


@torch.inference_mode()
def predict_teacher_outputs(model, inputs: np.ndarray, device: torch.device, batch_size: int):
    physical_outputs = []
    response_outputs = []
    behavior_outputs = []
    model.eval()
    for start in range(0, len(inputs), batch_size):
        batch = torch.from_numpy(inputs[start:start + batch_size]).to(device)
        physical_h = model.encode(st2.mask_physical_context(batch))
        context_h = model.encode(st2.mask_behavior_season(batch))
        physical_outputs.append(torch.sigmoid(
            model.physical_head(F.normalize(physical_h, dim=1))
        ).cpu().numpy())
        response_outputs.append(torch.sigmoid(
            model.response_head(F.normalize(physical_h, dim=1))
        ).cpu().numpy())
        behavior_outputs.append(torch.sigmoid(
            model.behavior_head(F.normalize(context_h, dim=1))
        ).cpu().numpy())
    return (
        np.concatenate(physical_outputs),
        np.concatenate(response_outputs),
        np.concatenate(behavior_outputs),
    )


def fit_regression_probe(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_targets: Mapping[str, np.ndarray],
    test_targets: Mapping[str, np.ndarray],
    train_weight: np.ndarray,
    test_weight: np.ndarray,
    seed: int,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for name in train_targets:
        model = HistGradientBoostingRegressor(
            learning_rate=0.08,
            max_iter=100,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=seed,
        )
        model.fit(train_features, train_targets[name], sample_weight=train_weight)
        prediction = model.predict(test_features)
        expected = test_targets[name]
        result[name] = {
            "corr": weighted_corr(prediction, expected, test_weight),
            "spearman": weighted_spearman(prediction, expected, test_weight),
            "mae": float(mean_absolute_error(expected, prediction, sample_weight=test_weight)),
            "r2": float(r2_score(expected, prediction, sample_weight=test_weight)),
            "prediction_std": float(np.std(prediction)),
            "target_std": float(np.std(expected)),
        }
    return result


def run_p_only_probe(
    model,
    data: Mapping[str, Any],
    transfer: Mapping[str, Any],
    target_config: TeacherTargetConfig,
    device: torch.device,
    batch_size: int,
    seed: int,
    train_samples: int,
    eval_samples: int,
) -> Dict[str, Any]:
    if not SKLEARN_AVAILABLE:
        return {"status": "SKIPPED", "reason": "scikit-learn 不可用"}
    rng = np.random.default_rng(seed)
    meta = data["meta"]
    accepted = np.asarray(transfer["accepted"]).astype(bool)
    shifts = np.asarray(transfer["policy_shift"])
    train_split = meta["split"].to_numpy() == "train"
    train_candidates = np.flatnonzero(accepted & train_split & (shifts == 0))
    if len(train_candidates) < 100:
        train_candidates = np.flatnonzero(accepted & train_split)
    if len(train_candidates) == 0:
        return {"status": "SKIPPED", "reason": "没有可用训练政策迁移样本"}
    if len(train_candidates) > train_samples:
        train_candidates = np.sort(rng.choice(train_candidates, train_samples, replace=False))

    result: Dict[str, Any] = {
        "status": "OK",
        "purpose": "检查 P 与相对电价角色是否包含可供 Student 蒸馏的信号；不代表现实负类概率校准",
        "train_split": "train users; prefer policy_shift=0",
        "train_samples": int(len(train_candidates)),
        "eval": {},
    }

    train_inputs = np.asarray(transfer["positive_inputs"])[train_candidates]
    train_p_raw = np.asarray(transfer["positive_p_raw"])[train_candidates]
    train_q_raw = np.asarray(transfer["positive_q_raw"])[train_candidates]
    train_role = np.asarray(transfer["target_role_seq"])[train_candidates]
    train_role_jump = train_inputs[:, 4]
    train_features = build_policy_relative_features(train_inputs[:, 0], train_role, train_role_jump)
    train_weight = np.maximum(np.asarray(transfer["transfer_weight"])[train_candidates], 1e-4)
    train_constraint = compute_stage2_targets(train_p_raw, train_q_raw, train_role, config=target_config)
    train_teacher = predict_teacher_outputs(model, train_inputs, device, batch_size)

    constraint_train_targets = {
        "physical.storage_evidence": train_constraint["physical_targets"][:, PHYSICAL_TARGET_NAMES.index("storage_evidence")],
        "response.response_confidence": train_constraint["response_targets"][:, RESPONSE_TARGET_NAMES.index("response_confidence")],
        "behavior.policy_pattern": train_constraint["behavior_targets"][:, BEHAVIOR_TARGET_NAMES.index("policy_pattern")],
        "behavior.storage_behavior": train_constraint["behavior_targets"][:, BEHAVIOR_TARGET_NAMES.index("storage_behavior")],
    }
    teacher_train_targets = {
        "physical.storage_evidence": train_teacher[0][:, PHYSICAL_TARGET_NAMES.index("storage_evidence")],
        "response.response_confidence": train_teacher[1][:, RESPONSE_TARGET_NAMES.index("response_confidence")],
        "behavior.policy_pattern": train_teacher[2][:, BEHAVIOR_TARGET_NAMES.index("policy_pattern")],
        "behavior.storage_behavior": train_teacher[2][:, BEHAVIOR_TARGET_NAMES.index("storage_behavior")],
    }

    for split in ("val", "test"):
        split_mask = meta["split"].to_numpy() == split
        candidates = np.flatnonzero(accepted & split_mask & (shifts != 0))
        if len(candidates) < 100:
            candidates = np.flatnonzero(accepted & split_mask)
        if len(candidates) == 0:
            result["eval"][split] = {"status": "SKIPPED", "reason": "无可用样本"}
            continue
        if len(candidates) > eval_samples:
            candidates = np.sort(rng.choice(candidates, eval_samples, replace=False))
        inputs = np.asarray(transfer["positive_inputs"])[candidates]
        p_raw = np.asarray(transfer["positive_p_raw"])[candidates]
        q_raw = np.asarray(transfer["positive_q_raw"])[candidates]
        role = np.asarray(transfer["target_role_seq"])[candidates]
        role_jump = inputs[:, 4]
        weight = np.maximum(np.asarray(transfer["transfer_weight"])[candidates], 1e-4)
        positive_features = build_policy_relative_features(inputs[:, 0], role, role_jump)
        base_features = build_policy_relative_features(
            scale_power(p_raw - q_raw), role, role_jump
        )

        classifier = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=100,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=seed,
        )
        train_base_features = build_policy_relative_features(
            scale_power(train_p_raw - train_q_raw), train_role, train_role_jump
        )
        classifier.fit(
            np.concatenate((train_features, train_base_features), axis=0),
            np.concatenate((np.ones(len(train_features)), np.zeros(len(train_features)))),
        )
        test_features = np.concatenate((positive_features, base_features), axis=0)
        test_labels = np.concatenate((np.ones(len(candidates)), np.zeros(len(candidates))))
        probabilities = classifier.predict_proba(test_features)[:, 1]
        positive_probability = probabilities[:len(candidates)]
        base_probability = probabilities[len(candidates):]

        constraint = compute_stage2_targets(p_raw, q_raw, role, config=target_config)
        teacher = predict_teacher_outputs(model, inputs, device, batch_size)
        constraint_test_targets = {
            "physical.storage_evidence": constraint["physical_targets"][:, PHYSICAL_TARGET_NAMES.index("storage_evidence")],
            "response.response_confidence": constraint["response_targets"][:, RESPONSE_TARGET_NAMES.index("response_confidence")],
            "behavior.policy_pattern": constraint["behavior_targets"][:, BEHAVIOR_TARGET_NAMES.index("policy_pattern")],
            "behavior.storage_behavior": constraint["behavior_targets"][:, BEHAVIOR_TARGET_NAMES.index("storage_behavior")],
        }
        teacher_test_targets = {
            "physical.storage_evidence": teacher[0][:, PHYSICAL_TARGET_NAMES.index("storage_evidence")],
            "response.response_confidence": teacher[1][:, RESPONSE_TARGET_NAMES.index("response_confidence")],
            "behavior.policy_pattern": teacher[2][:, BEHAVIOR_TARGET_NAMES.index("policy_pattern")],
            "behavior.storage_behavior": teacher[2][:, BEHAVIOR_TARGET_NAMES.index("storage_behavior")],
        }
        result["eval"][split] = {
            "status": "OK",
            "samples": int(len(candidates)),
            "pseudo_b_binary": {
                "auc": float(roc_auc_score(test_labels, probabilities)),
                "paired_accuracy": float(np.mean(positive_probability > base_probability)),
                "positive_probability_mean": float(positive_probability.mean()),
                "pseudo_probability_mean": float(base_probability.mean()),
                "mean_probability_drop": float((positive_probability - base_probability).mean()),
            },
            "constraint_target_regression": fit_regression_probe(
                train_features,
                positive_features,
                constraint_train_targets,
                constraint_test_targets,
                train_weight,
                weight,
                seed,
            ),
            "teacher_output_regression": fit_regression_probe(
                train_features,
                positive_features,
                teacher_train_targets,
                teacher_test_targets,
                train_weight,
                weight,
                seed,
            ),
        }
    return result


def plot_target_diagnostic(
    output_dir: Path,
    split: str,
    group: str,
    name: str,
    prediction: np.ndarray,
    target: np.ndarray,
    calibration: Sequence[Mapping[str, Any]],
    max_points: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    chosen = np.arange(len(prediction))
    if len(chosen) > max_points:
        chosen = rng.choice(chosen, size=max_points, replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(target[chosen], prediction[chosen], s=8, alpha=0.35)
    axes[0].plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    axes[0].set_xlabel("target")
    axes[0].set_ylabel("prediction")
    axes[0].set_title(f"{split} {group}.{name}: prediction vs target")
    selected_calibration = [
        row for row in calibration
        if row["split"] == split and row["group"] == group and row["name"] == name
    ]
    if selected_calibration:
        x = [row["target_mean"] for row in selected_calibration]
        y = [row["prediction_mean"] for row in selected_calibration]
        axes[1].plot(x, y, marker="o")
    axes[1].plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    axes[1].set_xlabel("bin target mean")
    axes[1].set_ylabel("bin prediction mean")
    axes[1].set_title("quantile-bin calibration")
    fig.tight_layout()
    fig.savefig(output_dir / f"target_{split}_{group}_{name}.png", dpi=160)
    plt.close(fig)


def plot_representation_spectrum(output_dir: Path, spectrum_rows: Sequence[Mapping[str, Any]], split: str) -> None:
    selected = [row for row in spectrum_rows if row["split"] == split]
    if not selected:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    for view in sorted({row["view"] for row in selected}):
        rows = [row for row in selected if row["view"] == view]
        ax.plot(
            [row["index"] for row in rows],
            [max(row["eigenvalue"], 1e-15) for row in rows],
            label=view,
        )
    ax.set_yscale("log")
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("eigenvalue (log)")
    ax.set_title(f"Stage-2 representation spectrum ({split})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"representation_spectrum_{split}.png", dpi=160)
    plt.close(fig)


def plot_pseudo(output_dir: Path, pair_rows: Sequence[Mapping[str, Any]], split: str) -> None:
    selected = [row for row in pair_rows if row["split"] == split]
    if not selected:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    for variant in ("shift", "shift_sign"):
        values = [row["score_drop"] for row in selected if row["variant"] == variant]
        if values:
            ax.hist(values, bins=60, density=True, alpha=0.45, label=variant)
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("real score - pseudo-A score")
    ax.set_ylabel("density")
    ax.set_title(f"Pseudo-A score drop ({split})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"pseudo_a_drop_{split}.png", dpi=160)
    plt.close(fig)


def plot_policy(output_dir: Path, pair_rows: Sequence[Mapping[str, Any]], split: str) -> None:
    selected = [row for row in pair_rows if row["split"] == split]
    if not selected:
        return
    target = np.asarray([row["storage_behavior_target_drop"] for row in selected])
    prediction = np.asarray([row["storage_behavior_predicted_drop"] for row in selected])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(target, prediction, s=10, alpha=0.35)
    lower = float(min(target.min(), prediction.min()))
    upper = float(max(target.max(), prediction.max()))
    ax.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1)
    ax.axhline(0.0, linewidth=0.8)
    ax.axvline(0.0, linewidth=0.8)
    ax.set_xlabel("target transfer-mismatch drop")
    ax.set_ylabel("predicted transfer-mismatch drop")
    ax.set_title(f"Policy counterfactual calibration ({split})")
    fig.tight_layout()
    fig.savefig(output_dir / f"policy_drop_{split}.png", dpi=160)
    plt.close(fig)


def plot_core_heads(output_dir: Path, split: str, predictions: Mapping[str, np.ndarray]) -> None:
    p = predictions["physical"][:, PHYSICAL_TARGET_NAMES.index("storage_evidence")]
    r = predictions["response"][:, RESPONSE_TARGET_NAMES.index("response_confidence")]
    b = predictions["behavior"][:, BEHAVIOR_TARGET_NAMES.index("storage_behavior")]
    for left_name, left, right_name, right in (
        ("physical", p, "response", r),
        ("physical", p, "behavior", b),
        ("response", r, "behavior", b),
    ):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(left, right, s=9, alpha=0.3)
        ax.set_xlabel(left_name)
        ax.set_ylabel(right_name)
        ax.set_title(f"Core teacher heads ({split}): {left_name} vs {right_name}")
        fig.tight_layout()
        fig.savefig(output_dir / f"core_{split}_{left_name}_{right_name}.png", dpi=160)
        plt.close(fig)


def plot_user_summary(output_dir: Path, rows: Sequence[Mapping[str, Any]], split: str) -> None:
    selected = [
        row for row in rows
        if row["split"] == split
        and row["group"] == "behavior"
        and row["name"] == "storage_behavior"
    ]
    if not selected:
        return
    target = [row["target_mean"] for row in selected]
    prediction = [row["prediction_mean"] for row in selected]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(target, prediction, s=22, alpha=0.65)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("user mean target")
    ax.set_ylabel("user mean prediction")
    ax.set_title(f"User-level storage_behavior ({split})")
    fig.tight_layout()
    fig.savefig(output_dir / f"user_storage_behavior_{split}.png", dpi=160)
    plt.close(fig)



def load_history(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        for raw in reader:
            row: Dict[str, Any] = {}
            for key, value in raw.items():
                if value is None:
                    row[key] = ""
                    continue
                text = value.strip()
                if text == "":
                    row[key] = ""
                    continue
                try:
                    row[key] = float(text)
                    if key == "epoch":
                        row[key] = int(float(text))
                except ValueError:
                    row[key] = text
            rows.append(row)
    return rows


def history_diagnostics(
    history_rows: Sequence[Mapping[str, Any]],
    checkpoint: Mapping[str, Any],
) -> Dict[str, Any]:
    if not history_rows:
        return {"status": "MISSING"}
    score_rows = [row for row in history_rows if isinstance(row.get("score"), (int, float))]
    if not score_rows:
        return {"status": "INVALID", "reason": "history 中没有 score"}
    best_row = max(score_rows, key=lambda row: float(row["score"]))
    final_row = score_rows[-1]
    best_epoch = int(best_row.get("epoch", 0))
    checkpoint_epoch = int(checkpoint.get("epoch", checkpoint.get("best_epoch", 0)) or 0)
    checkpoint_score = finite_or_nan(checkpoint.get("best_score"))
    best_score = float(best_row["score"])
    train_loss_key = "train_loss"
    val_loss_key = "val_loss"
    return {
        "status": "OK",
        "epochs": int(len(score_rows)),
        "best_epoch_from_history": best_epoch,
        "best_score_from_history": best_score,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_score": checkpoint_score,
        "checkpoint_epoch_matches": checkpoint_epoch == best_epoch,
        "checkpoint_score_abs_error": abs(checkpoint_score - best_score) if np.isfinite(checkpoint_score) else None,
        "final_epoch": int(final_row.get("epoch", 0)),
        "final_score": float(final_row.get("score", 0.0)),
        "best_train_loss": finite_or_nan(best_row.get(train_loss_key)),
        "best_val_loss": finite_or_nan(best_row.get(val_loss_key)),
        "best_loss_gap": finite_or_nan(best_row.get(val_loss_key)) - finite_or_nan(best_row.get(train_loss_key)),
        "final_train_loss": finite_or_nan(final_row.get(train_loss_key)),
        "final_val_loss": finite_or_nan(final_row.get(val_loss_key)),
        "final_loss_gap": finite_or_nan(final_row.get(val_loss_key)) - finite_or_nan(final_row.get(train_loss_key)),
        "score_drop_after_best": best_score - float(final_row.get("score", 0.0)),
        "best_phase": best_row.get("phase", ""),
        "final_phase": final_row.get("phase", ""),
    }


def plot_history(output_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    epochs = np.asarray([row.get("epoch", i + 1) for i, row in enumerate(rows)], dtype=float)

    def line_plot(filename: str, title: str, specs: Sequence[Tuple[str, str]], ylabel: str):
        available = [(key, label) for key, label in specs if any(isinstance(row.get(key), (int, float)) for row in rows)]
        if not available:
            return
        fig, ax = plt.subplots(figsize=(9, 6))
        for key, label in available:
            values = [finite_or_nan(row.get(key)) for row in rows]
            ax.plot(epochs, values, label=label)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)

    line_plot("training_score.png", "Stage-2 validation composite score", (("score", "score"),), "score")
    line_plot("training_loss.png", "Stage-2 train/validation loss", (("train_loss", "train"), ("val_loss", "val")), "loss")
    line_plot(
        "training_correlations.png",
        "Stage-2 validation target correlations",
        (("val_physical_corr_mean", "physical"), ("val_response_corr_mean", "response"), ("val_behavior_corr_mean", "behavior")),
        "mean Pearson r",
    )
    line_plot(
        "training_counterfactuals.png",
        "Stage-2 validation counterfactual metrics",
        (("val_policy_pair_accuracy", "policy pair"), ("val_pseudo_pair_accuracy", "pseudo-A pair"), ("val_pseudo_score_drop", "pseudo-A drop"), ("val_policy_score_drop", "policy drop")),
        "metric",
    )
    line_plot(
        "training_representation.png",
        "Stage-2 validation representation health",
        (("val_physical_embedding_effective_rank", "H physical erank"), ("val_context_embedding_effective_rank", "H context erank"), ("val_stage1_anchor_embedding_effective_rank", "Stage-1 erank"), ("val_embedding_rank_retention", "rank retention")),
        "value",
    )


def build_worst_case_rows(
    per_sample_rows: Sequence[Mapping[str, Any]],
    user_metric_rows: Sequence[Mapping[str, Any]],
    top_k: int = 100,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    sample_rows: List[Dict[str, Any]] = []
    for row in per_sample_rows:
        item = dict(row)
        behavior_error = abs(
            float(item.get("behavior_storage_behavior_pred", 0.0))
            - float(item.get("behavior_storage_behavior_target", 0.0))
        )
        physical_error = abs(
            float(item.get("physical_storage_evidence_pred", 0.0))
            - float(item.get("physical_storage_evidence_target", 0.0))
        )
        response_error = abs(
            float(item.get("response_response_confidence_pred", 0.0))
            - float(item.get("response_response_confidence_target", 0.0))
        )
        item["storage_behavior_abs_error"] = behavior_error
        item["storage_evidence_abs_error"] = physical_error
        item["response_confidence_abs_error"] = response_error
        item["core_error_mean"] = float(np.mean([behavior_error, physical_error, response_error]))
        sample_rows.append(item)
    sample_rows.sort(key=lambda row: (row.get("split") != "val", -float(row["core_error_mean"])))
    worst_samples = sample_rows[:top_k]

    chosen_users = [
        dict(row) for row in user_metric_rows
        if row.get("group") == "behavior" and row.get("name") == "storage_behavior"
    ]
    chosen_users.sort(key=lambda row: (row.get("split") != "val", -float(row.get("mae", 0.0))))
    return worst_samples, chosen_users[:top_k]


def evaluate_verdict(
    checkpoint: Mapping[str, Any],
    integrity_failures: Sequence[str],
    integrity_warnings: Sequence[str],
    target_metrics: Sequence[Mapping[str, Any]],
    pseudo_summary: Sequence[Mapping[str, Any]],
    policy_summary: Sequence[Mapping[str, Any]],
    policy_invariance: Mapping[str, Mapping[str, float]],
    representation_health: Mapping[str, Any],
    core_rows: Sequence[Mapping[str, Any]],
    p_only_result: Mapping[str, Any],
) -> Dict[str, Any]:
    reasons = list(integrity_failures)
    warnings_list = list(integrity_warnings)

    val_rows = [row for row in target_metrics if row["split"] == "val"]
    thresholds = {"physical": 0.50, "response": 0.50, "behavior": 0.45}
    for row in val_rows:
        corr_threshold = thresholds[row["group"]]
        if float(row["target_std"]) < 0.03:
            warnings_list.append(f"{row['group']}.{row['name']} 目标方差偏低：{row['target_std']:.4f}")
        if float(row["pearson"]) < corr_threshold:
            reasons.append(
                f"{row['group']}.{row['name']} Pearson不足：{row['pearson']:.3f} < {corr_threshold:.2f}"
            )
        if float(row["mae"]) > 0.18:
            reasons.append(f"{row['group']}.{row['name']} MAE过高：{row['mae']:.3f}")
        if float(row["pred_std"]) < 0.005:
            reasons.append(f"{row['group']}.{row['name']} 输出接近常数：std={row['pred_std']:.4f}")
        elif float(row["pred_std"]) < 0.01:
            warnings_list.append(f"{row['group']}.{row['name']} 输出方差偏低：std={row['pred_std']:.4f}")
        if float(row["calibration_monotonicity"]) < 0.50:
            warnings_list.append(
                f"{row['group']}.{row['name']} 分箱单调性偏低：{row['calibration_monotonicity']:.3f}"
            )

    val_repr = representation_health.get("val", {})
    for view in ("physical", "context"):
        stats = val_repr.get(view, {})
        if not stats:
            reasons.append(f"缺少 val {view} 表征诊断")
            continue
        if stats.get("raw_std_min", 0.0) < 0.005:
            reasons.append(f"val {view} raw_std_min 接近塌缩：{stats.get('raw_std_min', 0.0):.4f}")
        elif stats.get("raw_std_min", 0.0) < 0.01:
            warnings_list.append(f"val {view} raw_std_min 偏低：{stats.get('raw_std_min', 0.0):.4f}")
        if stats.get("normalized_std_min", 0.0) < 0.012 and stats.get("effective_rank", 0.0) < 4.0:
            reasons.append(f"val {view} 归一化表征接近塌缩")
        elif stats.get("normalized_std_min", 0.0) < 0.02:
            warnings_list.append(
                f"val {view} normalized_std_min 偏低：{stats.get('normalized_std_min', 0.0):.4f}"
            )
        if stats.get("effective_rank", 0.0) < 6.0:
            warnings_list.append(f"val {view} 有效秩偏低：{stats.get('effective_rank', 0.0):.2f}")
    rank_retention = val_repr.get("rank_retention")
    if rank_retention is not None:
        if rank_retention < 0.25:
            reasons.append(f"Stage-2 相对 Stage-1 有效秩保留不足：{rank_retention:.3f}")
        elif rank_retention < 0.45:
            warnings_list.append(f"Stage-2 相对 Stage-1 有效秩保留偏低：{rank_retention:.3f}")

    pseudo_map = {
        row["variant"]: row
        for row in pseudo_summary
        if row["split"] == "val"
    }
    all_pseudo = pseudo_map.get("all", {})
    if float(all_pseudo.get("score_drop_mean", 0.0)) < 0.08:
        reasons.append("A类错配总体分数下降不足")
    if float(all_pseudo.get("pair_accuracy", 0.0)) < 0.70:
        reasons.append("A类错配总体成对准确率不足")
    for variant in ("shift", "shift_sign"):
        row = pseudo_map.get(variant, {})
        if float(row.get("pair_accuracy", 0.0)) < 0.60:
            reasons.append(f"A类 {variant} 成对准确率不足")
        if float(row.get("score_drop_mean", 0.0)) < 0.03:
            warnings_list.append(f"A类 {variant} 平均分数下降偏低")

    policy_map = {
        row["name"]: row
        for row in policy_summary
        if row["split"] == "val"
    }
    policy = policy_map.get("storage_behavior", {})
    target_drop = float(policy.get("target_drop_mean", 0.0))
    minimum_drop = max(0.03, 0.5 * target_drop)
    if float(policy.get("predicted_drop_mean", 0.0)) < minimum_drop:
        reasons.append("政策错配 storage_behavior 分数下降不足")
    if float(policy.get("pair_accuracy", 0.0)) < 0.70:
        reasons.append("同步政策迁移高于政策错配的比例不足")
    if float(policy.get("drop_pearson", 0.0)) < 0.0:
        reasons.append("政策响应强弱 Pearson 为负")
    elif float(policy.get("drop_pearson", 0.0)) < 0.20:
        warnings_list.append("政策响应强弱 Pearson 偏低")
    val_invariance = policy_invariance.get("val", {})
    if float(val_invariance.get("physical_max_error", 1.0)) > 1e-6:
        reasons.append("只替换政策角色时 physical head 发生变化")
    if float(val_invariance.get("response_max_error", 1.0)) > 1e-6:
        reasons.append("只替换政策角色时 response head 发生变化")

    val_core = [row for row in core_rows if row["split"] == "val"]
    excess_values = [
        float(row["excess"])
        for row in val_core
        if row["metric"] == "pair_correlation" and row["excess"] != ""
    ]
    pred_rank = next((float(row["prediction"]) for row in val_core if row["metric"] == "effective_rank"), 0.0)
    target_rank = next((float(row["target"]) for row in val_core if row["metric"] == "effective_rank"), 0.0)
    if excess_values and max(excess_values) > 0.20 and pred_rank < 0.75 * max(target_rank, EPS):
        warnings_list.append("三个核心教师分数可能存在多头复制")

    if p_only_result.get("status") == "OK":
        val_probe = p_only_result.get("eval", {}).get("val", {})
        teacher_reg = val_probe.get("teacher_output_regression", {})
        storage_probe = teacher_reg.get("behavior.storage_behavior", {})
        if storage_probe and float(storage_probe.get("corr", 0.0)) < 0.35:
            warnings_list.append(
                f"P-only 对 Teacher storage_behavior 可蒸馏相关性偏低：{storage_probe.get('corr', 0.0):.3f}"
            )
    elif p_only_result:
        warnings_list.append(f"P-only 探针未完成：{p_only_result.get('reason', 'unknown')}")

    reasons = list(dict.fromkeys(reasons))
    warnings_list = list(dict.fromkeys(warnings_list))
    verdict = "REVIEW" if reasons else ("PASS_WITH_WARNINGS" if warnings_list else "PASS")
    return {
        "verdict": verdict,
        "reasons": reasons,
        "warnings": warnings_list,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_score": checkpoint.get("best_score"),
        "target_pipeline_version": checkpoint.get("target_pipeline_version"),
        "stage2_objective_version": checkpoint.get("stage2_objective_version"),
        "model_arch_version": checkpoint.get("model_arch_version"),
    }


def write_markdown_summary(
    path: Path,
    verdict: Mapping[str, Any],
    target_metrics: Sequence[Mapping[str, Any]],
    pseudo_summary: Sequence[Mapping[str, Any]],
    policy_summary: Sequence[Mapping[str, Any]],
    representation_health: Mapping[str, Any],
    user_level_summary: Sequence[Mapping[str, Any]],
    p_only_result: Mapping[str, Any],
    output_files: Sequence[str],
) -> None:
    lines = [
        "# Stage-2 教师完整诊断摘要",
        "",
        f"- verdict: **{verdict['verdict']}**",
        f"- checkpoint_epoch: {verdict.get('checkpoint_epoch')}",
        f"- checkpoint_score: {verdict.get('checkpoint_score')}",
        "",
    ]
    if verdict.get("reasons"):
        lines.extend(["## 必须处理的问题", ""])
        lines.extend([f"- {item}" for item in verdict["reasons"]])
        lines.append("")
    if verdict.get("warnings"):
        lines.extend(["## 警告", ""])
        lines.extend([f"- {item}" for item in verdict["warnings"]])
        lines.append("")

    lines.extend(["## Validation 目标拟合", "", "| group | name | Pearson | Spearman | MAE | pred_std | target_std |", "|---|---|---:|---:|---:|---:|---:|"])
    for row in target_metrics:
        if row["split"] == "val":
            lines.append(
                f"| {row['group']} | {row['name']} | {row['pearson']:.3f} | "
                f"{row['spearman']:.3f} | {row['mae']:.3f} | "
                f"{row['pred_std']:.3f} | {row['target_std']:.3f} |"
            )

    lines.extend(["", "## Validation A 类错配", "", "| variant | score_drop | pair_accuracy | q05 | q50 | q95 |", "|---|---:|---:|---:|---:|---:|"])
    for row in pseudo_summary:
        if row["split"] == "val":
            lines.append(
                f"| {row['variant']} | {row['score_drop_mean']:.3f} | "
                f"{row['pair_accuracy']:.3f} | {row['score_drop_q05']:.3f} | "
                f"{row['score_drop_q50']:.3f} | {row['score_drop_q95']:.3f} |"
            )

    lines.extend(["", "## Validation 政策反事实", "", "| target | predicted_drop | target_drop | pair | drop_r |", "|---|---:|---:|---:|---:|"])
    for row in policy_summary:
        if row["split"] == "val":
            lines.append(
                f"| {row['name']} | {row['predicted_drop_mean']:.3f} | "
                f"{row['target_drop_mean']:.3f} | {row['pair_accuracy']:.3f} | "
                f"{row['drop_pearson']:.3f} |"
            )

    val_repr = representation_health.get("val", {})
    lines.extend(["", "## Validation 表征健康", "", "| view | raw_std_min | norm_std_min | erank | k90 | k99 |", "|---|---:|---:|---:|---:|---:|"])
    for view in ("physical", "context", "transfer_context", "mismatch_context", "stage1_anchor"):
        stats = val_repr.get(view)
        if stats:
            lines.append(
                f"| {view} | {stats['raw_std_min']:.4f} | {stats['normalized_std_min']:.4f} | "
                f"{stats['effective_rank']:.2f} | {stats['k90']} | {stats['k99']} |"
            )
    if "rank_retention" in val_repr:
        lines.append("")
        lines.append(f"- Stage-2 / Stage-1 effective-rank retention: {val_repr['rank_retention']:.3f}")

    if user_level_summary:
        lines.extend(["", "## 用户级汇总", "", "| split | target | user Pearson | user Spearman | user MAE |", "|---|---|---:|---:|---:|"])
        for row in user_level_summary:
            lines.append(
                f"| {row['split']} | {row['group']}.{row['name']} | "
                f"{row['user_mean_pearson']:.3f} | {row['user_mean_spearman']:.3f} | "
                f"{row['user_mean_mae']:.3f} |"
            )

    lines.extend(["", "## P-only 可蒸馏性", ""])
    if p_only_result.get("status") == "OK":
        for split, result in p_only_result.get("eval", {}).items():
            if result.get("status") != "OK":
                lines.append(f"- {split}: {result.get('reason', 'SKIPPED')}")
                continue
            teacher_storage = result.get("teacher_output_regression", {}).get("behavior.storage_behavior", {})
            pseudo = result.get("pseudo_b_binary", {})
            lines.append(
                f"- {split}: teacher storage_behavior corr={teacher_storage.get('corr', 0.0):.3f}, "
                f"pseudo-B paired_accuracy={pseudo.get('paired_accuracy', 0.0):.3f}, "
                f"AUC={pseudo.get('auc', 0.0):.3f}（仅反事实辅助）"
            )
    else:
        lines.append(f"- {p_only_result.get('reason', '未执行')}")

    lines.extend(["", "## 输出文件", ""])
    lines.extend([f"- `{name}`" for name in output_files])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Stage-2 教师完整诊断（增强版）")
    parser.add_argument("--phase2_ckpt", type=str, default="step_2/best_model_st2.pt")
    parser.add_argument("--phase1_ckpt", type=str, default=None)
    parser.add_argument("--history_path", type=str, default=None, help="默认使用 checkpoint 同目录/phase2_training_history.csv")
    parser.add_argument("--data_path", type=str, default="processed_result.xlsx")
    parser.add_argument("--split_manifest", type=str, default="data_split/user_split.csv")
    parser.add_argument("--user_col", type=str, default="关联用户号")
    parser.add_argument("--policy_name", type=str, default="old")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--time_type", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--max_train_samples", type=int, default=8192)
    parser.add_argument("--max_eval_samples", type=int, default=0, help="0 表示 val/test 全量")
    parser.add_argument("--max_per_user", type=int, default=0, help="0 表示不限制每户天数")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--out_dir", type=str, default="diag_step2_full")
    parser.add_argument("--pseudo_shift_points", type=int, default=8)
    parser.add_argument("--calibration_bins", type=int, default=10)
    parser.add_argument("--plot_max_points", type=int, default=5000)
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--skip_p_only_probe", action="store_true")
    parser.add_argument("--p_only_train_samples", type=int, default=6000)
    parser.add_argument("--p_only_eval_samples", type=int, default=3000)
    parser.add_argument("--force_cache", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    st2.set_seed(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve_path(args.phase2_ckpt)
    output_dir = resolve_path(args.out_dir)
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Stage-2 checkpoint 不存在：{checkpoint_path}")

    model, checkpoint, model_kwargs = load_model(checkpoint_path, device)
    data = load_storage_data(
        data_path=args.data_path,
        train_ratio=args.train_ratio,
        time_type=args.time_type,
        seed=args.seed,
        split_manifest_path=args.split_manifest,
        user_col=args.user_col,
        policy_name=args.policy_name,
    )
    target_config = TeacherTargetConfig(**checkpoint["data_info"]["target_config"])
    target_cache_path = output_dir / "teacher_targets_v5.npz"
    targets = build_stage2_target_cache(
        data,
        target_cache_path,
        config=target_config,
        force=args.force_cache,
    )
    transfer_config = PolicyTransferConfig(**checkpoint["data_info"]["policy_transfer_config"])
    transfer_cache_path = checkpoint_path.parent / "policy_transfer_v3.npz"
    transfer = build_policy_transfer_cache(
        data,
        transfer_cache_path,
        config=transfer_config,
        seed=args.seed,
        force=args.force_cache,
    )
    transfer_targets, mismatch_targets = st2.build_transfer_target_cache(
        checkpoint_path.parent / "teacher_transfer_targets_v1.npz",
        data,
        transfer,
        target_config,
        transfer_config,
    )
    transfer_inputs = torch.from_numpy(np.asarray(transfer["positive_inputs"], dtype=np.float32))
    mismatch_array = np.asarray(transfer["positive_inputs"], dtype=np.float32).copy()
    mismatch_array[:, 3, :] = np.asarray(transfer["mismatch_role_seq"], dtype=np.float32)
    mismatch_array[:, 4, :] = np.asarray(transfer["mismatch_role_jump"], dtype=np.float32)
    mismatch_inputs = torch.from_numpy(mismatch_array)
    sample_weight = data["meta"]["sample_weight"].to_numpy(np.float32)

    history_path = resolve_path(args.history_path) if args.history_path else checkpoint_path.parent / "phase2_training_history.csv"
    history_rows = load_history(history_path)
    history_summary = history_diagnostics(history_rows, checkpoint)
    save_json(output_dir / "training_history_summary.json", history_summary)

    integrity_rows, integrity_failures, integrity_warnings = validate_integrity(
        checkpoint, model, data, targets, transfer
    )
    save_csv(output_dir / "integrity_checks.csv", integrity_rows)

    stage1_model, stage1_warning = load_stage1_anchor_model(
        checkpoint, model_kwargs, args.phase1_ckpt, device
    )
    if stage1_warning:
        integrity_warnings.append(stage1_warning)

    target_metric_rows: List[Dict[str, Any]] = []
    target_distribution_rows: List[Dict[str, Any]] = []
    calibration_bin_rows: List[Dict[str, Any]] = []
    per_sample_rows: List[Dict[str, Any]] = []
    pseudo_pair_rows: List[Dict[str, Any]] = []
    pseudo_summary_rows: List[Dict[str, Any]] = []
    policy_pair_rows: List[Dict[str, Any]] = []
    policy_summary_rows: List[Dict[str, Any]] = []
    policy_strata_rows: List[Dict[str, Any]] = []
    policy_invariance: Dict[str, Dict[str, float]] = {}
    representation_health: Dict[str, Any] = {}
    spectrum_rows: List[Dict[str, Any]] = []
    core_rows: List[Dict[str, Any]] = []
    user_metric_rows: List[Dict[str, Any]] = []
    user_summary_rows: List[Dict[str, Any]] = []
    split_predictions: Dict[str, Dict[str, np.ndarray]] = {}
    selected_indices_by_split: Dict[str, np.ndarray] = {}

    target_std = float((checkpoint.get("args", {}) or {}).get("target_std", 0.08))
    covariance_weight = float((checkpoint.get("args", {}) or {}).get("reg_covariance_weight", 5.0))

    for split_offset, split in enumerate(("train", "val", "test")):
        max_samples = args.max_train_samples if split == "train" else args.max_eval_samples
        indices = select_indices(
            data["split_indices"][split],
            data["meta"],
            max_samples=max_samples,
            max_per_user=args.max_per_user,
            seed=args.seed + split_offset,
        )
        selected_indices_by_split[split] = indices
        print(f"[Diag] split={split}, samples={len(indices)}, users={data['meta'].iloc[indices]['user_id'].nunique()}")
        predictions = collect_predictions(
            model,
            stage1_model,
            data["X"],
            transfer_inputs,
            mismatch_inputs,
            indices,
            device,
            args.batch_size,
            args.pseudo_shift_points,
        )
        split_predictions[split] = predictions
        for name, values in predictions.items():
            if not np.isfinite(values).all():
                integrity_failures.append(f"{split}.{name} 包含 NaN/Inf")

        metrics, distributions, calibration = build_target_diagnostics(
            split,
            predictions,
            targets,
            indices,
            sample_weight,
            args.calibration_bins,
        )
        target_metric_rows.extend(metrics)
        target_distribution_rows.extend(distributions)
        calibration_bin_rows.extend(calibration)
        per_sample_rows.extend(make_per_sample_rows(
            split, indices, data["meta"], predictions, targets
        ))
        pseudo_pairs, pseudo_summary = pseudo_a_diagnostics(
            split,
            indices,
            data["meta"],
            predictions,
            targets,
            sample_weight,
        )
        pseudo_pair_rows.extend(pseudo_pairs)
        pseudo_summary_rows.extend(pseudo_summary)
        policy_pairs, policy_summary, policy_strata, invariance = policy_diagnostics(
            split,
            indices,
            data["meta"],
            predictions,
            transfer_targets,
            mismatch_targets,
            transfer,
            sample_weight,
        )
        policy_pair_rows.extend(policy_pairs)
        policy_summary_rows.extend(policy_summary)
        policy_strata_rows.extend(policy_strata)
        policy_invariance[split] = invariance
        repr_result, split_spectrum = representation_diagnostics(
            split, predictions, target_std, covariance_weight
        )
        representation_health[split] = repr_result
        spectrum_rows.extend(split_spectrum)
        core_rows.extend(core_head_diagnostics(split, predictions, targets, indices))
        user_rows, user_summary = build_user_metrics(
            split, indices, data["meta"], predictions, targets
        )
        user_metric_rows.extend(user_rows)
        user_summary_rows.extend(user_summary)

    save_csv(output_dir / "target_metrics.csv", target_metric_rows)
    save_csv(output_dir / "target_distributions.csv", target_distribution_rows)
    save_csv(output_dir / "calibration_bins.csv", calibration_bin_rows)
    save_csv(output_dir / "per_sample_predictions.csv", per_sample_rows)
    save_csv(output_dir / "pseudo_a_pairs.csv", pseudo_pair_rows)
    save_csv(output_dir / "pseudo_a_summary.csv", pseudo_summary_rows)
    save_csv(output_dir / "policy_counterfactual_pairs.csv", policy_pair_rows)
    save_csv(output_dir / "policy_counterfactual_summary.csv", policy_summary_rows)
    save_csv(output_dir / "policy_counterfactual_strata.csv", policy_strata_rows)
    save_csv(output_dir / "representation_spectrum.csv", spectrum_rows)
    save_csv(output_dir / "core_head_metrics.csv", core_rows)
    save_csv(output_dir / "user_metrics.csv", user_metric_rows)
    save_csv(output_dir / "user_level_summary.csv", user_summary_rows)
    save_json(output_dir / "representation_health.json", representation_health)

    p_only_result: Dict[str, Any]
    if args.skip_p_only_probe:
        p_only_result = {"status": "SKIPPED", "reason": "--skip_p_only_probe"}
    else:
        try:
            p_only_result = run_p_only_probe(
                model,
                data,
                transfer,
                target_config,
                device,
                args.batch_size,
                args.seed,
                args.p_only_train_samples,
                args.p_only_eval_samples,
            )
        except Exception as exc:
            p_only_result = {"status": "FAILED", "reason": repr(exc)}
            integrity_warnings.append(f"P-only 探针失败：{exc}")
    save_json(output_dir / "p_only_distillability.json", p_only_result)

    if not args.skip_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_history(plots_dir, history_rows)
        for split in ("val", "test"):
            predictions = split_predictions[split]
            indices = selected_indices_by_split[split]
            for group, names, target_key, _ in GROUP_SPECS:
                for target_index, name in enumerate(names):
                    plot_target_diagnostic(
                        plots_dir,
                        split,
                        group,
                        name,
                        predictions[group][:, target_index],
                        targets[target_key][indices, target_index],
                        calibration_bin_rows,
                        args.plot_max_points,
                        args.seed + target_index,
                    )
            plot_representation_spectrum(plots_dir, spectrum_rows, split)
            plot_pseudo(plots_dir, pseudo_pair_rows, split)
            plot_policy(plots_dir, policy_pair_rows, split)
            plot_core_heads(plots_dir, split, predictions)
            plot_user_summary(plots_dir, user_metric_rows, split)

    worst_samples, worst_users = build_worst_case_rows(per_sample_rows, user_metric_rows)
    save_csv(output_dir / "worst_samples.csv", worst_samples)
    save_csv(output_dir / "worst_users.csv", worst_users)

    verdict = evaluate_verdict(
        checkpoint,
        integrity_failures,
        integrity_warnings,
        target_metric_rows,
        pseudo_summary_rows,
        policy_summary_rows,
        policy_invariance,
        representation_health,
        core_rows,
        p_only_result,
    )
    full_result = {
        "verdict": verdict,
        "checkpoint": {
            "path": str(checkpoint_path),
            "epoch": checkpoint.get("epoch"),
            "best_score": checkpoint.get("best_score"),
            "model_arch_version": checkpoint.get("model_arch_version"),
            "target_pipeline_version": checkpoint.get("target_pipeline_version"),
            "stage2_objective_version": checkpoint.get("stage2_objective_version"),
            "validation_metrics_saved_in_checkpoint": checkpoint.get("validation_metrics"),
            "data_info": checkpoint.get("data_info"),
        },
        "training_history": history_summary,
        "runtime": {
            "device": str(device),
            "selected_samples": {split: int(len(indices)) for split, indices in selected_indices_by_split.items()},
            "selected_users": {
                split: int(data["meta"].iloc[indices]["user_id"].nunique())
                for split, indices in selected_indices_by_split.items()
            },
            "stage1_anchor_loaded": stage1_model is not None,
            "target_std": target_std,
            "covariance_weight": covariance_weight,
        },
        "integrity": integrity_rows,
        "target_metrics": target_metric_rows,
        "target_distributions": target_distribution_rows,
        "pseudo_a_summary": pseudo_summary_rows,
        "policy_counterfactual_summary": policy_summary_rows,
        "policy_invariance": policy_invariance,
        "representation_health": representation_health,
        "core_head_metrics": core_rows,
        "user_level_summary": user_summary_rows,
        "p_only_distillability": p_only_result,
    }
    save_json(output_dir / "diagnostics_full.json", full_result)

    output_files = [
        "diagnostics_full.json",
        "diagnostics_summary.md",
        "integrity_checks.csv",
        "training_history_summary.json",
        "target_metrics.csv",
        "target_distributions.csv",
        "calibration_bins.csv",
        "per_sample_predictions.csv",
        "user_metrics.csv",
        "user_level_summary.csv",
        "worst_samples.csv",
        "worst_users.csv",
        "pseudo_a_pairs.csv",
        "pseudo_a_summary.csv",
        "policy_counterfactual_pairs.csv",
        "policy_counterfactual_summary.csv",
        "policy_counterfactual_strata.csv",
        "representation_health.json",
        "representation_spectrum.csv",
        "core_head_metrics.csv",
        "p_only_distillability.json",
        "plots/",
    ]
    write_markdown_summary(
        output_dir / "diagnostics_summary.md",
        verdict,
        target_metric_rows,
        pseudo_summary_rows,
        policy_summary_rows,
        representation_health,
        user_summary_rows,
        p_only_result,
        output_files,
    )
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    print(f"[Done] outputs={output_dir}")


if __name__ == "__main__":
    main()
