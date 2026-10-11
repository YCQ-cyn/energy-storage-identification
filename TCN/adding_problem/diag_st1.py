# -*- coding: utf-8 -*-
"""
Stage-1 comprehensive diagnostics for the storage-response project (Ridge-stable revision).

This script is designed for the current Stage-1 pipeline:
  P/Q -> TCN-LSTM backbone H -> projection head Z
  clean-pq-v3-shared-p-scale

It does NOT retrain or modify the checkpoint. It audits:
  1) checkpoint/data/split/normalization consistency;
  2) train/val/test representation health for H and Z;
  3) augmentation invariance and positive/negative distance distributions;
  4) P/Q channel usage (zero/permutation ablations);
  5) train-fitted PCA applied to train/val/test in the same coordinate system;
  6) user-identity concentration in H;
  7) physical linear probes on unseen validation users;
  8) optional full-curve linear reconstruction probe;
  9) clipping rates and shared-P-scale normalization diagnostics;
 10) a machine-readable acceptance report and a Chinese summary report.

Recommended usage (run from TCN/adding_problem):
  python diag_st1.py --ckpt step_1/best_model_st1.pt --cuda

The script imports the project's current add_test_st1.py and utils.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_PARENT = SCRIPT_DIR.parents[1] if len(SCRIPT_DIR.parents) >= 2 else SCRIPT_DIR
for candidate in (SCRIPT_DIR, PROJECT_PARENT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    from add_test_st1 import (
        AugmentConfig,
        EncoderWithHead,
        apply_sync_intensity_to_x,
        set_seed,
    )
    from utils import (
        DATA_PIPELINE_VERSION,
        P_CLIP_MAX,
        Q_CLIP_ABS,
        load_storage_data,
    )
except ImportError:
    from TCN.adding_problem.add_test_st1 import (
        AugmentConfig,
        EncoderWithHead,
        apply_sync_intensity_to_x,
        set_seed,
    )
    from TCN.adding_problem.utils import (
        DATA_PIPELINE_VERSION,
        P_CLIP_MAX,
        Q_CLIP_ABS,
        load_storage_data,
    )


EPS = 1e-12
GLOBAL_LOG_FILE = None


class TeeLogger:
    def __init__(self, stream, file_handle):
        self.stream = stream
        self.file_handle = file_handle

    def write(self, message: str) -> None:
        self.stream.write(message)
        self.file_handle.write(message)

    def flush(self) -> None:
        self.stream.flush()
        self.file_handle.flush()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(dict(payload)), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def setup_logging(out_dir: Path, ckpt: Path) -> Path:
    global GLOBAL_LOG_FILE
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"{timestamp}_stage1_diagnostics.md"
    GLOBAL_LOG_FILE = log_path.open("w", encoding="utf-8")
    GLOBAL_LOG_FILE.write(
        "# Stage-1 综合诊断日志\n\n"
        f"- time: {timestamp}\n"
        f"- checkpoint: {ckpt}\n"
        f"- output: {out_dir}\n\n"
        "```text\n"
    )
    GLOBAL_LOG_FILE.flush()
    sys.stdout = TeeLogger(sys.__stdout__, GLOBAL_LOG_FILE)
    sys.stderr = TeeLogger(sys.__stderr__, GLOBAL_LOG_FILE)
    return log_path


def close_logging() -> None:
    global GLOBAL_LOG_FILE
    if GLOBAL_LOG_FILE is not None:
        try:
            GLOBAL_LOG_FILE.write("\n```\n")
            GLOBAL_LOG_FILE.flush()
        finally:
            GLOBAL_LOG_FILE.close()
            GLOBAL_LOG_FILE = None


def resolve_existing_path(path_value: str, search_roots: Sequence[Path]) -> Path:
    path = Path(path_value).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.extend(root / path for root in search_roots)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"找不到文件: {path_value}; searched={[str(c) for c in candidates]}")


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = pd.Series(np.asarray(x).reshape(-1)).rank(method="average").to_numpy()
    y_rank = pd.Series(np.asarray(y).reshape(-1)).rank(method="average").to_numpy()
    return safe_corr(x_rank, y_rank)


def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
        y_pred = y_pred[:, None]
    weights = weights / max(weights.sum(), EPS)
    mean = np.sum(y_true * weights[:, None], axis=0, keepdims=True)
    numerator = np.sum(weights[:, None] * np.square(y_true - y_pred))
    denominator = np.sum(weights[:, None] * np.square(y_true - mean))
    return float(1.0 - numerator / max(denominator, EPS))


def weighted_mae(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if y_true.ndim == 1:
        error = np.abs(y_true - y_pred)
    else:
        error = np.abs(y_true - y_pred).mean(axis=1)
    return float(np.sum(error * weights) / max(weights.sum(), EPS))


def user_balanced_weights(user_ids: Sequence[str]) -> np.ndarray:
    user_ids = np.asarray(user_ids).astype(str)
    _, inverse, counts = np.unique(user_ids, return_inverse=True, return_counts=True)
    weights = 1.0 / np.maximum(counts[inverse], 1)
    return weights / max(weights.mean(), EPS)


def sample_indices(indices: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if max_points <= 0 or len(indices) <= max_points:
        return indices.copy()
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(indices, size=max_points, replace=False))


def instantiate_model(args: argparse.Namespace, device: torch.device) -> EncoderWithHead:
    return EncoderWithHead(
        input_channels=2,
        nhid=int(args.nhid),
        levels=int(args.levels),
        ksize=int(args.ksize),
        dropout=float(args.dropout),
        lstm_hidden=int(getattr(args, "lstm_hidden", args.nhid)),
        lstm_layers=int(getattr(args, "lstm_layers", 1)),
        embedding_dim=int(getattr(args, "embedding_dim", args.nhid)),
        proj_dim=int(getattr(args, "proj_dim", 128)),
    ).to(device)


def load_checkpoint_and_data(cli_args, device: torch.device):
    ckpt_path = resolve_existing_path(
        cli_args.ckpt,
        (Path.cwd(), SCRIPT_DIR, SCRIPT_DIR / "step_1"),
    )
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError("checkpoint 缺少 model_state_dict")
    checkpoint_args = argparse.Namespace(**checkpoint.get("args", {}))

    data_path = cli_args.data_path or getattr(checkpoint_args, "data_path", "processed_result.xlsx")
    split_manifest = cli_args.split_manifest or getattr(
        checkpoint_args, "split_manifest", "data_split/user_split.csv"
    )
    user_col = cli_args.user_col or getattr(checkpoint_args, "user_col", "关联用户号")
    policy_name = getattr(checkpoint_args, "policy_name", "old")
    time_type = getattr(checkpoint_args, "time_type", None)
    train_ratio = float(getattr(checkpoint_args, "train_ratio", 0.9))
    seed = int(getattr(checkpoint_args, "seed", 1111))

    data = load_storage_data(
        data_path=data_path,
        train_ratio=train_ratio,
        time_type=time_type,
        seed=seed,
        split_manifest_path=split_manifest,
        user_col=user_col,
        policy_name=policy_name,
    )

    model = instantiate_model(checkpoint_args, device)
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint 结构不匹配: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    model.eval()

    return ckpt_path, checkpoint, checkpoint_args, data, model


def audit_integrity(
    ckpt_path: Path,
    checkpoint: Mapping[str, Any],
    data: Mapping[str, Any],
) -> Dict[str, Any]:
    meta = data["meta"]
    split_users = {
        split: set(meta.loc[meta["split"] == split, "user_id"].astype(str))
        for split in ("train", "val", "test")
    }
    overlaps = {
        "train_val": sorted(split_users["train"] & split_users["val"]),
        "train_test": sorted(split_users["train"] & split_users["test"]),
        "val_test": sorted(split_users["val"] & split_users["test"]),
    }
    checkpoint_data = checkpoint.get("data_info", {}) or {}
    checks = {
        "checkpoint_exists": ckpt_path.is_file(),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_sha256": file_sha256(ckpt_path),
        "pipeline_current": data.get("pipeline_version", ""),
        "pipeline_checkpoint": checkpoint_data.get("pipeline_version", ""),
        "pipeline_match": checkpoint_data.get("pipeline_version", "") in {
            "",
            data.get("pipeline_version", ""),
        },
        "data_sha_current": data.get("data_sha256", ""),
        "data_sha_checkpoint": checkpoint_data.get("data_sha256", ""),
        "data_sha_match": checkpoint_data.get("data_sha256", "") in {
            "",
            data.get("data_sha256", ""),
        },
        "split_sha_current": data.get("split_manifest_sha256", ""),
        "split_sha_checkpoint": checkpoint_data.get("split_manifest_sha256", ""),
        "split_sha_match": checkpoint_data.get("split_manifest_sha256", "") in {
            "",
            data.get("split_manifest_sha256", ""),
        },
        "split_nonempty": {
            split: int((meta["split"] == split).sum()) > 0 for split in ("train", "val", "test")
        },
        "user_counts": {split: len(users) for split, users in split_users.items()},
        "sample_counts": {
            split: int((meta["split"] == split).sum()) for split in ("train", "val", "test")
        },
        "user_overlap": overlaps,
        "user_disjoint": all(len(value) == 0 for value in overlaps.values()),
        "normalization": data.get("normalization", {}),
    }
    checks["integrity_ok"] = bool(
        checks["pipeline_match"]
        and checks["data_sha_match"]
        and checks["split_sha_match"]
        and checks["user_disjoint"]
        and all(checks["split_nonempty"].values())
    )
    return checks


def clipping_diagnostics(data: Mapping[str, Any]) -> Dict[str, Any]:
    x = data["X"].cpu().numpy()
    meta = data["meta"]
    result: Dict[str, Any] = {
        "normalization": data.get("normalization", {}),
        "splits": {},
    }
    for split in ("train", "val", "test"):
        idx = data["split_indices"][split]
        p = x[idx, 0]
        q = x[idx, 1]
        p_point_clip = np.isclose(p, float(data["normalization"].get("p_clip_max", P_CLIP_MAX)), atol=1e-6)
        q_point_clip = np.isclose(np.abs(q), float(data["normalization"].get("q_clip_abs", Q_CLIP_ABS)), atol=1e-6)
        split_meta = meta.iloc[idx]
        result["splits"][split] = {
            "p_point_clip_rate": float(p_point_clip.mean()),
            "q_point_clip_rate": float(q_point_clip.mean()),
            "p_sample_clip_rate": float(p_point_clip.any(axis=1).mean()),
            "q_sample_clip_rate": float(q_point_clip.any(axis=1).mean()),
            "p_scale_ref_quantiles": np.quantile(
                split_meta["p_scale_ref"].to_numpy(dtype=np.float64),
                [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1],
            ),
            "q_to_p_p95_quantiles": np.quantile(
                split_meta["q_to_p_p95"].to_numpy(dtype=np.float64),
                [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1],
            ),
        }
    return result


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    x: torch.Tensor,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    h_list: List[np.ndarray] = []
    z_list: List[np.ndarray] = []
    index_tensor = torch.from_numpy(np.asarray(indices, dtype=np.int64))
    for start in range(0, len(index_tensor), batch_size):
        batch_indices = index_tensor[start : start + batch_size]
        xb = x.index_select(0, batch_indices).to(device=device, dtype=torch.float32)
        h, z = model(xb)
        h_list.append(h.cpu().numpy())
        z_list.append(z.cpu().numpy())
    if not h_list:
        raise ValueError("embedding 提取样本为空")
    return np.concatenate(h_list), np.concatenate(z_list)


def spectrum_stats(embedding: np.ndarray) -> Dict[str, Any]:
    embedding = np.asarray(embedding, dtype=np.float64)
    finite = bool(np.isfinite(embedding).all())
    if not finite:
        embedding = np.nan_to_num(embedding)
    centered = embedding - embedding.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)[::-1]
    total = float(eigenvalues.sum())
    if total <= EPS:
        probabilities = np.zeros_like(eigenvalues)
        k90 = k99 = 0
        eig1_ratio = 1.0
        erank = 0.0
        participation_ratio = 0.0
    else:
        probabilities = eigenvalues / total
        cumulative = np.cumsum(probabilities)
        k90 = int(np.searchsorted(cumulative, 0.90) + 1)
        k99 = int(np.searchsorted(cumulative, 0.99) + 1)
        eig1_ratio = float(probabilities[0])
        positive = probabilities[probabilities > 1e-15]
        erank = float(np.exp(-np.sum(positive * np.log(positive))))
        participation_ratio = float(1.0 / np.sum(np.square(probabilities)))

    std = embedding.std(axis=0)
    corr = np.corrcoef(centered, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    offdiag = np.abs(corr[~np.eye(corr.shape[0], dtype=bool)])
    norm = np.linalg.norm(embedding, axis=1)
    thresholds = (0.015, 0.02, 0.03, 0.05)
    return {
        "sample_count": int(len(embedding)),
        "dimension": int(embedding.shape[1]),
        "finite": finite,
        "std_min": float(std.min()),
        "std_mean": float(std.mean()),
        "std_quantiles": np.quantile(std, [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]),
        "dead_dims": {str(threshold): int((std < threshold).sum()) for threshold in thresholds},
        "norm_mean": float(norm.mean()),
        "norm_quantiles": np.quantile(norm, [0, 0.05, 0.5, 0.95, 1]),
        "k90": k90,
        "k99": k99,
        "eig1_ratio": eig1_ratio,
        "effective_rank": erank,
        "participation_ratio": participation_ratio,
        "mean_abs_offdiag_corr": float(offdiag.mean()) if offdiag.size else 0.0,
        "p95_abs_offdiag_corr": float(np.quantile(offdiag, 0.95)) if offdiag.size else 0.0,
        "max_abs_offdiag_corr": float(offdiag.max()) if offdiag.size else 0.0,
        "eigenvalues": eigenvalues,
    }


def plot_spectrum_and_std(
    spectra: Mapping[str, Mapping[str, Mapping[str, Any]]],
    out_dir: Path,
) -> None:
    for representation in ("H", "Z"):
        plt.figure(figsize=(7, 5))
        for split in ("train", "val", "test"):
            values = np.asarray(spectra[split][representation]["eigenvalues"], dtype=np.float64)
            plt.plot(np.arange(1, len(values) + 1), np.maximum(values, 1e-12), label=split)
        plt.yscale("log")
        plt.xlabel("eigenvalue index")
        plt.ylabel("eigenvalue (log)")
        plt.title(f"{representation} covariance spectrum")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"st1_{representation.lower()}_spectrum.png", dpi=180)
        plt.close()

        plt.figure(figsize=(7, 5))
        for split in ("train", "val", "test"):
            std_quantiles = spectra[split][representation]["std_quantiles"]
            # Detailed values are generated again from output CSV, so the plot uses quantile markers.
            plt.plot(
                [0, 5, 25, 50, 75, 95, 100],
                std_quantiles,
                marker="o",
                label=split,
            )
        plt.xlabel("dimension std percentile")
        plt.ylabel("std")
        plt.title(f"{representation} dimension-std quantiles")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"st1_{representation.lower()}_std_quantiles.png", dpi=180)
        plt.close()


def deterministic_augment(
    x_batch: torch.Tensor,
    config: AugmentConfig,
    seed: int,
) -> torch.Tensor:
    outputs = []
    for index, sample in enumerate(x_batch):
        rng = np.random.RandomState(seed + index * 1009)
        outputs.append(apply_sync_intensity_to_x(sample, config, rng=rng))
    return torch.stack(outputs)


@torch.no_grad()
def distance_diagnostics(
    model: torch.nn.Module,
    x: torch.Tensor,
    indices: np.ndarray,
    device: torch.device,
    config: AugmentConfig,
    batch_size: int,
    max_samples: int,
    seed: int,
    out_dir: Path,
    split: str,
) -> Dict[str, Any]:
    selected = sample_indices(indices, max_samples, seed)
    h_pos, z_pos, h_neg, z_neg = [], [], [], []
    h_cos_pos, z_cos_pos, h_cos_neg, z_cos_neg = [], [], [], []
    index_tensor = torch.from_numpy(selected)
    for start in range(0, len(index_tensor), batch_size):
        batch_indices = index_tensor[start : start + batch_size]
        xb = x.index_select(0, batch_indices).to(device=device, dtype=torch.float32)
        x1 = deterministic_augment(xb, config, seed + start * 7)
        x2 = deterministic_augment(xb, config, seed + start * 7 + 1)
        h1, z1 = model(x1)
        h2, z2 = model(x2)
        if len(xb) > 1:
            permutation = torch.roll(torch.arange(len(xb), device=device), shifts=1)
        else:
            permutation = torch.arange(len(xb), device=device)
        h2_neg = h2[permutation]
        z2_neg = z2[permutation]
        h_pos.append(torch.sum((h1 - h2).square(), dim=1).cpu().numpy())
        z_pos.append(torch.sum((z1 - z2).square(), dim=1).cpu().numpy())
        h_neg.append(torch.sum((h1 - h2_neg).square(), dim=1).cpu().numpy())
        z_neg.append(torch.sum((z1 - z2_neg).square(), dim=1).cpu().numpy())
        h_cos_pos.append(F.cosine_similarity(h1, h2, dim=1).cpu().numpy())
        z_cos_pos.append(F.cosine_similarity(z1, z2, dim=1).cpu().numpy())
        h_cos_neg.append(F.cosine_similarity(h1, h2_neg, dim=1).cpu().numpy())
        z_cos_neg.append(F.cosine_similarity(z1, z2_neg, dim=1).cpu().numpy())

    arrays = {
        "H": {
            "pos": np.concatenate(h_pos),
            "neg": np.concatenate(h_neg),
            "cos_pos": np.concatenate(h_cos_pos),
            "cos_neg": np.concatenate(h_cos_neg),
        },
        "Z": {
            "pos": np.concatenate(z_pos),
            "neg": np.concatenate(z_neg),
            "cos_pos": np.concatenate(z_cos_pos),
            "cos_neg": np.concatenate(z_cos_neg),
        },
    }
    result: Dict[str, Any] = {"sample_count": int(len(selected))}
    for representation, values in arrays.items():
        pos, neg = values["pos"], values["neg"]
        result[representation] = {
            "d2_pos_mean": float(pos.mean()),
            "d2_neg_mean": float(neg.mean()),
            "d2_ratio": float(pos.mean() / max(neg.mean(), EPS)),
            "d2_pos_quantiles": np.quantile(pos, [0.5, 0.9, 0.95, 0.99, 1]),
            "d2_neg_quantiles": np.quantile(neg, [0.01, 0.05, 0.5, 0.95, 1]),
            "cos_pos_mean": float(values["cos_pos"].mean()),
            "cos_neg_mean": float(values["cos_neg"].mean()),
            "distance_order_accuracy": float(np.mean(pos < neg)),
        }
        plt.figure(figsize=(7, 5))
        upper = float(np.quantile(np.concatenate([pos, neg]), 0.995))
        bins = np.linspace(0.0, max(upper, 1e-6), 80)
        plt.hist(pos, bins=bins, density=True, alpha=0.55, label="same sample / two views")
        plt.hist(neg, bins=bins, density=True, alpha=0.45, label="different samples")
        plt.xlabel("squared distance")
        plt.ylabel("density")
        plt.title(f"{split} {representation}: positive vs negative distance")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"st1_{split}_{representation.lower()}_d2.png", dpi=180)
        plt.close()
    return result


@torch.no_grad()
def augmentation_sensitivity(
    model: torch.nn.Module,
    x: torch.Tensor,
    indices: np.ndarray,
    device: torch.device,
    checkpoint_config: AugmentConfig,
    batch_size: int,
    max_samples: int,
    seed: int,
) -> Dict[str, Any]:
    selected = sample_indices(indices, max_samples, seed)
    configs = {
        "scale_only": AugmentConfig(0.9, 1.1, 0.0, checkpoint_config.smooth_window, 0.0),
        "noise_only": AugmentConfig(1.0, 1.0, 0.0, checkpoint_config.smooth_window, checkpoint_config.noise_std),
        "smooth_only": AugmentConfig(1.0, 1.0, 1.0, checkpoint_config.smooth_window, 0.0),
        "combined": checkpoint_config,
    }
    accum: Dict[str, Dict[str, List[np.ndarray]]] = {
        name: {"h_rel": [], "z_rel": [], "h_cos": [], "z_cos": []} for name in configs
    }
    index_tensor = torch.from_numpy(selected)
    for start in range(0, len(index_tensor), batch_size):
        batch_indices = index_tensor[start : start + batch_size]
        xb = x.index_select(0, batch_indices).to(device=device, dtype=torch.float32)
        h0, z0 = model(xb)
        for name, config in configs.items():
            xa = deterministic_augment(xb, config, seed + start * 11 + len(name))
            ha, za = model(xa)
            accum[name]["h_rel"].append(
                (torch.norm(ha - h0, dim=1) / (torch.norm(h0, dim=1) + 1e-6)).cpu().numpy()
            )
            accum[name]["z_rel"].append(
                (torch.norm(za - z0, dim=1) / (torch.norm(z0, dim=1) + 1e-6)).cpu().numpy()
            )
            accum[name]["h_cos"].append(F.cosine_similarity(ha, h0, dim=1).cpu().numpy())
            accum[name]["z_cos"].append(F.cosine_similarity(za, z0, dim=1).cpu().numpy())
    result = {}
    for name, values in accum.items():
        result[name] = {}
        for key, chunks in values.items():
            array = np.concatenate(chunks)
            result[name][key + "_mean"] = float(array.mean())
            result[name][key + "_p95"] = float(np.quantile(array, 0.95))
    return result


@torch.no_grad()
def channel_ablation(
    model: torch.nn.Module,
    x: torch.Tensor,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    max_samples: int,
    seed: int,
) -> Dict[str, Any]:
    selected = sample_indices(indices, max_samples, seed)
    records = {
        name: {"h_rel": [], "z_rel": [], "h_cos": [], "z_cos": []}
        for name in ("zero_P", "zero_Q", "permute_P", "permute_Q")
    }
    index_tensor = torch.from_numpy(selected)
    for start in range(0, len(index_tensor), batch_size):
        batch_indices = index_tensor[start : start + batch_size]
        xb = x.index_select(0, batch_indices).to(device=device, dtype=torch.float32)
        h0, z0 = model(xb)
        permutation = torch.roll(torch.arange(len(xb), device=device), shifts=1)
        variants = {}
        zero_p = xb.clone(); zero_p[:, 0] = 0.0; variants["zero_P"] = zero_p
        zero_q = xb.clone(); zero_q[:, 1] = 0.0; variants["zero_Q"] = zero_q
        perm_p = xb.clone(); perm_p[:, 0] = xb[permutation, 0]; variants["permute_P"] = perm_p
        perm_q = xb.clone(); perm_q[:, 1] = xb[permutation, 1]; variants["permute_Q"] = perm_q
        for name, xv in variants.items():
            h, z = model(xv)
            records[name]["h_rel"].append(
                (torch.norm(h - h0, dim=1) / (torch.norm(h0, dim=1) + 1e-6)).cpu().numpy()
            )
            records[name]["z_rel"].append(
                (torch.norm(z - z0, dim=1) / (torch.norm(z0, dim=1) + 1e-6)).cpu().numpy()
            )
            records[name]["h_cos"].append(F.cosine_similarity(h, h0, dim=1).cpu().numpy())
            records[name]["z_cos"].append(F.cosine_similarity(z, z0, dim=1).cpu().numpy())
    result: Dict[str, Any] = {}
    for name, values in records.items():
        result[name] = {}
        for key, chunks in values.items():
            array = np.concatenate(chunks)
            result[name][key + "_mean"] = float(array.mean())
            result[name][key + "_p95"] = float(np.quantile(array, 0.95))
    return result


def fit_train_pca_and_plot(
    embeddings: Mapping[str, np.ndarray],
    metadata: Mapping[str, pd.DataFrame],
    physical_targets: Mapping[str, Mapping[str, np.ndarray]],
    out_dir: Path,
    seed: int,
    max_plot_points: int,
) -> Dict[str, Any]:
    train_h = embeddings["train"]
    n_components = min(10, train_h.shape[1])
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(train_h)
    transformed = {split: pca.transform(embeddings[split]) for split in embeddings}
    stats = {
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
    }

    plt.figure(figsize=(9, 7))
    for split, marker, alpha in (("train", ".", 0.12), ("val", "o", 0.35), ("test", "^", 0.35)):
        coords = transformed[split]
        rng = np.random.RandomState(seed + {"train": 1, "val": 2, "test": 3}[split])
        if len(coords) > max_plot_points:
            selected = rng.choice(len(coords), max_plot_points, replace=False)
            coords = coords[selected]
        plt.scatter(coords[:, 0], coords[:, 1], s=8, alpha=alpha, marker=marker, label=split)
    plt.xlabel("PC1 (train-fitted)")
    plt.ylabel("PC2 (train-fitted)")
    plt.title("Stage-1 H: train-fitted PCA, shared coordinates")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "st1_pca_train_val_test_shared.png", dpi=190)
    plt.close()

    val_coords = transformed["val"]
    val_meta = metadata["val"].reset_index(drop=True)
    user_codes, user_names = pd.factorize(val_meta["user_id"].astype(str), sort=True)
    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(val_coords[:, 0], val_coords[:, 1], c=user_codes, s=9, alpha=0.55, cmap="tab20")
    plt.colorbar(scatter, label="validation user code")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"Validation H PCA by user ({len(user_names)} users)")
    plt.tight_layout()
    plt.savefig(out_dir / "st1_pca_val_by_user.png", dpi=190)
    plt.close()

    for target_name in (
        "storage_to_base_ratio",
        "energy_balance",
        "charge_in_valley_ratio",
        "discharge_in_peak_ratio",
    ):
        values = physical_targets["val"][target_name]
        plt.figure(figsize=(9, 7))
        scatter = plt.scatter(val_coords[:, 0], val_coords[:, 1], c=values, s=9, alpha=0.55, cmap="viridis")
        plt.colorbar(scatter, label=target_name)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"Validation H PCA by {target_name}")
        plt.tight_layout()
        plt.savefig(out_dir / f"st1_pca_val_by_{target_name}.png", dpi=190)
        plt.close()
        stats[f"corr_pc1_{target_name}"] = safe_corr(val_coords[:, 0], values)
        stats[f"corr_pc2_{target_name}"] = safe_corr(val_coords[:, 1], values)
        stats[f"spearman_pc1_{target_name}"] = safe_spearman(val_coords[:, 0], values)
        stats[f"spearman_pc2_{target_name}"] = safe_spearman(val_coords[:, 1], values)
    return stats


def user_identity_diagnostics(
    h: np.ndarray,
    meta: pd.DataFrame,
    k: int = 5,
    max_samples: int = 6000,
    seed: int = 1111,
) -> Dict[str, Any]:
    if len(h) <= k:
        return {"available": False, "reason": "too_few_samples"}
    if max_samples > 0 and len(h) > max_samples:
        rng = np.random.RandomState(seed)
        selected = np.sort(rng.choice(len(h), max_samples, replace=False))
        h = h[selected]
        meta = meta.iloc[selected].reset_index(drop=True)
    scaler = StandardScaler().fit(h)
    h_scaled = scaler.transform(h)
    user_ids = meta["user_id"].astype(str).to_numpy()
    neighbors = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(h_scaled)
    _, indices = neighbors.kneighbors(h_scaled)
    neighbor_users = user_ids[indices[:, 1:]]
    same = neighbor_users == user_ids[:, None]
    hit_rate = float(same.mean())
    top1 = float(same[:, 0].mean())
    _, counts = np.unique(user_ids, return_counts=True)
    chance = float(np.sum(np.square(counts / counts.sum())))

    centroids = {}
    within_distances = []
    for user_id in np.unique(user_ids):
        values = h_scaled[user_ids == user_id]
        centroid = values.mean(axis=0)
        centroids[user_id] = centroid
        within_distances.extend(np.linalg.norm(values - centroid, axis=1).tolist())
    centroid_matrix = np.stack(list(centroids.values()))
    if len(centroid_matrix) > 1:
        pairwise = np.linalg.norm(
            centroid_matrix[:, None, :] - centroid_matrix[None, :, :], axis=2
        )
        upper = pairwise[np.triu_indices(len(centroid_matrix), k=1)]
        between_centroid_mean = float(upper.mean())
    else:
        between_centroid_mean = float("nan")
    within_mean = float(np.mean(within_distances))
    return {
        "available": True,
        "k": k,
        "same_user_neighbor_rate": hit_rate,
        "same_user_top1_rate": top1,
        "chance_rate": chance,
        "lift_over_chance": float(hit_rate / max(chance, EPS)),
        "within_user_distance_mean": within_mean,
        "between_user_centroid_distance_mean": between_centroid_mean,
        "between_within_ratio": float(between_centroid_mean / max(within_mean, EPS)),
    }


def compute_physical_targets(data: Mapping[str, Any], indices: np.ndarray) -> Dict[str, np.ndarray]:
    p = np.asarray(data["P_raw"])[indices].astype(np.float64)
    q = np.asarray(data["Q_raw"])[indices].astype(np.float64)
    x = data["X"].cpu().numpy()[indices]
    role = x[:, 3, :]
    charge = np.clip(q, 0.0, None)
    discharge = np.clip(-q, 0.0, None)
    base = p - q
    charge_energy = charge.sum(axis=1)
    discharge_energy = discharge.sum(axis=1)
    total_action = charge_energy + discharge_energy
    storage_to_base = np.abs(q).sum(axis=1) / (np.abs(base).sum(axis=1) + 1e-6)
    energy_balance = 1.0 - np.abs(charge_energy - discharge_energy) / (total_action + 1e-6)
    energy_balance = np.where(total_action > 1e-8, energy_balance, 0.0)
    charge_in_valley = (charge * (role < -0.5)).sum(axis=1) / (charge_energy + 1e-6)
    discharge_in_peak = (discharge * (role > 0.5)).sum(axis=1) / (discharge_energy + 1e-6)
    base_negative_ratio = (base < 0.0).mean(axis=1)
    return {
        "storage_to_base_ratio": storage_to_base.astype(np.float32),
        "energy_balance": energy_balance.astype(np.float32),
        "charge_in_valley_ratio": charge_in_valley.astype(np.float32),
        "discharge_in_peak_ratio": discharge_in_peak.astype(np.float32),
        "base_negative_ratio": base_negative_ratio.astype(np.float32),
    }


def split_probe_train_users(user_ids: np.ndarray, seed: int, tune_fraction: float = 0.2):
    unique_users = np.unique(user_ids.astype(str))
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(unique_users)
    tune_count = max(1, int(round(len(unique_users) * tune_fraction)))
    tune_count = min(tune_count, len(unique_users) - 1)
    tune_users = set(shuffled[:tune_count].tolist())
    tune_mask = np.array([str(user) in tune_users for user in user_ids])
    fit_mask = ~tune_mask
    return fit_mask, tune_mask


def _fit_ridge_stable(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    alpha: float,
    solver: str,
    tol: float,
    max_iter: int,
) -> Tuple[Ridge, List[str]]:
    """Fit Ridge in float64 and record every numerical warning."""
    model = Ridge(
        alpha=float(alpha),
        solver=str(solver),
        tol=float(tol),
        max_iter=int(max_iter),
        copy_X=True,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            sample_weight=np.asarray(sample_weight, dtype=np.float64),
        )
    messages = [f"{type(item.message).__name__}: {item.message}" for item in caught]
    return model, messages


def fit_ridge_group_tuned(
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_users: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    val_users: np.ndarray,
    alphas: Sequence[float],
    seed: int,
    fit_mask: Optional[np.ndarray] = None,
    tune_mask: Optional[np.ndarray] = None,
    solver: str = "lsqr",
    tol: float = 1e-8,
    max_iter: int = 10000,
) -> Dict[str, Any]:
    # Use float64 throughout the probe. This changes diagnostics only, never the checkpoint.
    x_train = np.asarray(x_train, dtype=np.float64)
    x_val = np.asarray(x_val, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    y_val = np.asarray(y_val, dtype=np.float64)
    train_users = np.asarray(train_users).astype(str)
    val_users = np.asarray(val_users).astype(str)

    if fit_mask is None or tune_mask is None:
        fit_mask, tune_mask = split_probe_train_users(train_users, seed)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    tune_mask = np.asarray(tune_mask, dtype=bool)
    if not fit_mask.any() or not tune_mask.any():
        raise ValueError("Ridge probe fit/tune user split cannot be empty")

    scaler_tune = StandardScaler().fit(x_train[fit_mask])
    x_fit = scaler_tune.transform(x_train[fit_mask]).astype(np.float64, copy=False)
    x_tune = scaler_tune.transform(x_train[tune_mask]).astype(np.float64, copy=False)
    tune_weights = user_balanced_weights(train_users[tune_mask]).astype(np.float64)
    fit_weights = user_balanced_weights(train_users[fit_mask]).astype(np.float64)

    alpha_results: List[Dict[str, Any]] = []
    best_alpha: Optional[float] = None
    best_score = -np.inf
    for alpha in alphas:
        model, warning_messages = _fit_ridge_stable(
            x_fit, y_train[fit_mask], fit_weights, alpha, solver, tol, max_iter
        )
        prediction = model.predict(x_tune)
        score = weighted_r2(y_train[tune_mask], prediction, tune_weights)
        if not np.isfinite(score):
            score = -np.inf
        alpha_results.append({
            "alpha": float(alpha),
            "tune_user_balanced_r2": float(score),
            "warning_count": len(warning_messages),
            "warnings": warning_messages[:5],
            "n_iter": None if getattr(model, "n_iter_", None) is None
            else np.asarray(model.n_iter_).reshape(-1).tolist(),
        })
        if score > best_score:
            best_score = float(score)
            best_alpha = float(alpha)

    if best_alpha is None:
        raise RuntimeError("No finite Ridge alpha candidate was found")

    scaler = StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train).astype(np.float64, copy=False)
    x_val_scaled = scaler.transform(x_val).astype(np.float64, copy=False)
    final, final_warning_messages = _fit_ridge_stable(
        x_train_scaled,
        y_train,
        user_balanced_weights(train_users),
        best_alpha,
        solver,
        tol,
        max_iter,
    )
    train_pred = final.predict(x_train_scaled)
    val_pred = final.predict(x_val_scaled)
    train_weights = user_balanced_weights(train_users)
    val_weights = user_balanced_weights(val_users)
    return {
        "selected_alpha": best_alpha,
        "alpha_search": alpha_results,
        "solver": solver,
        "tol": float(tol),
        "max_iter": int(max_iter),
        "fit_user_count": int(np.unique(train_users[fit_mask]).size),
        "tune_user_count": int(np.unique(train_users[tune_mask]).size),
        "final_warning_count": len(final_warning_messages),
        "final_warnings": final_warning_messages[:10],
        "final_n_iter": None if getattr(final, "n_iter_", None) is None
        else np.asarray(final.n_iter_).reshape(-1).tolist(),
        "train_r2_user_balanced": weighted_r2(y_train, train_pred, train_weights),
        "val_r2_user_balanced": weighted_r2(y_val, val_pred, val_weights),
        "val_mae_user_balanced": weighted_mae(y_val, val_pred, val_weights),
        "val_pearson": safe_corr(y_val, val_pred),
        "val_spearman": safe_spearman(y_val, val_pred),
        "val_target_std": float(np.std(y_val)),
        "prediction_std": float(np.std(val_pred)),
    }


def physical_probe(
    trained_h: Mapping[str, np.ndarray],
    random_h: Mapping[str, np.ndarray],
    raw_features: Mapping[str, np.ndarray],
    targets: Mapping[str, Mapping[str, np.ndarray]],
    metadata: Mapping[str, pd.DataFrame],
    alphas: Sequence[float],
    seed: int,
    solver: str = "lsqr",
    tol: float = 1e-8,
    max_iter: int = 10000,
) -> Dict[str, Any]:
    train_users = metadata["train"]["user_id"].astype(str).to_numpy()
    val_users = metadata["val"]["user_id"].astype(str).to_numpy()
    result: Dict[str, Any] = {"targets": {}, "composite_targets": [
        "storage_to_base_ratio",
        "energy_balance",
        "charge_in_valley_ratio",
        "discharge_in_peak_ratio",
    ]}
    feature_sets = {
        "trained_H": trained_h,
        "random_H": random_h,
        "raw_PQ_normalized": raw_features,
    }
    for target_name in targets["train"]:
        result["targets"][target_name] = {}
        # Fair comparison: trained_H, random_H and raw_PQ must tune on the same users.
        target_seed = seed + sum(ord(char) for char in target_name)
        fit_mask, tune_mask = split_probe_train_users(train_users, target_seed)
        for feature_name, features in feature_sets.items():
            result["targets"][target_name][feature_name] = fit_ridge_group_tuned(
                features["train"],
                targets["train"][target_name],
                train_users,
                features["val"],
                targets["val"][target_name],
                val_users,
                alphas,
                target_seed,
                fit_mask=fit_mask,
                tune_mask=tune_mask,
                solver=solver,
                tol=tol,
                max_iter=max_iter,
            )
    composites = {}
    for feature_name in feature_sets:
        values = [
            result["targets"][target][feature_name]["val_r2_user_balanced"]
            for target in result["composite_targets"]
        ]
        composites[feature_name] = float(np.mean(values))
    result["composite"] = composites
    result["trained_minus_random"] = composites["trained_H"] - composites["random_H"]
    result["probe_config"] = {
        "alphas": [float(value) for value in alphas],
        "solver": solver,
        "tol": float(tol),
        "max_iter": int(max_iter),
        "fair_same_tune_users_across_features": True,
        "dtype": "float64",
    }
    warning_count = 0
    for target_results in result["targets"].values():
        for metrics in target_results.values():
            warning_count += int(metrics.get("final_warning_count", 0))
            warning_count += sum(
                int(item.get("warning_count", 0))
                for item in metrics.get("alpha_search", [])
            )
    result["numerical_warning_count"] = int(warning_count)
    result["numerically_stable"] = bool(warning_count == 0)
    return result


def curve_probe(
    trained_h: Mapping[str, np.ndarray],
    random_h: Mapping[str, np.ndarray],
    target_curves: Mapping[str, np.ndarray],
    metadata: Mapping[str, pd.DataFrame],
    alphas: Sequence[float],
    seed: int,
    solver: str = "lsqr",
    tol: float = 1e-8,
    max_iter: int = 10000,
) -> Dict[str, Any]:
    train_users = metadata["train"]["user_id"].astype(str).to_numpy()
    val_users = metadata["val"]["user_id"].astype(str).to_numpy()
    result = {}
    # Fair comparison: both feature sets use exactly the same fit/tune users.
    fit_mask, tune_mask = split_probe_train_users(train_users, seed)
    for name, features in (("trained_H", trained_h), ("random_H", random_h)):
        result[name] = fit_ridge_group_tuned(
            features["train"],
            target_curves["train"],
            train_users,
            features["val"],
            target_curves["val"],
            val_users,
            alphas,
            seed,
            fit_mask=fit_mask,
            tune_mask=tune_mask,
            solver=solver,
            tol=tol,
            max_iter=max_iter,
        )
    result["trained_minus_random"] = (
        result["trained_H"]["val_r2_user_balanced"]
        - result["random_H"]["val_r2_user_balanced"]
    )
    result["probe_config"] = {
        "alphas": [float(value) for value in alphas],
        "solver": solver,
        "tol": float(tol),
        "max_iter": int(max_iter),
        "fair_same_tune_users_across_features": True,
        "dtype": "float64",
    }
    warning_count = 0
    for feature_name in ("trained_H", "random_H"):
        metrics = result[feature_name]
        warning_count += int(metrics.get("final_warning_count", 0))
        warning_count += sum(
            int(item.get("warning_count", 0))
            for item in metrics.get("alpha_search", [])
        )
    result["numerical_warning_count"] = int(warning_count)
    result["numerically_stable"] = bool(warning_count == 0)
    return result


def render_probe_stability_summary(
    physical: Mapping[str, Any],
    curve: Optional[Mapping[str, Any]],
    checkpoint_path: Path,
    checkpoint_epoch: Any,
    solver: str,
    alphas: Sequence[float],
) -> str:
    lines = [
        "# Stage-1 稳定物理探针复核",
        "",
        f"- checkpoint: `{checkpoint_path}`",
        f"- checkpoint epoch: `{checkpoint_epoch}`",
        f"- Ridge solver: `{solver}`",
        f"- alpha candidates: `{[float(v) for v in alphas]}`",
        "- feature/target dtype: `float64`",
        "- scaler: only fit on probe training users",
        "- alpha tuning: user-disjoint inner split from training users",
        "",
        "## Physical probe composite",
        "",
        f"- trained_H: `{physical['composite']['trained_H']:.6f}`",
        f"- random_H: `{physical['composite']['random_H']:.6f}`",
        f"- raw_PQ_normalized: `{physical['composite']['raw_PQ_normalized']:.6f}`",
        f"- trained_H - random_H: `{physical['trained_minus_random']:+.6f}`",
        f"- numerical warning count: `{physical.get('numerical_warning_count', 0)}`",
        f"- numerically stable: `{physical.get('numerically_stable', False)}`",
        "",
        "## Per-target validation user-balanced R²",
        "",
        "| target | trained_H | random_H | raw_PQ | trained-random |",
        "|---|---:|---:|---:|---:|",
    ]
    for target, feature_results in physical["targets"].items():
        trained = feature_results["trained_H"]["val_r2_user_balanced"]
        random_value = feature_results["random_H"]["val_r2_user_balanced"]
        raw_value = feature_results["raw_PQ_normalized"]["val_r2_user_balanced"]
        lines.append(
            f"| {target} | {trained:.6f} | {random_value:.6f} | "
            f"{raw_value:.6f} | {trained-random_value:+.6f} |"
        )
    if curve is not None:
        lines.extend([
            "",
            "## Full-curve probe",
            "",
            f"- trained_H val R²: `{curve['trained_H']['val_r2_user_balanced']:.6f}`",
            f"- random_H val R²: `{curve['random_H']['val_r2_user_balanced']:.6f}`",
            f"- trained_H - random_H: `{curve['trained_minus_random']:+.6f}`",
            f"- numerical warning count: `{curve.get('numerical_warning_count', 0)}`",
            f"- numerically stable: `{curve.get('numerically_stable', False)}`",
        ])
    lines.extend([
        "",
        "## Interpretation",
        "",
        "This report only re-checks numerical stability of the linear probes. "
        "It does not retrain or modify Stage-1.",
        "",
    ])
    return "\n".join(lines)


def write_probe_csv(probe: Mapping[str, Any], path: Path) -> None:
    rows = []
    for target, feature_results in probe["targets"].items():
        for feature_name, metrics in feature_results.items():
            rows.append({
                "target": target,
                "feature": feature_name,
                "selected_alpha": metrics["selected_alpha"],
                "train_r2_user_balanced": metrics["train_r2_user_balanced"],
                "val_r2_user_balanced": metrics["val_r2_user_balanced"],
                "val_mae_user_balanced": metrics["val_mae_user_balanced"],
                "val_pearson": metrics["val_pearson"],
                "val_spearman": metrics["val_spearman"],
            })
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def acceptance_decision(report: Mapping[str, Any]) -> Dict[str, Any]:
    integrity = report["integrity"]
    h_val = report["spectra"]["val"]["H"]
    z_val = report["spectra"]["val"]["Z"]
    d2_val = report["distance"]["val"]["Z"]
    physical = report["physical_probe"]
    clipping = report["clipping"]["splits"]
    channel = report["channel_ablation"]

    checks = {
        "integrity": bool(integrity["integrity_ok"]),
        "finite_embeddings": bool(h_val["finite"] and z_val["finite"]),
        "h_noncollapse": bool(
            h_val["effective_rank"] >= 8
            and h_val["k99"] >= 10
            and h_val["std_min"] > 0.015
            and h_val["eig1_ratio"] < 0.60
        ),
        "z_noncollapse": bool(
            z_val["effective_rank"] >= 20
            and z_val["k99"] >= 20
            and z_val["eig1_ratio"] < 0.40
        ),
        "view_invariance": bool(
            d2_val["d2_ratio"] < 0.10
            and d2_val["distance_order_accuracy"] > 0.95
        ),
        "normalization_clipping_ok": bool(
            clipping["train"]["p_point_clip_rate"] < 0.02
            and clipping["train"]["q_point_clip_rate"] < 0.02
            and clipping["val"]["p_point_clip_rate"] < 0.03
            and clipping["val"]["q_point_clip_rate"] < 0.03
        ),
        "physical_probe_numerically_stable": bool(
            physical.get("numerically_stable", False)
        ),
        "physical_probe_not_materially_worse_than_random": bool(
            physical.get("numerically_stable", False)
            and physical["trained_minus_random"] >= -0.03
        ),
        "uses_both_channels": bool(
            channel["zero_P"]["h_rel_mean"] > 0.03
            and channel["zero_Q"]["h_rel_mean"] > 0.03
        ),
    }
    hard_checks = (
        "integrity",
        "finite_embeddings",
        "h_noncollapse",
        "z_noncollapse",
        "view_invariance",
        "normalization_clipping_ok",
        "uses_both_channels",
    )
    accepted = all(checks[name] for name in hard_checks)
    warnings = []
    if not checks["physical_probe_numerically_stable"]:
        warnings.append(
            "物理线性探针仍存在数值求解警告；当前trained-random差值只作参考，"
            "不得据此否定或通过Stage-1。"
        )
    elif not checks["physical_probe_not_materially_worse_than_random"]:
        warnings.append("训练后H的核心物理线性探针明显弱于随机H；必须先审查再进入Stage-2。")
    if report["user_identity"]["val"].get("lift_over_chance", 0.0) > 3.0:
        warnings.append("验证集H存在较强同用户聚集；Stage-2必须继续采用用户独立验证。")
    if report.get("curve_probe") and report["curve_probe"].get("trained_minus_random", 0.0) < -0.15:
        warnings.append("训练后H的整曲线线性恢复能力低于随机H；Stage-2主干不宜永久冻结。")
    if not checks["normalization_clipping_ok"]:
        warnings.append("共享尺度归一化存在较高裁剪率，请检查P_CLIP_MAX/Q_CLIP_ABS。")

    return {
        "accepted": accepted,
        "core_checks": checks,
        "warnings": warnings,
        "stage2_ablation_required": True,
        "interpretation": (
            "通过仅表示Stage-1结构健康、数据链路一致，可作为Stage-2候选；"
            "不能替代Stage-2随机初始化消融。"
        ),
    }


def render_summary_markdown(report: Mapping[str, Any], acceptance: Mapping[str, Any]) -> str:
    integrity = report["integrity"]
    h = report["spectra"]["val"]["H"]
    z = report["spectra"]["val"]["Z"]
    d2 = report["distance"]["val"]["Z"]
    probe = report["physical_probe"]
    clipping = report["clipping"]["splits"]
    user_diag = report["user_identity"]["val"]
    channel = report["channel_ablation"]

    lines = [
        "# Stage-1 综合诊断结论",
        "",
        f"- checkpoint epoch: {integrity['checkpoint_epoch']}",
        f"- pipeline: {integrity['pipeline_current']}",
        f"- accepted: **{acceptance['accepted']}**",
        f"- Stage-2 ablation required: **{acceptance['stage2_ablation_required']}**",
        "",
        "## 核心结构指标（验证用户）",
        "",
        "| 指标 | H | Z |",
        "|---|---:|---:|",
        f"| effective rank | {h['effective_rank']:.4f} | {z['effective_rank']:.4f} |",
        f"| k90 | {h['k90']} | {z['k90']} |",
        f"| k99 | {h['k99']} | {z['k99']} |",
        f"| eig1 ratio | {h['eig1_ratio']:.4f} | {z['eig1_ratio']:.4f} |",
        f"| std min | {h['std_min']:.4f} | {z['std_min']:.4f} |",
        f"| std mean | {h['std_mean']:.4f} | {z['std_mean']:.4f} |",
        "",
        "## 视图一致性",
        "",
        f"- Z d2 positive mean: {d2['d2_pos_mean']:.6f}",
        f"- Z d2 negative mean: {d2['d2_neg_mean']:.6f}",
        f"- Z d2 ratio: {d2['d2_ratio']:.6f}",
        f"- distance order accuracy: {d2['distance_order_accuracy']:.4f}",
        "",
        "## 物理线性探针",
        "",
        f"- trained H composite: {probe['composite']['trained_H']:.4f}",
        f"- random H composite: {probe['composite']['random_H']:.4f}",
        f"- trained-random: {probe['trained_minus_random']:+.4f}",
        f"- Ridge solver: {probe.get('probe_config', {}).get('solver')}",
        f"- numerical warning count: {probe.get('numerical_warning_count', 0)}",
        f"- numerically stable: {probe.get('numerically_stable', False)}",
        "",
        "| 目标 | trained H | random H | raw normalized P/Q |",
        "|---|---:|---:|---:|",
    ]
    for target in probe["composite_targets"]:
        item = probe["targets"][target]
        lines.append(
            f"| {target} | {item['trained_H']['val_r2_user_balanced']:.4f} | "
            f"{item['random_H']['val_r2_user_balanced']:.4f} | "
            f"{item['raw_PQ_normalized']['val_r2_user_balanced']:.4f} |"
        )
    lines.extend([
        "",
        "## 归一化与裁剪",
        "",
        f"- q_global_scale: {report['clipping']['normalization'].get('q_global_scale')}",
        f"- train P/Q point clip rate: {clipping['train']['p_point_clip_rate']:.6f} / {clipping['train']['q_point_clip_rate']:.6f}",
        f"- val P/Q point clip rate: {clipping['val']['p_point_clip_rate']:.6f} / {clipping['val']['q_point_clip_rate']:.6f}",
        "",
        "## 通道使用",
        "",
        f"- zero P -> H relative change: {channel['zero_P']['h_rel_mean']:.4f}",
        f"- zero Q -> H relative change: {channel['zero_Q']['h_rel_mean']:.4f}",
        f"- permute P -> H relative change: {channel['permute_P']['h_rel_mean']:.4f}",
        f"- permute Q -> H relative change: {channel['permute_Q']['h_rel_mean']:.4f}",
        "",
        "## 用户聚集",
        "",
        f"- val 5-NN same-user rate: {user_diag.get('same_user_neighbor_rate')}",
        f"- chance rate: {user_diag.get('chance_rate')}",
        f"- lift over chance: {user_diag.get('lift_over_chance')}",
        "",
        "## 验收检查",
        "",
    ])
    for name, value in acceptance["core_checks"].items():
        lines.append(f"- {name}: **{value}**")
    if acceptance["warnings"]:
        lines.extend(["", "## 警告", ""])
        lines.extend(f"- {warning}" for warning in acceptance["warnings"])
    lines.extend([
        "",
        "## 使用口径",
        "",
        acceptance["interpretation"],
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-1 comprehensive diagnostics")
    parser.add_argument("--ckpt", default="step_1/best_model_st1.pt")
    parser.add_argument("--data_path", default=None, help="默认使用checkpoint中的data_path")
    parser.add_argument("--split_manifest", default=None, help="默认使用checkpoint中的split_manifest")
    parser.add_argument("--user_col", default=None)
    parser.add_argument("--out_dir", default="diag_step1")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_points_per_split", type=int, default=0, help="0表示使用全部样本")
    parser.add_argument("--max_distance_samples", type=int, default=4096)
    parser.add_argument("--max_ablation_samples", type=int, default=2048)
    parser.add_argument("--max_plot_points", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--knn_k", type=int, default=5)
    parser.add_argument("--skip_curve_probe", action="store_true")
    parser.add_argument(
        "--probe_alphas", nargs="+", type=float,
        default=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
        help="Ridge alpha candidates; tiny near-unregularized values are intentionally excluded.",
    )
    parser.add_argument("--ridge_solver", default="lsqr", choices=["lsqr", "svd", "cholesky"])
    parser.add_argument("--ridge_tol", type=float, default=1e-8)
    parser.add_argument("--ridge_max_iter", type=int, default=10000)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    out_dir = Path(cli_args.out_dir).expanduser().resolve()
    ckpt_hint = Path(cli_args.ckpt).expanduser()
    log_path = setup_logging(out_dir, ckpt_hint)
    start_time = time.perf_counter()
    try:
        device = torch.device("cuda" if cli_args.cuda and torch.cuda.is_available() else "cpu")
        print(f"[Device] {device}")
        set_seed(cli_args.seed)
        ckpt_path, checkpoint, checkpoint_args, data, model = load_checkpoint_and_data(cli_args, device)
        print(f"[Checkpoint] {ckpt_path}")
        print(f"[Checkpoint] epoch={checkpoint.get('epoch')}, objective={checkpoint.get('objective_version')}")
        print(f"[Data] pipeline={data.get('pipeline_version')}")

        integrity = audit_integrity(ckpt_path, checkpoint, data)
        print(f"[Integrity] ok={integrity['integrity_ok']}, user_disjoint={integrity['user_disjoint']}")
        clipping = clipping_diagnostics(data)
        print(f"[Normalization] {clipping['normalization']}")

        x = data["X"][:, :2, :].float()
        sampled_indices = {
            split: sample_indices(
                data["split_indices"][split],
                cli_args.max_points_per_split,
                cli_args.seed + offset,
            )
            for split, offset in (("train", 1), ("val", 2), ("test", 3))
        }
        metadata = {
            split: data["meta"].iloc[indices].reset_index(drop=True)
            for split, indices in sampled_indices.items()
        }
        physical_targets = {
            split: compute_physical_targets(data, indices)
            for split, indices in sampled_indices.items()
        }

        trained_h: Dict[str, np.ndarray] = {}
        trained_z: Dict[str, np.ndarray] = {}
        for split in ("train", "val", "test"):
            trained_h[split], trained_z[split] = extract_embeddings(
                model,
                x,
                sampled_indices[split],
                device,
                cli_args.batch_size,
            )
            print(
                f"[Embedding-{split}] H={trained_h[split].shape}, Z={trained_z[split].shape}"
            )

        spectra: Dict[str, Dict[str, Any]] = {}
        for split in ("train", "val", "test"):
            spectra[split] = {
                "H": spectrum_stats(trained_h[split]),
                "Z": spectrum_stats(trained_z[split]),
            }
            print(
                f"[Spectrum-{split}] H erank={spectra[split]['H']['effective_rank']:.3f}, "
                f"k90={spectra[split]['H']['k90']}, k99={spectra[split]['H']['k99']} | "
                f"Z erank={spectra[split]['Z']['effective_rank']:.3f}"
            )
        plot_spectrum_and_std(spectra, out_dir)

        augment_config = AugmentConfig.from_args(checkpoint_args)
        distance = {}
        for split, offset in (("train", 101), ("val", 102), ("test", 103)):
            distance[split] = distance_diagnostics(
                model,
                x,
                data["split_indices"][split],
                device,
                augment_config,
                cli_args.batch_size,
                cli_args.max_distance_samples,
                cli_args.seed + offset,
                out_dir,
                split,
            )
            print(f"[Distance-{split}] Z ratio={distance[split]['Z']['d2_ratio']:.6f}")

        augmentation = augmentation_sensitivity(
            model,
            x,
            data["split_indices"]["val"],
            device,
            augment_config,
            cli_args.batch_size,
            cli_args.max_ablation_samples,
            cli_args.seed + 210,
        )
        channel = channel_ablation(
            model,
            x,
            data["split_indices"]["val"],
            device,
            cli_args.batch_size,
            cli_args.max_ablation_samples,
            cli_args.seed + 220,
        )
        print(f"[Channel] zero_P H change={channel['zero_P']['h_rel_mean']:.4f}")
        print(f"[Channel] zero_Q H change={channel['zero_Q']['h_rel_mean']:.4f}")

        pca_stats = fit_train_pca_and_plot(
            trained_h,
            metadata,
            physical_targets,
            out_dir,
            cli_args.seed,
            cli_args.max_plot_points,
        )

        user_identity = {
            split: user_identity_diagnostics(
                trained_h[split],
                metadata[split],
                cli_args.knn_k,
                max_samples=6000,
                seed=cli_args.seed + {"train": 301, "val": 302, "test": 303}[split],
            )
            for split in ("train", "val", "test")
        }
        print(f"[User-Val] {user_identity['val']}")

        random_seed = cli_args.seed + 900001
        set_seed(random_seed)
        random_model = instantiate_model(checkpoint_args, device)
        random_model.eval()
        random_h: Dict[str, np.ndarray] = {}
        for split in ("train", "val"):
            random_h[split], _ = extract_embeddings(
                random_model,
                x,
                sampled_indices[split],
                device,
                cli_args.batch_size,
            )

        raw_features = {
            split: x.index_select(0, torch.from_numpy(sampled_indices[split]))
            .numpy()
            .reshape(len(sampled_indices[split]), -1)
            for split in ("train", "val")
        }
        alpha_grid = [float(value) for value in cli_args.probe_alphas]
        physical = physical_probe(
            {split: trained_h[split] for split in ("train", "val")},
            random_h,
            raw_features,
            {split: physical_targets[split] for split in ("train", "val")},
            {split: metadata[split] for split in ("train", "val")},
            alpha_grid,
            cli_args.seed + 1000,
            solver=cli_args.ridge_solver,
            tol=cli_args.ridge_tol,
            max_iter=cli_args.ridge_max_iter,
        )
        write_probe_csv(physical, out_dir / "stage1_physical_probe.csv")
        print(f"[Physical] trained-random composite={physical['trained_minus_random']:+.4f}")

        curve = None
        if not cli_args.skip_curve_probe:
            target_curves = {
                split: raw_features[split] for split in ("train", "val")
            }
            curve = curve_probe(
                {split: trained_h[split] for split in ("train", "val")},
                random_h,
                target_curves,
                {split: metadata[split] for split in ("train", "val")},
                alpha_grid,
                cli_args.seed + 2000,
                solver=cli_args.ridge_solver,
                tol=cli_args.ridge_tol,
                max_iter=cli_args.ridge_max_iter,
            )
            print(f"[Curve] trained-random={curve['trained_minus_random']:+.4f}")

        report: Dict[str, Any] = {
            "version": "stage1-diagnostics-v10.1-ridge-stable",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "runtime_seconds": time.perf_counter() - start_time,
            "checkpoint": str(ckpt_path),
            "checkpoint_args": vars(checkpoint_args),
            "objective_version": checkpoint.get("objective_version", ""),
            "integrity": integrity,
            "clipping": clipping,
            "sampled_indices": {split: indices for split, indices in sampled_indices.items()},
            "spectra": spectra,
            "distance": distance,
            "augmentation_sensitivity": augmentation,
            "channel_ablation": channel,
            "pca": pca_stats,
            "user_identity": user_identity,
            "probe_config": {
                "alphas": alpha_grid,
                "solver": cli_args.ridge_solver,
                "tol": cli_args.ridge_tol,
                "max_iter": cli_args.ridge_max_iter,
                "same_tune_users_for_compared_features": True,
                "dtype": "float64",
            },
            "physical_probe": physical,
            "curve_probe": curve,
        }
        acceptance = acceptance_decision(report)
        report["acceptance"] = acceptance

        # Remove very large arrays from the compact acceptance file.
        compact_spectra = {
            split: {
                rep: {k: v for k, v in values.items() if k != "eigenvalues"}
                for rep, values in split_values.items()
            }
            for split, split_values in spectra.items()
        }
        compact = {
            "checkpoint": str(ckpt_path),
            "checkpoint_epoch": integrity["checkpoint_epoch"],
            "pipeline_version": integrity["pipeline_current"],
            "normalization": clipping["normalization"],
            "spectra": compact_spectra,
            "validation_d2": distance["val"],
            "physical_probe_composite": physical["composite"],
            "physical_trained_minus_random": physical["trained_minus_random"],
            "physical_probe_numerically_stable": physical.get("numerically_stable", False),
            "physical_probe_numerical_warning_count": physical.get("numerical_warning_count", 0),
            "curve_probe_numerically_stable": None if curve is None else curve.get("numerically_stable", False),
            "curve_probe_numerical_warning_count": None if curve is None else curve.get("numerical_warning_count", 0),
            "channel_ablation": channel,
            "user_identity_val": user_identity["val"],
            **acceptance,
        }

        write_json(out_dir / "stage1_diagnostics_full.json", report)
        write_json(out_dir / "best_model_st1_acceptance.json", compact)
        write_json(out_dir / "stage1_physical_probe_stable.json", physical)
        if curve is not None:
            write_json(out_dir / "stage1_curve_probe_stable.json", curve)
        summary_markdown = render_summary_markdown(report, acceptance)
        (out_dir / "stage1_diagnostics_summary.md").write_text(summary_markdown, encoding="utf-8")
        stability_markdown = render_probe_stability_summary(
            physical=physical,
            curve=curve,
            checkpoint_path=ckpt_path,
            checkpoint_epoch=integrity["checkpoint_epoch"],
            solver=cli_args.ridge_solver,
            alphas=alpha_grid,
        )
        (out_dir / "stage1_probe_stability_summary.md").write_text(
            stability_markdown, encoding="utf-8"
        )

        print("[Output]", out_dir / "stage1_diagnostics_full.json")
        print("[Output]", out_dir / "best_model_st1_acceptance.json")
        print("[Output]", out_dir / "stage1_diagnostics_summary.md")
        print("[Output]", out_dir / "stage1_physical_probe.csv")
        print("[Output]", out_dir / "stage1_physical_probe_stable.json")
        if curve is not None:
            print("[Output]", out_dir / "stage1_curve_probe_stable.json")
        print("[Output]", out_dir / "stage1_probe_stability_summary.md")
        print(f"[Probe] numerically_stable={physical.get('numerically_stable', False)}, "
              f"warning_count={physical.get('numerical_warning_count', 0)}")
        print(f"[Acceptance] accepted={acceptance['accepted']}")
        for warning in acceptance["warnings"]:
            print(f"[Warning] {warning}")
    finally:
        close_logging()


if __name__ == "__main__":
    main()
