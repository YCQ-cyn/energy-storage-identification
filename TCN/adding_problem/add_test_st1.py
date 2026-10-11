import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "2"
import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Sampler, TensorDataset
matplotlib.use("Agg")
import matplotlib.pyplot as plt
PROJECT_PARENT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SAVE_DIR = SCRIPT_DIR / "step_1"
STAGE1_OBJECTIVE_VERSION = "joint-z-vicreg-h-vcreg-v8-batched-augment-equivalent"
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))
from TCN.adding_problem.model import (
    DEFAULT_TCN_LEVELS,
    TCN_LSTM,
    TCN_LSTM_ARCH_VERSION,
)
from TCN.adding_problem.utils import (
    augment_sync_intensity,
    augment_sync_intensity_torch,
    load_storage_data,
)


GLOBAL_LOG_FILE = None


@dataclass(frozen=True)
class AugmentConfig:
    scale_min: float = 0.9
    scale_max: float = 1.1
    smooth_prob: float = 0.2
    smooth_window: int = 3
    noise_std: float = 0.01

    @classmethod
    def from_args(cls, args):
        return cls(
            scale_min=float(getattr(args, "aug_scale_min", 0.9)),
            scale_max=float(getattr(args, "aug_scale_max", 1.1)),
            smooth_prob=float(getattr(args, "aug_smooth_prob", 0.2)),
            smooth_window=int(getattr(args, "aug_smooth_window", 3)),
            noise_std=float(getattr(args, "aug_noise_std", 0.01)),
        )


def set_seed(seed=1111, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)


class ConsoleLogger:
    def __init__(self, stream):
        self.stream = stream

    def write(self, message):
        self.stream.write(message)
        if GLOBAL_LOG_FILE is not None:
            GLOBAL_LOG_FILE.write(message)
            if message.endswith("\n"):
                GLOBAL_LOG_FILE.flush()

    def flush(self):
        self.stream.flush()
        if GLOBAL_LOG_FILE is not None:
            GLOBAL_LOG_FILE.flush()


def _spectrum_stats(embeddings):
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    covariance = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)[::-1]
    total = float(eigenvalues.sum())
    if not np.isfinite(total) or total <= 1e-12:
        return 0, 0, float("nan"), 0.0

    cumulative = np.cumsum(eigenvalues) / total
    probabilities = np.clip(eigenvalues / total, 1e-12, 1.0)
    effective_rank = float(np.exp(-(probabilities * np.log(probabilities)).sum()))
    return (
        int(np.searchsorted(cumulative, 0.90) + 1),
        int(np.searchsorted(cumulative, 0.99) + 1),
        float(eigenvalues[0] / total),
        effective_rank,
    )


@torch.no_grad()
def rank_probe(model, inputs, device, n=7168, seed=1111, batch_size=512):
    was_training = model.training
    model.eval()
    sample_count = min(n, len(inputs))
    if sample_count == 0:
        raise ValueError("秩诊断没有可用样本")
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(inputs), size=sample_count, replace=False)
    index_tensor = torch.from_numpy(indices).long()
    selected = inputs.index_select(0, index_tensor)
    h_parts = []
    z_parts = []
    for start in range(0, sample_count, int(batch_size)):
        batch = selected[start:start + int(batch_size)].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        h, z = model(batch)
        h_parts.append(h.float().cpu())
        z_parts.append(z.float().cpu())
    if was_training:
        model.train()
    h_values = torch.cat(h_parts, dim=0).numpy()
    z_values = torch.cat(z_parts, dim=0).numpy()
    return _spectrum_stats(h_values), _spectrum_stats(z_values)


def embedding_reg_loss(h1, h2, target_std=1.5e-1,
                       inv_weight=1.0, var_weight=1.0, cov_weight=1.0):
    if target_std <= 0:
        raise ValueError("target_std 必须大于 0")

    h1 = torch.nan_to_num(h1, nan=0.0, posinf=5.0, neginf=-5.0)
    h2 = torch.nan_to_num(h2, nan=0.0, posinf=5.0, neginf=-5.0)
    eps = 1e-4

    def view_regularization(embedding):
        centered = embedding - embedding.mean(dim=0, keepdim=True)
        std = torch.sqrt(torch.var(embedding, dim=0, unbiased=False) + eps)
        var_loss = F.relu(1.0 - std / target_std).mean()

        normalized = F.normalize(centered, p=2, dim=0, eps=eps)
        correlation = normalized.t() @ normalized
        correlation = torch.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
        off_diagonal = correlation - torch.diag(torch.diag(correlation))
        cov_loss = off_diagonal.square().sum() / (embedding.size(1) ** 2)
        return var_loss, cov_loss

    inv_loss = F.mse_loss(h1, h2) / (target_std ** 2)
    var1, cov1 = view_regularization(h1)
    var2, cov2 = view_regularization(h2)
    var_loss = 0.5 * (var1 + var2)
    cov_loss = 0.5 * (cov1 + cov2)

    all_h = torch.cat([h1, h2], dim=0)
    std = torch.sqrt(torch.var(all_h, dim=0, unbiased=False) + eps)
    total = inv_weight * inv_loss + var_weight * var_loss + cov_weight * cov_loss
    return total, inv_loss, var_loss, cov_loss, std.min(), std.mean()


def split_pq_from_x(x):
    if x.dim() == 2:
        x = x.unsqueeze(0)
    return x[:, 0:1, :], x[:, 1:2, :]


def apply_sync_intensity_to_x(x, augment_config=None, rng=None):
    augment_config = augment_config or AugmentConfig()
    single_sample = x.dim() == 2
    if single_sample:
        x = x.unsqueeze(0)

    output = []
    for sample in x:
        p_aug, q_aug = augment_sync_intensity(
            sample[0].detach().cpu().numpy(),
            sample[1].detach().cpu().numpy(),
            scale_min=augment_config.scale_min,
            scale_max=augment_config.scale_max,
            smooth_prob=augment_config.smooth_prob,
            smooth_window=augment_config.smooth_window,
            noise_std=augment_config.noise_std,
            rng=rng,
        )
        output.append(torch.stack([
            torch.from_numpy(p_aug),
            torch.from_numpy(q_aug),
        ]))

    augmented = torch.stack(output).to(device=x.device, dtype=x.dtype)
    return augmented[0] if single_sample else augmented


class EncoderWithHead(nn.Module):
    def __init__(self, input_channels, nhid, levels, ksize, dropout, lstm_hidden,
                 lstm_layers, embedding_dim, proj_dim=128):
        super().__init__()
        self.backbone = TCN_LSTM(
            input_channels=input_channels,
            num_channels=[nhid] * levels,
            kernel_size=ksize,
            dropout=dropout,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            embedding_dim=embedding_dim,
            use_weight_norm=False,
        )
        self.proj_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, proj_dim),
        )
        for module in self.proj_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        h = torch.clamp(self.backbone(x), -5.0, 5.0)
        return h, self.proj_head(h)

class VICRegLoss(nn.Module):
    def __init__(self, lambda_vicreg=1.0, lambda_reg=1.0, target_var=1.5e-1,
                 vicreg_inv=10.0, vicreg_var=10.0, vicreg_cov=1.0,
                 h_inv_weight=0.0, h_var_weight=1.0, h_cov_weight=20.0):
        super().__init__()
        self.lambda_vicreg = lambda_vicreg
        self.lambda_reg = lambda_reg
        self.target_var = target_var
        self.vicreg_inv = vicreg_inv
        self.vicreg_var = vicreg_var
        self.vicreg_cov = vicreg_cov
        self.h_inv_weight = h_inv_weight
        self.h_var_weight = h_var_weight
        self.h_cov_weight = h_cov_weight

    @staticmethod
    def _off_diagonal(matrix):
        size = matrix.shape[0]
        return matrix.flatten()[1:].view(size - 1, size + 1)[:, :-1]

    def _vicreg(self, z1, z2):
        z1 = torch.clamp(torch.nan_to_num(z1, nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)
        z2 = torch.clamp(torch.nan_to_num(z2, nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)
        inv_loss = F.mse_loss(z1, z2)

        std_z1 = torch.sqrt(torch.var(z1, dim=0, unbiased=False) + 1e-4)
        std_z2 = torch.sqrt(torch.var(z2, dim=0, unbiased=False) + 1e-4)
        var_loss = F.relu(1.0 - std_z1).mean() + F.relu(1.0 - std_z2).mean()

        batch_size, dimension = z1.shape
        if batch_size > 1:
            centered1 = z1 - z1.mean(dim=0, keepdim=True)
            centered2 = z2 - z2.mean(dim=0, keepdim=True)
            covariance1 = centered1.t() @ centered1 / (batch_size - 1)
            covariance2 = centered2.t() @ centered2 / (batch_size - 1)
            cov_loss = (
                self._off_diagonal(covariance1).square().sum()
                + self._off_diagonal(covariance2).square().sum()
            ) / dimension
        else:
            cov_loss = z1.new_tensor(0.0)

        total = self.vicreg_inv * inv_loss + self.vicreg_var * var_loss + self.vicreg_cov * cov_loss
        return total, inv_loss, var_loss, cov_loss

    def forward(self, h1, h2, z1, z2):
        vicreg_loss, inv_loss, var_loss, cov_loss = self._vicreg(z1, z2)
        if self.lambda_reg > 0:
            reg_loss, h_inv, h_var, h_cov, h_std_min, h_std_mean = embedding_reg_loss(
                h1,
                h2,
                target_std=self.target_var,
                inv_weight=self.h_inv_weight,
                var_weight=self.h_var_weight,
                cov_weight=self.h_cov_weight,
            )
        else:
            reg_loss = z1.new_tensor(0.0)
            h_inv = h_var = h_cov = h_std_min = h_std_mean = z1.new_tensor(0.0)

        loss = (
            self.lambda_vicreg * vicreg_loss
            + self.lambda_reg * reg_loss
        )
        return loss, {
            "L_vicreg": vicreg_loss.detach(),
            "L_reg": reg_loss.detach(),
            "H_inv": h_inv.detach(),
            "H_var": h_var.detach(),
            "H_cov": h_cov.detach(),
            "H_std_min": h_std_min.detach(),
            "H_std_mean": h_std_mean.detach(),
            "vic_inv": inv_loss.detach(),
            "vic_var": var_loss.detach(),
            "vic_cov": cov_loss.detach(),
        }


def build_fixed_validation_views(inputs, augment_config, seed):
    view1 = torch.empty_like(inputs)
    view2 = torch.empty_like(inputs)
    for index, sample in enumerate(inputs):
        sample_seed = int(seed) + index * 1009
        view1[index] = apply_sync_intensity_to_x(
            sample,
            augment_config,
            rng=np.random.RandomState(sample_seed),
        )
        view2[index] = apply_sync_intensity_to_x(
            sample,
            augment_config,
            rng=np.random.RandomState(sample_seed + 1),
        )
    return view1, view2


class UserBalancedBatchSampler(Sampler):
    def __init__(self, user_ids, sample_weights, batch_size, seed=1111,
                 user_sampling_power=0.5, drop_last=True):
        self.user_ids = np.asarray(user_ids).astype(str)
        self.sample_weights = np.asarray(sample_weights, dtype=np.float64)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.user_sampling_power = float(user_sampling_power)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        if len(self.user_ids) != len(self.sample_weights):
            raise ValueError("用户编号与样本权重长度不一致")
        self.users = np.unique(self.user_ids)
        if self.batch_size > len(self.users):
            raise ValueError(
                f"用户均衡批次要求 batch_size 不超过训练用户数，"
                f"当前为 {self.batch_size}>{len(self.users)}"
            )
        if not 0.0 <= self.user_sampling_power <= 1.0:
            raise ValueError("user_sampling_power 必须在 0 到 1 之间")

        self.user_indices = {}
        self.user_probabilities = {}
        user_sample_counts = []
        for user_id in self.users:
            indices = np.flatnonzero(self.user_ids == user_id)
            weights = np.maximum(self.sample_weights[indices], 0.0)
            probabilities = weights / weights.sum() if weights.sum() > 0 else None
            self.user_indices[user_id] = indices
            self.user_probabilities[user_id] = probabilities
            user_sample_counts.append(len(indices))
        user_weights = np.power(np.asarray(user_sample_counts, dtype=np.float64), self.user_sampling_power)
        self.user_sampling_probabilities = user_weights / user_weights.sum()

    def __len__(self):
        if self.drop_last:
            return len(self.user_ids) // self.batch_size
        return math.ceil(len(self.user_ids) / self.batch_size)

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(len(self)):
            selected_users = rng.choice(
                self.users,
                size=self.batch_size,
                replace=False,
                p=self.user_sampling_probabilities,
            )
            batch = [
                int(rng.choice(
                    self.user_indices[user_id],
                    p=self.user_probabilities[user_id],
                ))
                for user_id in selected_users
            ]
            yield batch


class FixedPermutationBatchSampler(Sampler):
    def __init__(self, sample_count, batch_size, seed=1111):
        self.sample_count = int(sample_count)
        self.batch_size = int(batch_size)
        self.indices = np.random.RandomState(seed).permutation(self.sample_count).tolist()

    def __len__(self):
        return math.ceil(self.sample_count / self.batch_size)

    def __iter__(self):
        for start in range(0, self.sample_count, self.batch_size):
            yield self.indices[start:start + self.batch_size]


def build_dataloaders_st1(args, device=None, return_validation=False):
    split_manifest = getattr(args, "split_manifest", None)
    if not split_manifest:
        raise ValueError("Stage-1 必须提供固定的 train/val/test 用户切分清单")
    data = load_storage_data(
        data_path=getattr(args, "data_path", "processed_result.xlsx"),
        time_type=getattr(args, "time_type", None),
        seed=args.seed,
        split_manifest_path=split_manifest,
        user_col=getattr(args, "user_col", "关联用户号"),
        policy_name=getattr(args, "policy_name", "old"),
    )
    if data["split_strategy"] != "user-manifest":
        raise RuntimeError("Stage-1 未使用固定用户切分清单")
    train_indices = data["split_indices"]["train"]
    val_indices = data["split_indices"]["val"]
    train_index_tensor = torch.from_numpy(train_indices)
    val_index_tensor = torch.from_numpy(val_indices)
    x_train = data["X"].index_select(0, train_index_tensor)[:, :2, :]
    x_val = data["X"].index_select(0, val_index_tensor)[:, :2, :]
    augment_config = AugmentConfig.from_args(args)

    val_view1, val_view2 = build_fixed_validation_views(
        x_val,
        augment_config,
        seed=int(getattr(args, "val_seed", args.seed + 100000)),
    )
    train_set = TensorDataset(x_train)
    val_set = TensorDataset(val_view1, val_view2)

    loader_generator = torch.Generator().manual_seed(args.seed)
    num_workers = int(getattr(args, "num_workers", 0))
    loader_options = {
        "num_workers": num_workers,
        "pin_memory": bool(device is not None and device.type == "cuda"),
    }
    if num_workers > 0:
        loader_options.update({"persistent_workers": True, "prefetch_factor": 2})
    train_meta = data["meta"].iloc[train_indices]
    if getattr(args, "user_balanced_sampling", False):
        batch_sampler = UserBalancedBatchSampler(
            user_ids=train_meta["user_id"].to_numpy(),
            sample_weights=train_meta["duplicate_weight"].to_numpy(dtype=np.float64),
            batch_size=args.batch_size,
            seed=args.seed,
            user_sampling_power=float(getattr(args, "user_sampling_power", 0.5)),
        )
        train_loader = DataLoader(
            train_set,
            batch_sampler=batch_sampler,
            generator=loader_generator,
            **loader_options,
        )
        sampling_strategy = "distinct-user-balanced-batches"
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=len(train_set) >= args.batch_size,
            generator=loader_generator,
            **loader_options,
        )
        sampling_strategy = "shuffled-samples"
    val_loader = DataLoader(
        val_set,
        batch_sampler=FixedPermutationBatchSampler(
            len(val_set), args.batch_size, seed=int(getattr(args, "val_seed", args.seed + 100000))
        ),
        **loader_options,
    )
    args.input_channels = 2
    data_info = {
        "pipeline_version": data["pipeline_version"],
        "data_path": data["data_path"],
        "data_sha256": data["data_sha256"],
        "split_manifest_path": data["split_manifest_path"],
        "split_manifest_sha256": data["split_manifest_sha256"],
        "split_strategy": data["split_strategy"],
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "reserved_test_samples": len(data["split_indices"]["test"]),
        "train_users": int(data["meta"].iloc[train_indices]["user_id"].nunique()),
        "val_users": int(data["meta"].iloc[val_indices]["user_id"].nunique()),
        "test_users": int(
            data["meta"].iloc[data["split_indices"]["test"]]["user_id"].nunique()
        ),
        "user_balanced_sampling": bool(getattr(args, "user_balanced_sampling", False)),
        "sampling_strategy": sampling_strategy,
        "user_sampling_power": float(getattr(args, "user_sampling_power", 0.5)),
        "validation_batching": "fixed-shuffled-batches",
        "augmentation_backend": "torch-batched-v1-equivalent-distribution",
        "normalization": data["normalization"],
    }
    print(
        f"[数据] train={len(x_train)}, val={len(x_val)}, "
        f"reserved_test={data_info['reserved_test_samples']} | 输入={tuple(x_train.shape[1:])}"
    )
    print(f"[数据] pipeline={data_info['pipeline_version']} | 增强={asdict(augment_config)}")
    print(
        f"[用户] train={data_info['train_users']}, val={data_info['val_users']}, "
        f"test={data_info['test_users']} | 采样={sampling_strategy} | "
        f"用户采样指数={data_info['user_sampling_power']:.2f}"
    )

    if return_validation:
        return train_loader, val_loader, x_train, x_val, data_info
    return train_loader, x_train


def build_model_and_optimizer(args, device):
    model = EncoderWithHead(
        input_channels=2,
        nhid=args.nhid,
        levels=args.levels,
        ksize=args.ksize,
        dropout=args.dropout,
        lstm_hidden=getattr(args, "lstm_hidden", args.nhid),
        lstm_layers=getattr(args, "lstm_layers", 1),
        embedding_dim=getattr(args, "embedding_dim", args.nhid),
        proj_dim=getattr(args, "proj_dim", 128),
    ).to(device)
    criterion = VICRegLoss(
        lambda_vicreg=1.0,
        lambda_reg=args.lambda_reg,
        target_var=getattr(args, "target_var", 1.5e-1),
        vicreg_inv=getattr(args, "vicreg_inv", 10.0),
        vicreg_var=getattr(args, "vicreg_var", 10.0),
        vicreg_cov=getattr(args, "vicreg_cov", 1.0),
        h_inv_weight=getattr(args, "h_inv_weight", 0.0),
        h_var_weight=getattr(args, "h_var_weight", 1.0),
        h_cov_weight=getattr(args, "h_cov_weight", 20.0),
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=getattr(args, "lr_patience", 2),
    )
    return model, criterion, optimizer, scheduler


def _empty_metrics():
    return {
        "loss": 0.0,
        "vicreg": 0.0,
        "reg": 0.0,
        "vic_inv": 0.0,
        "vic_var": 0.0,
        "vic_cov": 0.0,
        "h_inv": 0.0,
        "h_var": 0.0,
        "h_cov": 0.0,
        "h_std_min": 0.0,
        "h_std_mean": 0.0,
        "h_norm_mean": 0.0,
        "z_std_min": 0.0,
        "z_std_mean": 0.0,
        "d2_pos": 0.0,
        "d2_neg": 0.0,
        "samples": 0,
        "batches": 0,
    }


def _update_metrics(metrics, loss, log_values, h1, h2, z1, z2):
    z1_safe = torch.nan_to_num(z1.detach(), nan=0.0, posinf=1e4, neginf=-1e4)
    z2_safe = torch.nan_to_num(z2.detach(), nan=0.0, posinf=1e4, neginf=-1e4)
    z_all = torch.cat([z1_safe, z2_safe], dim=0)
    z_std = torch.sqrt(torch.var(z_all, dim=0, unbiased=False) + 1e-4)
    batch_size = z2_safe.size(0)
    shift = max(batch_size // 2, 1)
    negative_z2 = torch.roll(z2_safe, shifts=shift, dims=0)

    metrics["loss"] += loss.detach() * batch_size
    metrics["vicreg"] += log_values["L_vicreg"] * batch_size
    metrics["reg"] += log_values["L_reg"] * batch_size
    metrics["vic_inv"] += log_values["vic_inv"] * batch_size
    metrics["vic_var"] += log_values["vic_var"] * batch_size
    metrics["vic_cov"] += log_values["vic_cov"] * batch_size
    metrics["h_inv"] += log_values["H_inv"] * batch_size
    metrics["h_var"] += log_values["H_var"] * batch_size
    metrics["h_cov"] += log_values["H_cov"] * batch_size
    metrics["h_std_min"] += log_values["H_std_min"] * batch_size
    metrics["h_std_mean"] += log_values["H_std_mean"] * batch_size
    metrics["h_norm_mean"] += 0.5 * batch_size * (
        torch.norm(h1, dim=1).mean() + torch.norm(h2, dim=1).mean()
    )
    metrics["z_std_min"] += z_std.min() * batch_size
    metrics["z_std_mean"] += z_std.mean() * batch_size
    metrics["d2_pos"] += (
        torch.sum((z1_safe - z2_safe).square(), dim=1).mean() * batch_size
    )
    metrics["d2_neg"] += (
        torch.sum((z1_safe - negative_z2).square(), dim=1).mean() * batch_size
    )
    metrics["samples"] += batch_size
    metrics["batches"] += 1


def _finalize_metrics(metrics):
    metrics.pop("batches")
    samples = metrics.pop("samples")
    if samples == 0:
        raise RuntimeError("没有可用训练批次")
    names = list(metrics)
    values = torch.stack([metrics[name].detach() for name in names])
    averages = (values / samples).cpu().tolist()
    result = dict(zip(names, averages))
    result["d2_ratio"] = result["d2_pos"] / (result["d2_neg"] + 1e-8)
    return result


def _gradient_norm(model):
    gradients = [
        parameter.grad.detach().norm(2)
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        return next(model.parameters()).new_tensor(0.0)
    return torch.stack(gradients).norm(2)


def _encode_views(model, view1, view2, device, amp_enabled):
    combined = torch.cat([view1, view2], dim=0)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=amp_enabled,
    ):
        embedding, projection = model(combined)
    embedding1, embedding2 = embedding.float().chunk(2, dim=0)
    projection1, projection2 = projection.float().chunk(2, dim=0)
    return embedding1, embedding2, projection1, projection2


def train_one_epoch(model, criterion, loader, optimizer, device, clip,
                    augment_config, scaler, amp_enabled):
    model.train()
    metrics = _empty_metrics()
    gradient_norm_sum = torch.zeros((), device=device)
    data_seconds = 0.0
    compute_seconds = 0.0
    iteration_end = time.perf_counter()
    for (clean_inputs,) in loader:
        data_seconds += time.perf_counter() - iteration_end
        compute_start = time.perf_counter()
        clean_inputs = clean_inputs.to(device, non_blocking=True)
        view1 = augment_sync_intensity_torch(
            clean_inputs,
            scale_min=augment_config.scale_min,
            scale_max=augment_config.scale_max,
            smooth_prob=augment_config.smooth_prob,
            smooth_window=augment_config.smooth_window,
            noise_std=augment_config.noise_std,
        )
        view2 = augment_sync_intensity_torch(
            clean_inputs,
            scale_min=augment_config.scale_min,
            scale_max=augment_config.scale_max,
            smooth_prob=augment_config.smooth_prob,
            smooth_window=augment_config.smooth_window,
            noise_std=augment_config.noise_std,
        )
        optimizer.zero_grad(set_to_none=True)
        h1, h2, z1, z2 = _encode_views(
            model, view1, view2, device, amp_enabled
        )
        loss, log_values = criterion(h1, h2, z1, z2)
        if not torch.isfinite(loss):
            raise RuntimeError("Stage-1 出现非有限损失")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if clip is not None and clip > 0:
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        else:
            gradient_norm = _gradient_norm(model)
        scaler.step(optimizer)
        scaler.update()
        gradient_norm_sum += gradient_norm.detach()
        _update_metrics(metrics, loss.detach(), log_values, h1.detach(), h2.detach(), z1, z2)
        compute_seconds += time.perf_counter() - compute_start
        iteration_end = time.perf_counter()
    batch_count = metrics["batches"]
    result = _finalize_metrics(metrics)
    result["grad_norm"] = float(
        (gradient_norm_sum / max(batch_count, 1)).cpu().item()
    )
    result["data_seconds"] = data_seconds
    result["compute_seconds"] = compute_seconds
    return result


@torch.no_grad()
def evaluate(model, criterion, loader, device, amp_enabled):
    model.eval()
    metrics = _empty_metrics()
    data_seconds = 0.0
    compute_seconds = 0.0
    iteration_end = time.perf_counter()
    for x1, x2 in loader:
        data_seconds += time.perf_counter() - iteration_end
        if len(x1) < 2:
            continue
        compute_start = time.perf_counter()
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        h1, h2, z1, z2 = _encode_views(model, x1, x2, device, amp_enabled)
        loss, log_values = criterion(h1, h2, z1, z2)
        _update_metrics(metrics, loss, log_values, h1, h2, z1, z2)
        compute_seconds += time.perf_counter() - compute_start
        iteration_end = time.perf_counter()
    result = _finalize_metrics(metrics)
    result["data_seconds"] = data_seconds
    result["compute_seconds"] = compute_seconds
    return result


def _write_history_csv(history, save_dir):
    path = Path(save_dir) / "phase1_training_history.csv"
    if not history:
        return path
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    return path


def _plot_training_history(history, save_dir):
    epochs = np.asarray([row["epoch"] for row in history])
    plt.figure()
    plt.plot(epochs, [row["train_loss"] for row in history], label="train")
    plt.plot(epochs, [row["val_loss"] for row in history], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("VICReg loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "phase1_loss_curves.png"), dpi=200)
    plt.close()

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(epochs, [row["val_vic_inv"] for row in history], label="inv")
    axes[0, 0].plot(epochs, [row["val_vic_var"] for row in history], label="var")
    axes[0, 0].plot(epochs, [row["val_vic_cov"] for row in history], label="cov")
    axes[0, 0].set_title("Validation VICReg components")
    axes[0, 0].legend()

    for name in ("val_h_std_min", "val_h_std_mean", "val_z_std_min", "val_z_std_mean"):
        axes[0, 1].plot(epochs, [row[name] for row in history], label=name.removeprefix("val_"))
    axes[0, 1].set_title("Collapse diagnostics")
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, [row["train_d2_ratio"] for row in history], label="train")
    axes[1, 0].plot(epochs, [row["val_d2_ratio"] for row in history], label="val")
    axes[1, 0].set_title("Positive / negative distance ratio")
    axes[1, 0].legend()

    rank_rows = [row for row in history if row["val_h_erank"] is not None]
    if rank_rows:
        rank_epochs = [row["epoch"] for row in rank_rows]
        axes[1, 1].plot(rank_epochs, [row["val_h_erank"] for row in rank_rows], label="H")
        axes[1, 1].plot(rank_epochs, [row["val_z_erank"] for row in rank_rows], label="Z")
        axes[1, 1].legend()
    axes[1, 1].set_title("Validation effective rank")

    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.grid(True)
    figure.tight_layout()
    figure.savefig(Path(save_dir) / "phase1_monitoring.png", dpi=200)
    plt.close(figure)


def _atomic_torch_save(payload, path):
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def _transfer_state_dict(model):
    return model.state_dict()


def _write_run_summary(save_dir, payload):
    path = Path(save_dir) / "phase1_run_summary.json"
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    return path


@torch.no_grad()
def _plot_embeddings(model, inputs, augment_config, device, save_dir, seed=1111):
    model.eval()
    sample_count = min(50000, len(inputs))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(inputs), generator=generator)[:sample_count]
    samples = inputs.index_select(0, indices)
    rng1 = np.random.RandomState(seed + 1)
    rng2 = np.random.RandomState(seed + 2)

    h_list = []
    d2_list = []
    for start in range(0, sample_count, 256):
        batch = samples[start:start + 256].to(device)
        h, _ = model(batch)
        view1 = apply_sync_intensity_to_x(batch.clone(), augment_config, rng=rng1)
        view2 = apply_sync_intensity_to_x(batch.clone(), augment_config, rng=rng2)
        _, z1 = model(view1)
        _, z2 = model(view2)
        h_list.append(h.cpu().numpy())
        d2_list.append(torch.sum((z1 - z2).square(), dim=1).cpu().numpy())

    embeddings = np.concatenate(h_list, axis=0)
    distances = np.concatenate(d2_list, axis=0)
    projection = PCA(n_components=2).fit_transform(embeddings)

    plt.figure()
    plt.scatter(projection[:, 0], projection[:, 1], s=8, alpha=0.6)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "phase1_embedding_pca.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.hist(distances, bins=60, density=True)
    plt.xlabel("||z1-z2||²")
    plt.ylabel("Density")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "phase1_z_distance_sq_hist.png"), dpi=200)
    plt.close()


def train_stage1(args, device):
    set_seed(args.seed, deterministic=args.deterministic)
    print("=== Stage-1：P/Q 非对比式时序预训练 ===")
    print("Args:", args)

    train_loader, val_loader, x_train, x_val, data_info = build_dataloaders_st1(
        args, device, return_validation=True
    )
    model, criterion, optimizer, scheduler = build_model_and_optimizer(args, device)
    augment_config = AugmentConfig.from_args(args)
    amp_enabled = bool(getattr(args, "amp", False) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / args.save_name
    print(f"[输出] checkpoint={save_path} | 已存在则覆盖={save_path.exists()}")
    print(f"[目标] {STAGE1_OBJECTIVE_VERSION}")

    best_loss = float("inf")
    best_epoch = -1
    best_train_metrics = None
    best_val_metrics = None
    best_rank_metrics = None
    no_improve_epochs = 0
    history = []
    print(
        f"[Performance] amp={amp_enabled} | deterministic={args.deterministic} | "
        f"rank_batch={args.rank_probe_batch_size}"
    )

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_metrics = train_one_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            device,
            args.clip,
            augment_config,
            scaler,
            amp_enabled,
        )
        val_metrics = evaluate(model, criterion, val_loader, device, amp_enabled)
        monitor_loss = val_metrics["loss"]
        if not math.isfinite(monitor_loss):
            raise RuntimeError("Stage-1 验证损失无效")

        scheduler.step(monitor_loss)
        learning_rate = optimizer.param_groups[0]["lr"]

        print(
            f"[Epoch {epoch}] train={train_metrics['loss']:.5f} | "
            f"val={val_metrics['loss']:.5f} | lr={learning_rate:.3e}"
        )
        print(
            f"[Train] inv={train_metrics['vic_inv']:.5f} | var={train_metrics['vic_var']:.5f} | "
            f"cov={train_metrics['vic_cov']:.5f} | H_std={train_metrics['h_std_mean']:.4f} | "
            f"Z_std={train_metrics['z_std_mean']:.4f} | d2_ratio={train_metrics['d2_ratio']:.4f} | "
            f"grad_norm={train_metrics['grad_norm']:.4f}"
        )
        print(
            f"[Val] inv={val_metrics['vic_inv']:.5f} | var={val_metrics['vic_var']:.5f} | "
            f"cov={val_metrics['vic_cov']:.5f} | H_std={val_metrics['h_std_mean']:.4f} | "
            f"Z_std={val_metrics['z_std_mean']:.4f} | d2_ratio={val_metrics['d2_ratio']:.4f}"
        )
        print(
            f"[H-Reg] train(inv={train_metrics['h_inv']:.4f}, var={train_metrics['h_var']:.4f}, "
            f"cov={train_metrics['h_cov']:.4f}) | "
            f"val(inv={val_metrics['h_inv']:.4f}, var={val_metrics['h_var']:.4f}, "
            f"cov={val_metrics['h_cov']:.4f})"
        )
        train_rank = None
        val_rank = None
        is_candidate = monitor_loss < best_loss - args.best_min_delta
        interval = int(getattr(args, "rank_probe_interval", 3))
        if interval > 0 and (epoch == 1 or epoch % interval == 0):
            train_rank = rank_probe(
                model,
                x_train,
                device,
                seed=args.seed + 17,
                batch_size=args.rank_probe_batch_size,
            )
            val_rank = rank_probe(
                model,
                x_val,
                device,
                seed=args.val_seed + 29,
                batch_size=args.rank_probe_batch_size,
            )
            print(f"[Rank-Train] H={train_rank[0]} | Z={train_rank[1]}")
            print(f"[Rank-Val] H={val_rank[0]} | Z={val_rank[1]}")
        elif is_candidate:
            val_rank = rank_probe(
                model,
                x_val,
                device,
                seed=args.val_seed + 29,
                batch_size=args.rank_probe_batch_size,
            )
            print(f"[Rank-Val-BestCandidate] H={val_rank[0]} | Z={val_rank[1]}")

        history_row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "epoch_seconds": time.perf_counter() - epoch_start,
            "train_data_seconds": train_metrics["data_seconds"],
            "train_compute_seconds": train_metrics["compute_seconds"],
            "val_data_seconds": val_metrics["data_seconds"],
            "val_compute_seconds": val_metrics["compute_seconds"],
            "gpu_peak_memory_mb": (
                torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                if device.type == "cuda"
                else 0.0
            ),
        }
        metric_names = (
            "loss", "vicreg", "reg",
            "vic_inv", "vic_var", "vic_cov",
            "h_inv", "h_var", "h_cov", "h_std_min", "h_std_mean", "h_norm_mean",
            "z_std_min", "z_std_mean", "d2_pos", "d2_neg", "d2_ratio",
        )
        for split_name, metrics in (("train", train_metrics), ("val", val_metrics)):
            for metric_name in metric_names:
                history_row[f"{split_name}_{metric_name}"] = metrics[metric_name]
        history_row["train_grad_norm"] = train_metrics["grad_norm"]
        for split_name, rank_values in (("train", train_rank), ("val", val_rank)):
            for representation_index, representation_name in enumerate(("h", "z")):
                stats = rank_values[representation_index] if rank_values is not None else None
                for stat_index, stat_name in enumerate(("k90", "k99", "eig1_ratio", "erank")):
                    history_row[f"{split_name}_{representation_name}_{stat_name}"] = (
                        stats[stat_index] if stats is not None else None
                    )
        history.append(history_row)
        _write_history_csv(history, save_dir)
        print(
            f"[Timing] train(data={train_metrics['data_seconds']:.1f}s, "
            f"compute={train_metrics['compute_seconds']:.1f}s) | "
            f"val(data={val_metrics['data_seconds']:.1f}s, "
            f"compute={val_metrics['compute_seconds']:.1f}s) | "
            f"peak={history_row['gpu_peak_memory_mb']:.0f}MB"
        )

        if is_candidate:
            best_loss = monitor_loss
            best_epoch = epoch
            best_train_metrics = dict(train_metrics)
            best_val_metrics = dict(val_metrics)
            best_rank_metrics = {"train": train_rank, "val": val_rank}
            no_improve_epochs = 0
            _atomic_torch_save({
                "model_state_dict": _transfer_state_dict(model),
                "args": vars(args),
                "best_loss": best_loss,
                "best_val_loss": best_loss,
                "train_loss": train_metrics["loss"],
                "epoch": epoch,
                "data_info": data_info,
                "augment_config": asdict(augment_config),
                "objective_version": STAGE1_OBJECTIVE_VERSION,
                "model_arch_version": TCN_LSTM_ARCH_VERSION,
                "train_metrics": best_train_metrics,
                "val_metrics": best_val_metrics,
                "rank_metrics": best_rank_metrics,
            }, save_path)
            print(f"保存最佳模型：val_loss={best_loss:.6f} | {save_path}")
        else:
            no_improve_epochs += 1
            print(f"早停计数：{no_improve_epochs}")

        if args.early_stop_patience > 0 and no_improve_epochs >= args.early_stop_patience:
            print(f"验证集连续 {no_improve_epochs} 个 epoch 未改善，停止训练")
            break

    if best_epoch < 0:
        raise RuntimeError("Stage-1 未生成有效 checkpoint")

    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print(f"训练完成：best_val_loss={best_loss:.6f} | epoch={best_epoch}")
    history_path = _write_history_csv(history, save_dir)
    _plot_training_history(history, save_dir)
    _plot_embeddings(model, x_val, augment_config, device, save_dir, seed=args.val_seed)
    summary_path = _write_run_summary(save_dir, {
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "checkpoint": str(save_path),
        "history": str(history_path),
        "args": vars(args),
        "data_info": data_info,
        "augment_config": asdict(augment_config),
        "objective_version": STAGE1_OBJECTIVE_VERSION,
        "model_arch_version": TCN_LSTM_ARCH_VERSION,
        "train_metrics": best_train_metrics,
        "val_metrics": best_val_metrics,
        "rank_metrics": best_rank_metrics,
        "completed_epochs": len(history),
    })
    print(f"[输出] history={history_path}")
    print(f"[输出] summary={summary_path}")


def setup_console_logging(args):
    global GLOBAL_LOG_FILE
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = save_dir / f"{timestamp}_phase1.md"
    GLOBAL_LOG_FILE = log_path.open("w", encoding="utf-8")
    GLOBAL_LOG_FILE.write(
        f"# Stage-1 训练日志\n\n- start_time: {timestamp}\n- save_dir: {args.save_dir}\n\n```text\n"
    )
    GLOBAL_LOG_FILE.flush()
    sys.stdout = ConsoleLogger(sys.stdout)
    sys.stderr = ConsoleLogger(sys.stderr)
    print(f"训练日志：{log_path}")
    return log_path


def parse_args():
    parser = argparse.ArgumentParser(description="Stage-1")
    parser.add_argument("--data_path", type=str, default="processed_result.xlsx")
    parser.add_argument("--split_manifest", type=str, default="data_split/user_split.csv")
    parser.add_argument("--user_col", type=str, default="关联用户号")
    parser.add_argument("--policy_name", type=str, default="old", choices=["old", "new"])
    parser.add_argument("--time_type", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=7e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--nhid", type=int, default=64)
    parser.add_argument("--levels", type=int, default=DEFAULT_TCN_LEVELS)
    parser.add_argument("--ksize", type=int, default=7)
    parser.add_argument("--lstm_hidden", type=int, default=64)
    parser.add_argument("--lstm_layers", type=int, default=1)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--lambda_reg", type=float, default=1.0)
    parser.add_argument("--target_var", type=float, default=0.15)
    parser.add_argument("--h_inv_weight", type=float, default=0.0)
    parser.add_argument("--h_var_weight", type=float, default=1.0)
    parser.add_argument("--h_cov_weight", type=float, default=20.0)
    parser.add_argument("--vicreg_inv", type=float, default=10.0)
    parser.add_argument("--vicreg_var", type=float, default=10.0)
    parser.add_argument("--vicreg_cov", type=float, default=1.0)
    parser.add_argument("--aug_scale_min", type=float, default=0.9)
    parser.add_argument("--aug_scale_max", type=float, default=1.1)
    parser.add_argument("--aug_smooth_prob", type=float, default=0.2)
    parser.add_argument("--aug_smooth_window", type=int, default=3)
    parser.add_argument("--aug_noise_std", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--val_seed", type=int, default=101111)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--lr_patience", type=int, default=3)
    parser.add_argument("--early_stop_patience", type=int, default=15)
    parser.add_argument("--best_min_delta", type=float, default=1e-5)
    parser.add_argument("--rank_probe_interval", type=int, default=3)
    parser.add_argument("--rank_probe_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--user_balanced_sampling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--user_sampling_power", type=float, default=0.5)
    parser.add_argument("--save_dir", type=str, default=str(DEFAULT_SAVE_DIR))
    parser.add_argument("--save_name", type=str, default="best_model_st1.pt")
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.policy_name != "old":
        raise ValueError("Stage-1 真实 P/Q 来自旧政策，新政策迁移不在本阶段处理")
    args.save_dir = str(Path(args.save_dir).expanduser().resolve())
    log_path = setup_console_logging(args)
    if args.cuda and not torch.cuda.is_available():
        print("[设备] 未检测到可用 CUDA，自动使用 CPU")
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"设备：{device} | 日志：{log_path}")
    try:
        train_stage1(args, device)
    finally:
        global GLOBAL_LOG_FILE
        if GLOBAL_LOG_FILE is not None:
            GLOBAL_LOG_FILE.write("\n```\n")
            GLOBAL_LOG_FILE.flush()
            GLOBAL_LOG_FILE.close()
            GLOBAL_LOG_FILE = None


if __name__ == "__main__":
    main()
