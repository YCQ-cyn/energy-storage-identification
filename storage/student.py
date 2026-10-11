# -*- coding: utf-8 -*-
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, roc_auc_score

try:
    from . import teacher as st2
    from .distill import (
        STUDENT_CHANNEL_NAMES,
        STUDENT_DATASET_VERSION,
        STUDENT_TARGET_NAMES,
    )
    from .model import DEFAULT_TEMPORAL_LEVELS, TemporalSequenceModel, MODEL_ARCH_VERSION
    from .targets import TARGET_PIPELINE_VERSION
except ImportError:
    import teacher as st2
    from distill import (
        STUDENT_CHANNEL_NAMES,
        STUDENT_DATASET_VERSION,
        STUDENT_TARGET_NAMES,
    )
    from model import DEFAULT_TEMPORAL_LEVELS, TemporalSequenceModel, MODEL_ARCH_VERSION
    from targets import TARGET_PIPELINE_VERSION


STUDENT_OBJECTIVE_VERSION = "student-generalized-distill-v2-structured"
TARGET_LOSS_WEIGHTS = np.asarray([1.5, 0.7, 0.8, 0.5, 0.5, 0.5], dtype=np.float32)
SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}


class StudentModel(nn.Module):
    def __init__(self, input_channels=5, nhid=64, levels=DEFAULT_TEMPORAL_LEVELS,
                 ksize=7, dropout=0.1, lstm_hidden=64, lstm_layers=1,
                 embedding_dim=64):
        super().__init__()
        self.input_channels = int(input_channels)
        self.backbone = TemporalSequenceModel(
            input_channels=input_channels,
            num_channels=[nhid] * levels,
            kernel_size=ksize,
            dropout=dropout,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            embedding_dim=embedding_dim,
            use_weight_norm=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, len(STUDENT_TARGET_NAMES)),
        )

    def forward(self, inputs):
        embedding = self.backbone(inputs)
        logits = self.head(embedding)
        return {
            "embedding": embedding,
            "logits": logits,
            "scores": torch.sigmoid(logits),
        }


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_path(path):
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).resolve().parent / candidate


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_student_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    dataset_version = arrays.get("dataset_version")
    if (
        dataset_version is None
        or str(dataset_version.item()) != STUDENT_DATASET_VERSION
    ):
        raise RuntimeError("学生数据版本不匹配，请重新运行 distill.py")
    required = {
        "positive_inputs",
        "pseudo_inputs",
        "mismatch_inputs",
        "positive_targets",
        "pseudo_targets",
        "mismatch_targets",
        "split_id",
        "source_type",
        "user_code",
        "loss_weight",
        "sampling_weight",
        "policy_weight",
        "dp_clip_q99",
        "dataset_version",
        "teacher_epoch",
        "teacher_stage2_score",
        "teacher_checkpoint_sha256",
    }
    missing = required - set(arrays)
    if missing:
        raise KeyError(f"学生数据缺少字段: {sorted(missing)}")
    pair_count = len(arrays["split_id"])
    for name in ("positive_inputs", "pseudo_inputs", "mismatch_inputs"):
        if arrays[name].shape != (pair_count, len(STUDENT_CHANNEL_NAMES), 96):
            raise ValueError(f"{name} shape 异常: {arrays[name].shape}")
    for name in ("positive_targets", "pseudo_targets", "mismatch_targets"):
        if arrays[name].shape != (pair_count, len(STUDENT_TARGET_NAMES)):
            raise ValueError(f"{name} shape 异常: {arrays[name].shape}")
    for name, value in arrays.items():
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError(f"{name} 存在 NaN/Inf")
    if set(np.unique(arrays["split_id"])) != {0, 1, 2}:
        raise ValueError("学生数据必须包含 train、val、test")
    return arrays


def validate_dataset_teacher(arrays, checkpoint, checkpoint_path):
    dataset_epoch = int(arrays["teacher_epoch"][0])
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    if dataset_epoch != checkpoint_epoch:
        raise RuntimeError(
            f"学生数据来自教师 epoch={dataset_epoch}，当前教师为 epoch={checkpoint_epoch}；"
            "请重新运行 distill.py"
        )
    dataset_sha256 = str(arrays["teacher_checkpoint_sha256"].item())
    checkpoint_sha256 = file_sha256(checkpoint_path)
    if dataset_sha256 != checkpoint_sha256:
        raise RuntimeError("学生数据与当前 Stage-2 权重不一致，请重新运行 distill.py")
    if checkpoint.get("target_pipeline_version") != TARGET_PIPELINE_VERSION:
        raise RuntimeError("Stage-2 checkpoint 目标版本不匹配")
    if checkpoint.get("stage2_objective_version") != st2.STAGE2_OBJECTIVE_VERSION:
        raise RuntimeError("Stage-2 checkpoint 训练目标版本不匹配")
    if checkpoint.get("model_arch_version") != MODEL_ARCH_VERSION:
        raise RuntimeError("Stage-2 checkpoint 网络结构版本不匹配")


def model_kwargs_from_args(args):
    return {
        "input_channels": len(STUDENT_CHANNEL_NAMES),
        "nhid": args.nhid,
        "levels": args.levels,
        "ksize": args.ksize,
        "dropout": args.dropout,
        "lstm_hidden": args.lstm_hidden,
        "lstm_layers": args.lstm_layers,
        "embedding_dim": args.embedding_dim,
    }


def inherit_teacher_backbone(model, checkpoint):
    if checkpoint.get("target_pipeline_version") != TARGET_PIPELINE_VERSION:
        raise RuntimeError("Stage-2 checkpoint 目标版本不匹配")
    if checkpoint.get("stage2_objective_version") != st2.STAGE2_OBJECTIVE_VERSION:
        raise RuntimeError("Stage-2 checkpoint 训练目标版本不匹配")
    if checkpoint.get("model_arch_version") != MODEL_ARCH_VERSION:
        raise RuntimeError("Stage-2 checkpoint 网络结构版本不匹配")
    teacher_kwargs = checkpoint.get("model_kwargs")
    if not teacher_kwargs:
        raise RuntimeError("Stage-2 checkpoint 缺少 model_kwargs")
    teacher = st2.EncoderWithHead(**teacher_kwargs)
    teacher.load_state_dict(checkpoint["model_state_dict"], strict=True)
    source = teacher.backbone.state_dict()
    target = model.backbone.state_dict()
    first_layer_names = {
        "tcn.network.0.conv1.weight",
        "tcn.network.0.downsample.weight",
    }
    for name, value in source.items():
        if name in first_layer_names:
            continue
        if name not in target or target[name].shape != value.shape:
            raise RuntimeError(
                "学生骨干必须与当前 Stage-2 骨干一致；如需小模型请关闭教师继承"
            )
        target[name] = value.detach().clone()
    channel_map = ((0, 0), (3, 3), (4, 4))
    for name in first_layer_names:
        if name not in target or name not in source:
            raise RuntimeError(f"教师骨干缺少首层参数: {name}")
        remapped = torch.zeros_like(target[name])
        for student_channel, teacher_channel in channel_map:
            remapped[:, student_channel] = source[name][:, teacher_channel]
        target[name] = remapped
    model.backbone.load_state_dict(target, strict=True)


def configure_device(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = not args.deterministic
        torch.backends.cudnn.deterministic = args.deterministic
        torch.backends.cuda.matmul.allow_tf32 = not args.no_tf32
        torch.backends.cudnn.allow_tf32 = not args.no_tf32
    return device


def arrays_to_tensors(arrays, device, args):
    names = (
        "positive_inputs",
        "pseudo_inputs",
        "mismatch_inputs",
        "positive_targets",
        "pseudo_targets",
        "mismatch_targets",
        "split_id",
        "source_type",
        "user_code",
        "loss_weight",
        "sampling_weight",
        "policy_weight",
    )
    tensors = {name: torch.from_numpy(arrays[name]) for name in names}
    total_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
    cache_enabled = device.type == "cuda" and not args.no_gpu_data_cache
    if cache_enabled:
        free_bytes, _ = torch.cuda.mem_get_info(device)
        cache_enabled = total_bytes <= free_bytes * args.gpu_cache_max_fraction
    storage_device = device if cache_enabled else torch.device("cpu")
    if cache_enabled:
        tensors = {name: tensor.to(device) for name, tensor in tensors.items()}
    print(
        f"[Data] tensor_bytes={total_bytes / 1024 ** 2:.1f}MB, "
        f"storage_device={storage_device}"
    )
    return tensors, storage_device


def fetch(tensor, indices, device):
    if tensor.device == indices.device:
        selected = tensor.index_select(0, indices)
    else:
        selected = tensor.index_select(0, indices.cpu())
    return selected.to(device, non_blocking=True)


def shift_edge(power, shifts):
    shifted = torch.empty_like(power)
    for shift in torch.unique(shifts).tolist():
        mask = shifts == int(shift)
        values = power[mask]
        if shift == 0:
            shifted[mask] = values
        elif shift > 0:
            shifted[mask, :shift] = values[:, :1]
            shifted[mask, shift:] = values[:, :-shift]
        else:
            width = -int(shift)
            shifted[mask, :-width] = values[:, width:]
            shifted[mask, -width:] = values[:, -1:]
    return shifted


def apply_power_shift(inputs, shifts, dp_clip):
    if torch.all(shifts == 0):
        return inputs
    output = inputs.clone()
    power = shift_edge(inputs[:, 0], shifts)
    difference = torch.zeros_like(power)
    difference[:, 1:] = power[:, 1:] - power[:, :-1]
    output[:, 0] = power
    output[:, 1] = difference.clamp_min(0.0).clamp_max(dp_clip) / dp_clip
    output[:, 2] = (-difference).clamp_min(0.0).clamp_max(dp_clip) / dp_clip
    return output


def weighted_mean(values, weights):
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def weighted_std(values, weights):
    mean = weighted_mean(values, weights)
    variance = weighted_mean((values - mean).square(), weights)
    return variance.clamp_min(1e-6).sqrt()


def weighted_corr_loss(prediction, target, weights):
    prediction_centered = prediction - weighted_mean(prediction, weights)
    target_centered = target - weighted_mean(target, weights)
    covariance = weighted_mean(
        prediction_centered * target_centered,
        weights,
    )
    denominator = (
        weighted_std(prediction, weights)
        * weighted_std(target, weights)
    ).clamp_min(1e-6)
    return 1.0 - (covariance / denominator).clamp(-1.0, 1.0)


def structure_loss_scale(epoch, args):
    if epoch <= args.freeze_backbone_epochs:
        return 0.0
    ramp_epoch = epoch - args.freeze_backbone_epochs
    return min(1.0, ramp_epoch / max(args.structure_ramp_epochs, 1))


def weighted_smooth_l1(prediction, target, sample_weight, target_weight, beta):
    error = F.smooth_l1_loss(prediction, target, beta=beta, reduction="none")
    weight = sample_weight[:, None] * target_weight[None, :]
    return (error * weight).sum() / weight.sum().clamp_min(1e-6)


def weighted_corr(prediction, target, weight):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    weight = weight / (weight.sum() + 1e-12)
    prediction_centered = prediction - np.sum(prediction * weight)
    target_centered = target - np.sum(target * weight)
    covariance = np.sum(weight * prediction_centered * target_centered)
    denominator = np.sqrt(
        np.sum(weight * prediction_centered ** 2)
        * np.sum(weight * target_centered ** 2)
    )
    return float(covariance / (denominator + 1e-12))


def top_fraction_mean(values, ratio=0.20):
    values = np.asarray(values, dtype=np.float64)
    count = max(1, int(math.ceil(len(values) * ratio)))
    return float(np.mean(np.partition(values, len(values) - count)[-count:]))


def user_top_score_corr(prediction, target, user_code, source_type):
    mask = source_type == 0
    prediction = prediction[mask]
    target = target[mask]
    user_code = user_code[mask]
    predicted_user = []
    target_user = []
    for user in np.unique(user_code):
        user_mask = user_code == user
        predicted_user.append(top_fraction_mean(prediction[user_mask]))
        target_user.append(top_fraction_mean(target[user_mask]))
    if len(predicted_user) < 2:
        return 0.0
    return weighted_corr(predicted_user, target_user, np.ones(len(predicted_user)))


@torch.inference_mode()
def predict_split(model, tensors, indices, device, batch_size, dp_clip, shift_points=0,
                  include_mismatch=True, use_amp=True):
    model.eval()
    positive_rows = []
    pseudo_rows = []
    mismatch_rows = []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        positive = fetch(tensors["positive_inputs"], batch_indices, device).float()
        pseudo = fetch(tensors["pseudo_inputs"], batch_indices, device).float()
        shifts = torch.full(
            (len(batch_indices),), int(shift_points), dtype=torch.long, device=device
        )
        positive = apply_power_shift(positive, shifts, dp_clip)
        pseudo = apply_power_shift(pseudo, shifts, dp_clip)
        inputs = [positive, pseudo]
        if include_mismatch:
            mismatch = fetch(tensors["mismatch_inputs"], batch_indices, device).float()
            mismatch = apply_power_shift(mismatch, shifts, dp_clip)
            inputs.append(mismatch)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            output = model(torch.cat(inputs, dim=0))["scores"]
        size = len(batch_indices)
        positive_rows.append(output[:size].float().cpu())
        pseudo_rows.append(output[size:2 * size].float().cpu())
        if include_mismatch:
            mismatch_rows.append(output[2 * size:].float().cpu())
    result = {
        "positive": torch.cat(positive_rows).numpy(),
        "pseudo": torch.cat(pseudo_rows).numpy(),
    }
    if include_mismatch:
        result["mismatch"] = torch.cat(mismatch_rows).numpy()
    return result


def compute_metrics(arrays, indices, prediction):
    index_np = indices.cpu().numpy()
    target = arrays["positive_targets"][index_np]
    pseudo_target = arrays["pseudo_targets"][index_np]
    mismatch_target = arrays["mismatch_targets"][index_np]
    weight = arrays["loss_weight"][index_np]
    policy_weight = arrays["policy_weight"][index_np] * weight
    source_type = arrays["source_type"][index_np]
    user_code = arrays["user_code"][index_np]
    positive = prediction["positive"]
    pseudo = prediction["pseudo"]
    mismatch = prediction["mismatch"]
    metrics = {}
    correlations = []
    for target_index, target_name in enumerate(STUDENT_TARGET_NAMES):
        corr = weighted_corr(positive[:, target_index], target[:, target_index], weight)
        mae = float(mean_absolute_error(
            target[:, target_index], positive[:, target_index], sample_weight=weight
        ))
        pred_std = float(np.std(positive[:, target_index]))
        target_std = float(np.std(target[:, target_index]))
        metrics[f"corr_{target_name}"] = corr
        metrics[f"mae_{target_name}"] = mae
        metrics[f"pred_std_{target_name}"] = pred_std
        metrics[f"target_std_{target_name}"] = target_std
        metrics[f"std_ratio_{target_name}"] = pred_std / max(target_std, 1e-8)
        correlations.append(corr)
    labels = np.concatenate((np.ones(len(positive)), np.zeros(len(pseudo))))
    scores = np.concatenate((positive[:, 0], pseudo[:, 0]))
    metrics["mean_corr"] = float(np.mean(correlations))
    metrics["pseudo_auc"] = float(roc_auc_score(labels, scores))
    metrics["pseudo_pair_accuracy"] = float(np.mean(positive[:, 0] > pseudo[:, 0]))
    metrics["pseudo_student_drop"] = float(np.mean(positive[:, 0] - pseudo[:, 0]))
    metrics["pseudo_teacher_drop"] = float(np.mean(target[:, 0] - pseudo_target[:, 0]))
    metrics["pseudo_drop_error"] = abs(
        metrics["pseudo_student_drop"] - metrics["pseudo_teacher_drop"]
    )
    metrics["pseudo_drop_corr"] = weighted_corr(
        positive[:, 0] - pseudo[:, 0],
        target[:, 0] - pseudo_target[:, 0],
        weight,
    )
    metrics["core_user_top20_corr"] = user_top_score_corr(
        positive[:, 0], target[:, 0], user_code, source_type
    )
    for source_name, source_id in (("old", 0), ("transfer", 1)):
        source_mask = source_type == source_id
        metrics[f"core_corr_{source_name}"] = (
            weighted_corr(
                positive[source_mask, 0],
                target[source_mask, 0],
                weight[source_mask],
            ) if source_mask.sum() >= 2 else 0.0
        )
    metrics["core_source_min_corr"] = min(
        metrics["core_corr_old"], metrics["core_corr_transfer"]
    )
    policy_mask = policy_weight > 1e-8
    if policy_mask.any():
        teacher_drop = target[policy_mask, 0] - mismatch_target[policy_mask, 0]
        student_drop = positive[policy_mask, 0] - mismatch[policy_mask, 0]
        selected_weight = policy_weight[policy_mask]
        metrics["policy_teacher_drop"] = float(np.average(teacher_drop, weights=selected_weight))
        metrics["policy_student_drop"] = float(np.average(student_drop, weights=selected_weight))
        metrics["policy_drop_error"] = abs(
            metrics["policy_student_drop"] - metrics["policy_teacher_drop"]
        )
        metrics["policy_pair_accuracy"] = float(np.average(
            student_drop > 0.0, weights=selected_weight
        ))
        metrics["policy_drop_corr"] = weighted_corr(
            student_drop, teacher_drop, selected_weight
        )
    else:
        metrics.update({
            "policy_teacher_drop": 0.0,
            "policy_student_drop": 0.0,
            "policy_drop_error": 0.0,
            "policy_pair_accuracy": 0.0,
            "policy_drop_corr": 0.0,
        })
    core_corr = np.clip(metrics["corr_storage_behavior"], 0.0, 1.0)
    mean_corr = np.clip(metrics["mean_corr"], 0.0, 1.0)
    core_mae_score = math.exp(-metrics["mae_storage_behavior"] / 0.05)
    std_ratio = max(metrics["std_ratio_storage_behavior"], 1e-6)
    std_score = math.exp(-abs(math.log(std_ratio)))
    source_score = np.clip(metrics["core_source_min_corr"], 0.0, 1.0)
    pseudo_delta_score = math.exp(-metrics["pseudo_drop_error"] / 0.10)
    pseudo_pair_score = 0.5 * (
        metrics["pseudo_auc"] + metrics["pseudo_pair_accuracy"]
    )
    policy_score = math.exp(-metrics["policy_drop_error"] / 0.05)
    policy_corr_score = np.clip(metrics["policy_drop_corr"], 0.0, 1.0)
    metrics["composite"] = float(
        0.30 * core_corr
        + 0.16 * mean_corr
        + 0.12 * core_mae_score
        + 0.08 * std_score
        + 0.08 * np.clip(metrics["core_user_top20_corr"], 0.0, 1.0)
        + 0.10 * source_score
        + 0.06 * pseudo_delta_score
        + 0.03 * pseudo_pair_score
        + 0.05 * policy_score
        + 0.02 * policy_corr_score
    )
    return metrics


def build_optimizer(model, args, device):
    groups = [
        {"params": model.backbone.parameters(), "lr": args.lr_backbone},
        {"params": model.head.parameters(), "lr": args.lr_head},
    ]
    kwargs = {"lr": args.lr_head, "weight_decay": args.weight_decay}
    if device.type == "cuda" and not args.no_fused_adam:
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(groups, **kwargs)
    except TypeError:
        kwargs.pop("fused", None)
        return torch.optim.AdamW(groups, **kwargs)


def configure_epoch_learning_rate(model, optimizer, args, epoch, inherited):
    freeze_epochs = args.freeze_backbone_epochs if inherited else 0
    frozen = epoch <= freeze_epochs
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(not frozen)
    head_lr = optimizer.param_groups[1]["lr"]
    scheduler_scale = head_lr / max(args.lr_head, 1e-12)
    if frozen:
        backbone_lr = 0.0
    else:
        ramp_epoch = epoch - freeze_epochs
        ramp = min(1.0, ramp_epoch / max(args.unfreeze_lr_ramp_epochs, 1))
        backbone_lr = args.lr_backbone * scheduler_scale * ramp
    optimizer.param_groups[0]["lr"] = backbone_lr
    return frozen, backbone_lr, head_lr


def train_epoch(model, optimizer, scaler, tensors, train_indices, device,
                dp_clip, args, use_amp, epoch):
    model.train()
    sampling_weight = fetch(tensors["sampling_weight"], train_indices, device).double()
    sampling_weight = sampling_weight / sampling_weight.sum().clamp_min(1e-12)
    sampled_local = torch.multinomial(
        sampling_weight, len(train_indices), replacement=True
    )
    if train_indices.device == sampled_local.device:
        order = train_indices.index_select(0, sampled_local)
    else:
        order = train_indices.index_select(0, sampled_local.cpu())
    target_weight = torch.from_numpy(TARGET_LOSS_WEIGHTS).to(device)
    metric_names = (
        "train_loss",
        "train_positive_loss",
        "train_core_point_loss",
        "train_pseudo_soft_loss",
        "train_pseudo_delta_loss",
        "train_pseudo_calibration_loss",
        "train_pseudo_rank_loss",
        "train_policy_delta_loss",
        "train_policy_calibration_loss",
        "train_policy_rank_loss",
        "train_core_moment_loss",
        "train_core_relation_loss",
        "train_shift_consistency_loss",
        "train_gradient_norm",
        "train_gradient_clipped_fraction",
        "train_gradient_nonfinite_fraction",
        "train_amp_overflow_fraction",
    )
    metric_sum = torch.zeros(
        len(metric_names),
        dtype=torch.float64,
        device=device,
    )
    batches = 0
    structure_scale = structure_loss_scale(epoch, args)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for start in range(0, len(order), args.batch_size):
        indices = order[start:start + args.batch_size]
        positive = fetch(tensors["positive_inputs"], indices, device).float()
        pseudo = fetch(tensors["pseudo_inputs"], indices, device).float()
        mismatch = fetch(tensors["mismatch_inputs"], indices, device).float()
        positive_target = fetch(tensors["positive_targets"], indices, device).float()
        pseudo_target = fetch(tensors["pseudo_targets"], indices, device).float()
        mismatch_target = fetch(tensors["mismatch_targets"], indices, device).float()
        sample_weight = fetch(tensors["loss_weight"], indices, device).float()
        policy_weight = fetch(tensors["policy_weight"], indices, device).float() * sample_weight
        shift_indices = torch.empty(0, dtype=torch.long, device=device)
        shifted_positive = None
        if (
            structure_scale > 0.0
            and args.max_shift_points > 0
            and args.shift_consistency_fraction > 0.0
        ):
            shift_count = max(
                1,
                int(round(len(indices) * args.shift_consistency_fraction)),
            )
            shift_indices = torch.randperm(
                len(indices),
                device=device,
            )[:shift_count]
            if len(shift_indices) > 0:
                shift_width = torch.randint(
                    1,
                    args.max_shift_points + 1,
                    (len(shift_indices),),
                    device=device,
                )
                shift_sign = torch.where(
                    torch.rand(len(shift_indices), device=device) < 0.5,
                    -torch.ones_like(shift_width),
                    torch.ones_like(shift_width),
                )
                shifted_positive = apply_power_shift(
                    positive.index_select(0, shift_indices),
                    shift_width * shift_sign,
                    dp_clip,
                )
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            model_inputs = [positive, pseudo, mismatch]
            if shifted_positive is not None:
                model_inputs.append(shifted_positive)
            combined = model(torch.cat(model_inputs, dim=0))["scores"]
            batch_size = len(indices)
            positive_output = combined[:batch_size]
            pseudo_output = combined[batch_size:2 * batch_size]
            mismatch_output = combined[2 * batch_size:3 * batch_size]
            positive_loss = weighted_smooth_l1(
                positive_output,
                positive_target,
                sample_weight,
                target_weight,
                args.smooth_l1_beta,
            )
            core_point_loss = weighted_mean(
                (positive_output[:, 0] - positive_target[:, 0]).square(),
                sample_weight,
            )
            pseudo_error = F.smooth_l1_loss(
                pseudo_output[:, 0],
                pseudo_target[:, 0],
                beta=args.smooth_l1_beta,
                reduction="none",
            )
            pseudo_loss = weighted_mean(pseudo_error, sample_weight)
            teacher_pseudo_drop = (
                positive_target[:, 0] - pseudo_target[:, 0]
            ).detach()
            student_pseudo_drop = positive_output[:, 0] - pseudo_output[:, 0]
            pseudo_confidence_weight = sample_weight * (
                teacher_pseudo_drop.abs() / args.pseudo_confidence_scale
            ).clamp(0.25, 1.0)
            pseudo_delta_error = F.smooth_l1_loss(
                student_pseudo_drop,
                teacher_pseudo_drop,
                beta=args.smooth_l1_beta,
                reduction="none",
            )
            pseudo_delta_loss = weighted_mean(
                pseudo_delta_error,
                pseudo_confidence_weight,
            )
            pseudo_calibration_loss = F.smooth_l1_loss(
                weighted_mean(student_pseudo_drop, sample_weight),
                weighted_mean(teacher_pseudo_drop, sample_weight),
                beta=args.smooth_l1_beta,
            )
            pseudo_rank_mask = teacher_pseudo_drop >= args.pseudo_min_teacher_drop
            if pseudo_rank_mask.any():
                pseudo_required_margin = torch.minimum(
                    teacher_pseudo_drop[pseudo_rank_mask].clamp_min(0.0),
                    torch.full_like(
                        teacher_pseudo_drop[pseudo_rank_mask],
                        args.pseudo_margin,
                    ),
                )
                pseudo_rank_loss = weighted_mean(
                    F.relu(
                        pseudo_required_margin
                        - student_pseudo_drop[pseudo_rank_mask]
                    ),
                    sample_weight[pseudo_rank_mask],
                )
            else:
                pseudo_rank_loss = positive_loss.new_zeros(())
            policy_mask = policy_weight > 1e-8
            if policy_mask.any():
                teacher_drop = (
                    positive_target[policy_mask, 0] - mismatch_target[policy_mask, 0]
                ).detach()
                student_drop = (
                    positive_output[policy_mask, 0] - mismatch_output[policy_mask, 0]
                )
                selected_weight = policy_weight[policy_mask]
                policy_delta_loss = weighted_mean(
                    F.smooth_l1_loss(
                        student_drop,
                        teacher_drop,
                        beta=args.smooth_l1_beta,
                        reduction="none",
                    ),
                    selected_weight,
                )
                policy_calibration_loss = F.smooth_l1_loss(
                    weighted_mean(student_drop, selected_weight),
                    weighted_mean(teacher_drop, selected_weight),
                    beta=args.smooth_l1_beta,
                )
                policy_rank_mask = teacher_drop >= args.policy_min_teacher_drop
                if policy_rank_mask.any():
                    policy_required_margin = torch.minimum(
                        teacher_drop[policy_rank_mask].clamp_min(0.0),
                        torch.full_like(
                            teacher_drop[policy_rank_mask],
                            args.policy_margin,
                        ),
                    )
                    policy_rank_loss = weighted_mean(
                        F.relu(
                            policy_required_margin
                            - student_drop[policy_rank_mask]
                        ),
                        selected_weight[policy_rank_mask],
                    )
                else:
                    policy_rank_loss = positive_loss.new_zeros(())
            else:
                policy_delta_loss = positive_loss.new_zeros(())
                policy_calibration_loss = positive_loss.new_zeros(())
                policy_rank_loss = positive_loss.new_zeros(())

            core_prediction = positive_output[:, 0]
            core_target = positive_target[:, 0].detach()
            core_moment_loss = (
                F.smooth_l1_loss(
                    weighted_mean(core_prediction, sample_weight),
                    weighted_mean(core_target, sample_weight),
                    beta=args.smooth_l1_beta,
                )
                + F.smooth_l1_loss(
                    weighted_std(core_prediction, sample_weight),
                    weighted_std(core_target, sample_weight),
                    beta=args.smooth_l1_beta,
                )
            )
            core_relation_loss = weighted_corr_loss(
                core_prediction,
                core_target,
                sample_weight,
            )
            if shifted_positive is not None:
                shifted_output = combined[3 * batch_size:]
                shift_consistency_loss = weighted_smooth_l1(
                    shifted_output,
                    positive_output.index_select(0, shift_indices).detach(),
                    sample_weight.index_select(0, shift_indices),
                    target_weight,
                    args.smooth_l1_beta,
                )
            else:
                shift_consistency_loss = positive_loss.new_zeros(())
            loss = (
                positive_loss
                + args.lambda_core_point * core_point_loss
                + structure_scale * (
                    args.lambda_pseudo_soft * pseudo_loss
                    + args.lambda_pseudo_delta * pseudo_delta_loss
                    + args.lambda_pseudo_calibration * pseudo_calibration_loss
                    + args.lambda_pseudo_rank * pseudo_rank_loss
                    + args.lambda_policy_delta * policy_delta_loss
                    + args.lambda_policy_calibration * policy_calibration_loss
                    + args.lambda_policy_rank * policy_rank_loss
                    + args.lambda_core_moment * core_moment_loss
                    + args.lambda_core_relation * core_relation_loss
                    + args.lambda_shift_consistency * shift_consistency_loss
                )
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("学生训练出现 NaN/Inf")
        scale_before = float(scaler.get_scale())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        gradient_is_finite = torch.isfinite(gradient_norm.detach())
        if not scaler.is_enabled() and not gradient_is_finite.item():
            raise FloatingPointError("学生训练梯度范数出现 NaN/Inf")
        gradient_norm_for_log = torch.where(
            gradient_is_finite,
            gradient_norm.detach(),
            gradient_norm.detach().new_zeros(()),
        )
        gradient_clipped = (
            gradient_is_finite & (gradient_norm.detach() > args.clip)
        ).to(loss.dtype)
        gradient_nonfinite = (~gradient_is_finite).to(loss.dtype)
        scaler.step(optimizer)
        scaler.update()
        amp_overflow = loss.detach().new_tensor(
            float(scaler.get_scale()) < scale_before
        )
        metric_sum += torch.stack((
            loss.detach(),
            positive_loss.detach(),
            core_point_loss.detach(),
            pseudo_loss.detach(),
            pseudo_delta_loss.detach(),
            pseudo_calibration_loss.detach(),
            pseudo_rank_loss.detach(),
            policy_delta_loss.detach(),
            policy_calibration_loss.detach(),
            policy_rank_loss.detach(),
            core_moment_loss.detach(),
            core_relation_loss.detach(),
            shift_consistency_loss.detach(),
            gradient_norm_for_log,
            gradient_clipped,
            gradient_nonfinite,
            amp_overflow,
        )).to(torch.float64)
        batches += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    mean_metrics = (
        metric_sum / max(batches, 1)
    ).detach().cpu().numpy()
    return {
        **dict(zip(metric_names, mean_metrics.tolist())),
        "train_structure_scale": structure_scale,
        "train_wall_seconds": elapsed,
        "train_pairs_per_second": len(train_indices) / max(elapsed, 1e-8),
    }


def save_checkpoint(path, model, args, model_kwargs, epoch, metrics, arrays,
                    teacher_checkpoint, dataset_path):
    checkpoint = {
        "student_objective_version": STUDENT_OBJECTIVE_VERSION,
        "student_dataset_version": STUDENT_DATASET_VERSION,
        "model_arch_version": MODEL_ARCH_VERSION,
        "model_state_dict": model.state_dict(),
        "model_kwargs": model_kwargs,
        "epoch": int(epoch),
        "metrics": metrics,
        "channel_names": list(STUDENT_CHANNEL_NAMES),
        "target_names": list(STUDENT_TARGET_NAMES),
        "dp_clip_q99": float(arrays["dp_clip_q99"][0]),
        "teacher_epoch": int(teacher_checkpoint.get("epoch", -1)),
        "teacher_stage2_score": float(arrays["teacher_stage2_score"][0]),
        "teacher_checkpoint_sha256": str(
            arrays["teacher_checkpoint_sha256"].item()
        ),
        "teacher_checkpoint": str(resolve_path(args.phase2_ckpt)),
        "student_dataset": str(dataset_path),
        "args": vars(args),
        "score_semantics": "日级储能证据分数，不是经真实非储能样本校准的概率",
    }
    torch.save(checkpoint, path)


def validation_thresholds(prediction):
    pseudo = prediction["pseudo"][:, 0]
    return {
        "day_score_mid": float(np.quantile(pseudo, 0.90)),
        "day_score_high": float(np.quantile(pseudo, 0.98)),
        "note": "阈值仅控制B类反事实伪负高分率，不代表真实非储能误报率",
    }


def acceptance_verdict(metrics, shifted_metrics):
    reasons = []
    warnings = []
    if metrics["corr_storage_behavior"] < 0.58:
        reasons.append("storage_behavior 测试相关性低于 0.58")
    if metrics["mae_storage_behavior"] > 0.05:
        reasons.append("storage_behavior 测试 MAE 高于 0.05")
    if metrics["pseudo_auc"] < 0.94:
        reasons.append("B类伪负 AUC 低于 0.94")
    if metrics["pseudo_pair_accuracy"] < 0.98:
        reasons.append("B类伪负配对准确率低于 0.98")
    if metrics["pseudo_drop_error"] > 0.12:
        warnings.append("B类伪负教师/学生平均降幅误差高于 0.12")
    if metrics["policy_drop_error"] > 0.05:
        warnings.append("政策反事实平均降幅误差高于 0.05")
    if not 0.60 <= metrics["std_ratio_storage_behavior"] <= 1.60:
        warnings.append("storage_behavior 输出方差与教师偏差较大")
    minimum_shift_corr = min(
        value["corr_storage_behavior"] for value in shifted_metrics.values()
    )
    if metrics["corr_storage_behavior"] - minimum_shift_corr > 0.04:
        warnings.append("±2点错位导致核心相关性下降超过 0.04")
    return {
        "verdict": "PASS" if not reasons else "FAIL",
        "reasons": reasons,
        "warnings": warnings,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="当前 Stage-2 下游学生广义蒸馏训练")
    parser.add_argument("--student_npz", default="step_3/student_train_dataset.npz")
    parser.add_argument("--phase2_ckpt", default="step_2/best_model_st2.pt")
    parser.add_argument("--save_dir", default="train_student")
    parser.add_argument("--save_name", default="best_model_student.pt")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--nhid", type=int, default=64)
    parser.add_argument("--levels", type=int, default=DEFAULT_TEMPORAL_LEVELS)
    parser.add_argument("--ksize", type=int, default=7)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lstm_hidden", type=int, default=64)
    parser.add_argument("--lstm_layers", type=int, default=1)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--lr_backbone", type=float, default=5e-5)
    parser.add_argument("--lr_head", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--smooth_l1_beta", type=float, default=0.05)
    parser.add_argument("--lambda_core_point", type=float, default=0.75)
    parser.add_argument("--lambda_pseudo_soft", type=float, default=0.10)
    parser.add_argument("--lambda_pseudo_delta", type=float, default=0.10)
    parser.add_argument("--lambda_pseudo_calibration", type=float, default=0.02)
    parser.add_argument("--lambda_pseudo_rank", type=float, default=0.02)
    parser.add_argument("--lambda_policy_delta", type=float, default=0.08)
    parser.add_argument("--lambda_policy_calibration", type=float, default=0.04)
    parser.add_argument("--lambda_policy_rank", type=float, default=0.01)
    parser.add_argument("--lambda_core_moment", type=float, default=0.08)
    parser.add_argument("--lambda_core_relation", type=float, default=0.005)
    parser.add_argument("--lambda_shift_consistency", type=float, default=0.03)
    parser.add_argument("--pseudo_margin", type=float, default=0.12)
    parser.add_argument("--pseudo_min_teacher_drop", type=float, default=0.03)
    parser.add_argument("--pseudo_confidence_scale", type=float, default=0.10)
    parser.add_argument("--policy_margin", type=float, default=0.03)
    parser.add_argument("--policy_min_teacher_drop", type=float, default=0.01)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=2)
    parser.add_argument("--unfreeze_lr_ramp_epochs", type=int, default=3)
    parser.add_argument("--structure_ramp_epochs", type=int, default=3)
    parser.add_argument("--max_shift_points", type=int, default=2)
    parser.add_argument("--shift_consistency_fraction", type=float, default=0.40)
    parser.add_argument("--min_epochs", type=int, default=15)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--lr_patience", type=int, default=6)
    parser.add_argument("--best_min_delta", type=float, default=1e-4)
    parser.add_argument("--no_inherit_teacher", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_tf32", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no_fused_adam", action="store_true")
    parser.add_argument("--no_gpu_data_cache", action="store_true")
    parser.add_argument("--gpu_cache_max_fraction", type=float, default=0.30)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 <= args.shift_consistency_fraction <= 1.0:
        raise ValueError("shift_consistency_fraction 必须位于 [0, 1]")
    if args.pseudo_confidence_scale <= 0.0:
        raise ValueError("pseudo_confidence_scale 必须大于 0")
    set_seed(args.seed)
    device = configure_device(args)
    use_amp = not args.no_amp
    output_dir = resolve_path(args.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = resolve_path(args.student_npz)
    arrays = load_student_arrays(dataset_path)
    teacher_checkpoint_path = resolve_path(args.phase2_ckpt)
    teacher_checkpoint = load_checkpoint(
        teacher_checkpoint_path,
        torch.device("cpu"),
    )
    validate_dataset_teacher(arrays, teacher_checkpoint, teacher_checkpoint_path)
    tensors, storage_device = arrays_to_tensors(arrays, device, args)
    split_indices = {
        split_id: torch.as_tensor(
            np.flatnonzero(arrays["split_id"] == split_id),
            dtype=torch.long,
            device=storage_device,
        )
        for split_id in SPLIT_NAMES
    }
    print(
        "[Data] " + ", ".join(
            f"{SPLIT_NAMES[split_id]}={len(indices)}"
            for split_id, indices in split_indices.items()
        )
    )
    model_kwargs = model_kwargs_from_args(args)
    model = StudentModel(**model_kwargs).to(device)
    inherited = not args.no_inherit_teacher
    if inherited:
        inherit_teacher_backbone(model, teacher_checkpoint)
    optimizer = build_optimizer(model, args, device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.lr_patience
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    dp_clip = float(arrays["dp_clip_q99"][0])
    best_score = -float("inf")
    best_epoch = 0
    no_improve = 0
    history = []
    save_path = output_dir / args.save_name
    print(
        f"[Student] device={device}, inherited={inherited}, amp={use_amp}, "
        f"parameters={sum(parameter.numel() for parameter in model.parameters())}"
    )

    for epoch in range(1, args.epochs + 1):
        frozen, backbone_lr, head_lr = configure_epoch_learning_rate(
            model, optimizer, args, epoch, inherited
        )
        train_metrics = train_epoch(
            model,
            optimizer,
            scaler,
            tensors,
            split_indices[0],
            device,
            dp_clip,
            args,
            use_amp,
            epoch,
        )
        validation_prediction = predict_split(
            model,
            tensors,
            split_indices[1],
            device,
            args.eval_batch_size,
            dp_clip,
            use_amp=use_amp,
        )
        validation_metrics = compute_metrics(
            arrays, split_indices[1], validation_prediction
        )
        row = {
            "epoch": epoch,
            "backbone_frozen": int(frozen),
            "lr_backbone": backbone_lr,
            "lr_head": head_lr,
            **train_metrics,
            **{f"val_{name}": value for name, value in validation_metrics.items()},
        }
        history.append(row)
        score = validation_metrics["composite"]
        print(
            f"[Epoch {epoch:03d}] score={score:.4f} "
            f"core_r={validation_metrics['corr_storage_behavior']:.3f} "
            f"core_mae={validation_metrics['mae_storage_behavior']:.3f} "
            f"auc={validation_metrics['pseudo_auc']:.3f} "
            f"pair={validation_metrics['pseudo_pair_accuracy']:.3f} "
            f"pdrop={validation_metrics['pseudo_student_drop']:.3f}/"
            f"{validation_metrics['pseudo_teacher_drop']:.3f} "
            f"policy={validation_metrics['policy_student_drop']:.3f}/"
            f"{validation_metrics['policy_teacher_drop']:.3f} "
            f"std={validation_metrics['std_ratio_storage_behavior']:.2f} "
            f"grad={train_metrics['train_gradient_norm']:.3f} "
            f"speed={train_metrics['train_pairs_per_second']:.0f} pair/s"
        )
        improved = score > best_score + args.best_min_delta
        if improved:
            best_score = score
            best_epoch = epoch
            no_improve = 0
            save_checkpoint(
                save_path,
                model,
                args,
                model_kwargs,
                epoch,
                validation_metrics,
                arrays,
                teacher_checkpoint,
                dataset_path,
            )
        else:
            no_improve += 1
        if epoch > args.freeze_backbone_epochs + args.unfreeze_lr_ramp_epochs:
            scheduler.step(score)
        pd.DataFrame(history).to_csv(
            output_dir / "student_training_history.csv",
            index=False,
            encoding="utf-8-sig",
        )
        if epoch >= args.min_epochs and no_improve >= args.early_stop_patience:
            print(f"[EarlyStop] epoch={epoch}, best_epoch={best_epoch}")
            break

    best_checkpoint = load_checkpoint(save_path, device)
    model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
    validation_prediction = predict_split(
        model,
        tensors,
        split_indices[1],
        device,
        args.eval_batch_size,
        dp_clip,
        use_amp=use_amp,
    )
    validation_metrics = compute_metrics(arrays, split_indices[1], validation_prediction)
    test_prediction = predict_split(
        model,
        tensors,
        split_indices[2],
        device,
        args.eval_batch_size,
        dp_clip,
        use_amp=use_amp,
    )
    test_metrics = compute_metrics(arrays, split_indices[2], test_prediction)
    shifted_metrics = {}
    for shift in (-2, -1, 1, 2):
        shifted_prediction = predict_split(
            model,
            tensors,
            split_indices[2],
            device,
            args.eval_batch_size,
            dp_clip,
            shift_points=shift,
            use_amp=use_amp,
        )
        shifted_metrics[str(shift)] = compute_metrics(
            arrays, split_indices[2], shifted_prediction
        )
    verdict = acceptance_verdict(test_metrics, shifted_metrics)
    summary = {
        **verdict,
        "student_objective_version": STUDENT_OBJECTIVE_VERSION,
        "student_dataset_version": STUDENT_DATASET_VERSION,
        "teacher_epoch": int(teacher_checkpoint.get("epoch", -1)),
        "teacher_checkpoint_sha256": str(
            arrays["teacher_checkpoint_sha256"].item()
        ),
        "best_epoch": int(best_epoch),
        "best_validation_composite": float(best_score),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "shifted_test_metrics": shifted_metrics,
        "thresholds": validation_thresholds(validation_prediction),
        "channel_names": list(STUDENT_CHANNEL_NAMES),
        "target_names": list(STUDENT_TARGET_NAMES),
        "score_semantics": "日级储能证据分数，不是经真实非储能样本校准的概率",
    }
    (output_dir / "student_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "verdict": summary["verdict"],
        "best_epoch": best_epoch,
        "test_core_corr": test_metrics["corr_storage_behavior"],
        "test_core_mae": test_metrics["mae_storage_behavior"],
        "test_pseudo_auc": test_metrics["pseudo_auc"],
        "test_pair_accuracy": test_metrics["pseudo_pair_accuracy"],
        "policy_drop": [
            test_metrics["policy_student_drop"],
            test_metrics["policy_teacher_drop"],
        ],
    }, ensure_ascii=False, indent=2))
    print(f"[Done] {save_path}")


if __name__ == "__main__":
    main()
