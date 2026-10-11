import argparse
import csv
import gc
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "2")
SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_PARENT = SCRIPT_DIR.parents[1]
REQUIRED_STAGE1_OBJECTIVE_VERSION = (
    "joint-z-vicreg-h-vcreg-v8-batched-augment-equivalent"
)
STAGE2_OBJECTIVE_VERSION = "teacher-multitask-v3-hierarchy-p-attenuation"
TRANSFER_TARGET_CACHE_VERSION = "stage2-transfer-targets-v1"
P_ATTENUATION_CACHE_VERSION = "stage2-p-attenuation-v1"
STAGE2_TARGET_KEYS = (
    "physical_targets",
    "physical_weights",
    "response_targets",
    "response_weights",
    "behavior_targets",
    "behavior_weights",
)
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    from TCN.adding_problem.model import (
        DEFAULT_TCN_LEVELS,
        TCN_LSTM,
        TCN_LSTM_ARCH_VERSION,
    )
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
        summarize_stage2_targets,
    )
    from TCN.adding_problem.utils import (
        augment_sync_intensity_torch,
        load_storage_data,
        scale_pq,
    )
except ImportError:
    from model import DEFAULT_TCN_LEVELS, TCN_LSTM, TCN_LSTM_ARCH_VERSION
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
        summarize_stage2_targets,
    )
    from utils import augment_sync_intensity_torch, load_storage_data, scale_pq

class ConsoleTee:
    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, message):
        self.stream.write(message)
        self.log_file.write(message)

    def flush(self):
        self.stream.flush()
        self.log_file.flush()


def set_seed(seed=1111, cudnn_benchmark=False, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark and not deterministic)


def resolve_project_path(path):
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    local = SCRIPT_DIR / candidate
    return local if local.exists() or not candidate.exists() else candidate.resolve()


def load_torch_checkpoint(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def setup_console_logging(save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = save_dir / f"{timestamp}_phase2.md"
    log_file = log_path.open("w", encoding="utf-8")
    log_file.write(f"# Stage-2 教师训练日志\n\n- start_time: {timestamp}\n\n```text\n")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = ConsoleTee(original_stdout, log_file)
    sys.stderr = ConsoleTee(original_stderr, log_file)
    return log_path, log_file, original_stdout, original_stderr


def close_console_logging(log_file, original_stdout, original_stderr):
    try:
        log_file.write("\n```\n")
        log_file.flush()
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


@dataclass(frozen=True)
class Stage2AugmentConfig:
    scale_min: float = 0.95
    scale_max: float = 1.05
    smooth_prob: float = 0.20
    smooth_window: int = 3
    noise_std: float = 0.002


def mask_physical_context(inputs):
    masked = inputs.clone()
    masked[:, 2:5, :] = 0.0
    return masked


def mask_behavior_season(inputs):
    masked = inputs.clone()
    masked[:, 2, :] = 0.0
    return masked


def mask_policy_channels(inputs):
    return mask_physical_context(inputs)


def make_pseudo_a_variant(inputs, shift_points=8, sign_flip=False):
    if shift_points <= 0 or shift_points >= inputs.size(-1):
        raise ValueError("A 类错配位移必须位于 (0, 96)")
    pseudo = inputs.clone()
    storage = inputs[:, 1, :]
    shifted = torch.zeros_like(storage)
    if sign_flip:
        shifted[:, :-shift_points] = -storage[:, shift_points:]
    else:
        shifted[:, shift_points:] = storage[:, :-shift_points]
    pseudo[:, 1, :] = shifted
    return pseudo


def make_pseudo_a_inputs(inputs, shift_points=8, return_type=False):
    if shift_points <= 0 or shift_points >= inputs.size(-1):
        raise ValueError("A 类错配位移必须位于 (0, 96)")
    shift_only = make_pseudo_a_variant(inputs, shift_points, sign_flip=False)
    shift_sign = make_pseudo_a_variant(inputs, shift_points, sign_flip=True)
    midpoint = inputs.size(0) // 2
    pseudo = shift_only
    pseudo[midpoint:] = shift_sign[midpoint:]
    if not return_type:
        return pseudo
    pseudo_type = torch.zeros(
        inputs.size(0), device=inputs.device, dtype=torch.long
    )
    pseudo_type[midpoint:] = 1
    return pseudo, pseudo_type


def compute_strength_from_pq(power, storage, q_global_scale=1.0):
    storage_physical_scale = storage.abs().sum(dim=2).view(-1) * float(q_global_scale)
    power_scale = power.abs().sum(dim=2).view(-1) + 1e-7
    return (storage_physical_scale / power_scale).view(-1, 1)


class EncoderWithHead(nn.Module):
    def __init__(self, input_channels, nhid, levels, ksize, dropout, lstm_hidden,
                 lstm_layers, embedding_dim, proj_dim=128):
        super().__init__()
        self.input_channels = int(input_channels)
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
        self.physical_head = nn.Linear(embedding_dim, len(PHYSICAL_TARGET_NAMES))
        self.response_head = nn.Linear(embedding_dim, len(RESPONSE_TARGET_NAMES))
        self.behavior_head = nn.Linear(embedding_dim, len(BEHAVIOR_TARGET_NAMES))
        self._reset_heads()

    def _reset_heads(self):
        heads = [self.physical_head, self.response_head, self.behavior_head]
        for module in list(self.proj_head.modules()) + heads:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(self, inputs):
        if inputs.size(1) > self.input_channels:
            inputs = inputs[:, :self.input_channels, :]
        return torch.clamp(self.backbone(inputs), -5.0, 5.0)

    def forward(self, inputs):
        embedding = self.encode(inputs)
        projection = self.proj_head(embedding)
        return embedding, projection


def load_compatible_state_dict(model, state_dict,
                               skip_prefixes=(
                                   "behavior_head.",
                                   "physical_head.",
                                   "response_head.",
                                   "mode_head.",
                               )):
    model_state = model.state_dict()
    compatible = {}
    skipped = []
    for name, value in state_dict.items():
        if any(name.startswith(prefix) for prefix in skip_prefixes):
            skipped.append((name, "new Stage-2 head"))
            continue
        if name not in model_state:
            skipped.append((name, "not present"))
            continue
        target = model_state[name]
        if target.shape == value.shape:
            compatible[name] = value
            continue
        if (
            value.ndim == 3
            and target.ndim == 3
            and value.shape[0] == target.shape[0]
            and value.shape[2] == target.shape[2]
            and value.shape[1] < target.shape[1]
        ):
            expanded = torch.zeros_like(target)
            expanded[:, :value.shape[1], :] = value
            compatible[name] = expanded
            skipped.append((name, f"expanded {tuple(value.shape)} -> {tuple(target.shape)}"))
            continue
        skipped.append((name, f"shape {tuple(value.shape)} -> {tuple(target.shape)}"))
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return missing, unexpected, skipped


def embedding_regularization(clean_embedding, augmented_embedding, target_std=0.08,
                             covariance_weight=5.0):
    clean = F.normalize(
        torch.nan_to_num(clean_embedding, nan=0.0, posinf=5.0, neginf=-5.0),
        dim=1,
    )
    augmented = F.normalize(
        torch.nan_to_num(augmented_embedding, nan=0.0, posinf=5.0, neginf=-5.0),
        dim=1,
    )
    stable = 0.5 * (clean + augmented)
    std = torch.sqrt(torch.var(stable, dim=0, unbiased=False) + 1e-4)
    variance_loss = F.relu(target_std - std).mean()
    centered = stable - stable.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(centered.size(0) - 1, 1)
    off_diagonal = covariance - torch.diag(torch.diag(covariance))
    covariance_loss = off_diagonal.square().sum() / max(centered.size(1), 1)
    total = variance_loss + covariance_weight * covariance_loss
    return total, std.min(), std.mean(), variance_loss, covariance_loss


def weighted_smooth_l1(prediction, target, weight, dimension_weight=None):
    error = F.smooth_l1_loss(prediction, target, reduction="none")
    effective_weight = weight
    if dimension_weight is not None:
        effective_weight = effective_weight * dimension_weight.view(1, -1)
    return (error * effective_weight).sum() / (effective_weight.sum() + 1e-7)


class Stage2TeacherLoss(nn.Module):
    def __init__(self, physical_head, response_head, behavior_head,
                 lambda_physical=1.2, lambda_response=1.0, lambda_behavior=1.0,
                 lambda_transfer=0.6, lambda_policy_mismatch=0.4,
                 lambda_policy_rank=0.25, policy_margin=0.10,
                 lambda_pseudo_rank=0.2, pseudo_margin=0.15,
                 lambda_hierarchy=0.03, lambda_p_attenuation=0.05,
                 p_attenuation_margin=0.015, lambda_role_invariance=0.01,
                 lambda_consistency=0.15, lambda_reg=0.05, lambda_anchor=0.05,
                 warmup_epochs=5, target_std=0.08,
                 reg_covariance_weight=5.0):
        super().__init__()
        self.physical_head = physical_head
        self.response_head = response_head
        self.behavior_head = behavior_head
        self.lambda_physical = float(lambda_physical)
        self.lambda_response = float(lambda_response)
        self.lambda_behavior = float(lambda_behavior)
        self.lambda_transfer = float(lambda_transfer)
        self.lambda_policy_mismatch = float(lambda_policy_mismatch)
        self.lambda_policy_rank = float(lambda_policy_rank)
        self.policy_margin = float(policy_margin)
        self.lambda_pseudo_rank = float(lambda_pseudo_rank)
        self.pseudo_margin = float(pseudo_margin)
        self.lambda_hierarchy = float(lambda_hierarchy)
        self.lambda_p_attenuation = float(lambda_p_attenuation)
        self.p_attenuation_margin = float(p_attenuation_margin)
        self.lambda_role_invariance = float(lambda_role_invariance)
        self.lambda_consistency = float(lambda_consistency)
        self.lambda_reg = float(lambda_reg)
        self.lambda_anchor = float(lambda_anchor)
        self.warmup_epochs = int(warmup_epochs)
        self.target_std = float(target_std)
        self.reg_covariance_weight = float(reg_covariance_weight)
        self.storage_evidence_index = PHYSICAL_TARGET_NAMES.index("storage_evidence")
        self.pq_alignment_index = PHYSICAL_TARGET_NAMES.index("pq_alignment")
        self.edge_pair_index = PHYSICAL_TARGET_NAMES.index("edge_pair_match")
        self.segment_coherence_index = PHYSICAL_TARGET_NAMES.index(
            "visible_segment_coherence"
        )
        self.activity_strength_index = PHYSICAL_TARGET_NAMES.index("activity_strength")
        self.response_confidence_index = RESPONSE_TARGET_NAMES.index(
            "response_confidence"
        )
        self.policy_pattern_index = BEHAVIOR_TARGET_NAMES.index("policy_pattern")
        self.storage_behavior_index = BEHAVIOR_TARGET_NAMES.index("storage_behavior")
        self.register_buffer(
            "physical_dimension_weight",
            torch.tensor([1.0, 0.9, 0.9, 0.7, 1.2]),
        )
        self.register_buffer(
            "behavior_dimension_weight",
            torch.tensor([1.0, 1.0, 0.8, 1.1, 1.3]),
        )

    def _predict(self, head, embedding):
        return torch.sigmoid(head(F.normalize(embedding, dim=1)))

    def forward(self, embeddings, batch, epoch):
        physical_clean = self._predict(self.physical_head, embeddings["physical_clean"])
        physical_augmented = self._predict(
            self.physical_head, embeddings["physical_augmented"]
        )
        physical_transfer = self._predict(
            self.physical_head, embeddings["physical_transfer"]
        )
        response_clean = self._predict(self.response_head, embeddings["physical_clean"])
        response_augmented = self._predict(
            self.response_head, embeddings["physical_augmented"]
        )
        response_transfer = self._predict(
            self.response_head, embeddings["physical_transfer"]
        )
        behavior_clean = self._predict(self.behavior_head, embeddings["context_clean"])
        behavior_transfer = self._predict(
            self.behavior_head, embeddings["context_transfer"]
        )
        behavior_mismatch = self._predict(
            self.behavior_head, embeddings["context_mismatch"]
        )
        physical_pseudo = self._predict(self.physical_head, embeddings["pseudo"])
        response_pseudo = self._predict(self.response_head, embeddings["pseudo"])

        sample_weight = batch["sample_weight"].view(-1, 1)
        transfer_sample_weight = sample_weight * batch["transfer_weight"].view(-1, 1)
        physical_weight = batch["physical_weight"] * sample_weight
        response_weight = batch["response_weight"] * sample_weight
        behavior_weight = batch["behavior_weight"] * sample_weight
        transfer_physical_weight = batch["transfer_physical_weight"] * transfer_sample_weight
        transfer_response_weight = batch["transfer_response_weight"] * transfer_sample_weight
        transfer_behavior_weight = batch["transfer_behavior_weight"] * transfer_sample_weight
        mismatch_behavior_weight = batch["mismatch_behavior_weight"] * transfer_sample_weight

        physical_loss = 0.5 * (
            weighted_smooth_l1(
                physical_clean,
                batch["physical_target"],
                physical_weight,
                self.physical_dimension_weight,
            )
            + weighted_smooth_l1(
                physical_augmented,
                batch["physical_target"],
                physical_weight,
                self.physical_dimension_weight,
            )
        )
        response_loss = 0.5 * (
            weighted_smooth_l1(
                response_clean, batch["response_target"], response_weight
            )
            + weighted_smooth_l1(
                response_augmented, batch["response_target"], response_weight
            )
        )
        behavior_loss = weighted_smooth_l1(
            behavior_clean,
            batch["behavior_target"],
            behavior_weight,
            self.behavior_dimension_weight,
        )
        transfer_loss = (
            0.30 * weighted_smooth_l1(
                physical_transfer,
                batch["transfer_physical_target"],
                transfer_physical_weight,
                self.physical_dimension_weight,
            )
            + 0.30 * weighted_smooth_l1(
                response_transfer,
                batch["transfer_response_target"],
                transfer_response_weight,
            )
            + 0.40 * weighted_smooth_l1(
                behavior_transfer,
                batch["transfer_behavior_target"],
                transfer_behavior_weight,
                self.behavior_dimension_weight,
            )
        )
        mismatch_loss = weighted_smooth_l1(
            behavior_mismatch,
            batch["mismatch_behavior_target"],
            mismatch_behavior_weight,
            self.behavior_dimension_weight,
        )

        transfer_score = behavior_transfer[:, self.storage_behavior_index]
        mismatch_score = behavior_mismatch[:, self.storage_behavior_index]
        target_drop = (
            batch["transfer_behavior_target"][:, self.storage_behavior_index]
            - batch["mismatch_behavior_target"][:, self.storage_behavior_index]
        ).clamp_min(0.0)
        policy_rank_weight = transfer_sample_weight.view(-1) * target_drop
        predicted_drop = transfer_score - mismatch_score
        required_margin = torch.minimum(
            target_drop,
            torch.full_like(target_drop, self.policy_margin),
        )
        policy_rank_loss = (
            F.relu(required_margin - predicted_drop)
            * policy_rank_weight
        ).sum() / (policy_rank_weight.sum() + 1e-7)
        policy_calibration_weight = transfer_sample_weight.view(-1)
        policy_drop_calibration_loss = (
            F.smooth_l1_loss(predicted_drop, target_drop, reduction="none")
            * policy_calibration_weight
        ).sum() / (policy_calibration_weight.sum() + 1e-7)

        real_evidence = (
            0.70 * physical_clean[:, self.storage_evidence_index]
            + 0.30 * response_clean[:, 2]
        )
        pseudo_evidence = (
            0.70 * physical_pseudo[:, self.storage_evidence_index]
            + 0.30 * response_pseudo[:, 2]
        )
        pseudo_rank_weight = (
            sample_weight.view(-1)
            * batch["physical_weight"][:, self.storage_evidence_index]
            * batch["physical_target"][:, self.storage_evidence_index]
        )
        pseudo_rank_loss = (
            F.relu(self.pseudo_margin + pseudo_evidence - real_evidence)
            * pseudo_rank_weight
        ).sum() / (pseudo_rank_weight.sum() + 1e-7)
        evidence_drop = real_evidence - pseudo_evidence
        pseudo_pair_accuracy = (
            (evidence_drop > 0.0).to(evidence_drop.dtype) * pseudo_rank_weight
        ).sum() / (pseudo_rank_weight.sum() + 1e-7)
        pseudo_score_drop = (
            evidence_drop * pseudo_rank_weight
        ).sum() / (pseudo_rank_weight.sum() + 1e-7)

        def pseudo_type_metrics(type_index):
            type_mask = (batch["pseudo_a_type"] == type_index).to(
                pseudo_rank_weight.dtype
            )
            type_weight = pseudo_rank_weight * type_mask
            denominator = type_weight.sum() + 1e-7
            score_drop = (evidence_drop * type_weight).sum() / denominator
            pair_accuracy = (
                (evidence_drop > 0.0).to(evidence_drop.dtype) * type_weight
            ).sum() / denominator
            return score_drop, pair_accuracy

        pseudo_shift_drop, pseudo_shift_pair = pseudo_type_metrics(0)
        pseudo_shift_sign_drop, pseudo_shift_sign_pair = pseudo_type_metrics(1)

        edge_weight = batch["edge_weight"].view(-1)
        derived_evidence = (
            0.35 * physical_clean[:, self.pq_alignment_index]
            + 0.25 * physical_clean[:, self.segment_coherence_index]
            + 0.15 * physical_clean[:, self.activity_strength_index]
            + 0.25
            * physical_clean[:, self.edge_pair_index]
            * edge_weight
        ) / (0.75 + 0.25 * edge_weight + 1e-7)
        direct_evidence = physical_clean[:, self.storage_evidence_index]
        hierarchy_physical_weight = physical_weight[
            :, self.storage_evidence_index:self.storage_evidence_index + 1
        ]
        hierarchy_physical_loss = weighted_smooth_l1(
            direct_evidence.view(-1, 1),
            derived_evidence.view(-1, 1),
            hierarchy_physical_weight,
        )
        derived_behavior = (
            0.50 * direct_evidence
            + 0.25 * response_clean[:, self.response_confidence_index]
            + 0.25 * behavior_clean[:, self.policy_pattern_index]
        )
        direct_behavior = behavior_clean[:, self.storage_behavior_index]
        hierarchy_behavior_weight = behavior_weight[
            :, self.storage_behavior_index:self.storage_behavior_index + 1
        ]
        hierarchy_behavior_loss = weighted_smooth_l1(
            direct_behavior.view(-1, 1),
            derived_behavior.view(-1, 1),
            hierarchy_behavior_weight,
        )
        hierarchy_total_loss = 0.5 * (
            hierarchy_physical_loss + hierarchy_behavior_loss
        )
        evidence_formula_mae = torch.mean(torch.abs(direct_evidence - derived_evidence))
        behavior_formula_mae = torch.mean(torch.abs(direct_behavior - derived_behavior))

        zero = direct_evidence.new_tensor(0.0)
        attenuation_target_names = (
            "storage_evidence",
            "response_confidence",
            "storage_behavior",
        )
        attenuation_logs = {
            f"p_attenuation_{name}_pair_accuracy": zero
            for name in attenuation_target_names
        }
        attenuation_logs.update({
            f"p_attenuation_{name}_score_drop": zero
            for name in attenuation_target_names
        })
        attenuation_logs.update({
            f"p_attenuation_{name}_target_drop": zero
            for name in attenuation_target_names
        })
        attenuation_logs.update({
            f"p_attenuation_{name}_required_margin": zero
            for name in attenuation_target_names
        })
        attenuation_logs.update({
            f"p_attenuation_{name}_margin_accuracy": zero
            for name in attenuation_target_names
        })
        if (
            "physical_attenuation" in embeddings
            and "context_attenuation" in embeddings
            and "p_attenuation_index" in batch
        ):
            attenuation_index = batch["p_attenuation_index"]
            physical_attenuation = self._predict(
                self.physical_head, embeddings["physical_attenuation"]
            )
            response_attenuation = self._predict(
                self.response_head, embeddings["physical_attenuation"]
            )
            behavior_attenuation = self._predict(
                self.behavior_head, embeddings["context_attenuation"]
            )
            clean_core = torch.stack([
                physical_clean[attenuation_index, self.storage_evidence_index],
                response_clean[attenuation_index, self.response_confidence_index],
                behavior_clean[attenuation_index, self.storage_behavior_index],
            ], dim=1)
            attenuation_core = torch.stack([
                physical_attenuation[:, self.storage_evidence_index],
                response_attenuation[:, self.response_confidence_index],
                behavior_attenuation[:, self.storage_behavior_index],
            ], dim=1)
            attenuation_weight = (
                batch["attenuation_weight"][attenuation_index]
                * batch["sample_weight"][attenuation_index].view(-1, 1)
            )
            target_drop = batch["attenuation_target_drop"][attenuation_index]
            required_margin = torch.minimum(
                target_drop,
                torch.full_like(target_drop, self.p_attenuation_margin),
            )
            predicted_drop = clean_core - attenuation_core
            attenuation_violation = F.relu(required_margin - predicted_drop)
            p_attenuation_rank_loss = (
                attenuation_violation * attenuation_weight
            ).sum() / (attenuation_weight.sum() + 1e-7)
            p_attenuation_pair_accuracy = (
                (predicted_drop > 0.0).to(predicted_drop.dtype)
                * attenuation_weight
            ).sum() / (attenuation_weight.sum() + 1e-7)
            p_attenuation_score_drop = (
                predicted_drop * attenuation_weight
            ).sum() / (attenuation_weight.sum() + 1e-7)
            p_attenuation_target_drop = (
                target_drop * attenuation_weight
            ).sum() / (attenuation_weight.sum() + 1e-7)
            p_attenuation_required_margin = (
                required_margin * attenuation_weight
            ).sum() / (attenuation_weight.sum() + 1e-7)
            p_attenuation_margin_accuracy = (
                (predicted_drop >= required_margin).to(predicted_drop.dtype)
                * attenuation_weight
            ).sum() / (attenuation_weight.sum() + 1e-7)
            for target_index, target_name in enumerate(attenuation_target_names):
                target_weight = attenuation_weight[:, target_index]
                denominator = target_weight.sum() + 1e-7
                attenuation_logs[
                    f"p_attenuation_{target_name}_pair_accuracy"
                ] = (
                    (predicted_drop[:, target_index] > 0.0)
                    .to(predicted_drop.dtype)
                    * target_weight
                ).sum() / denominator
                attenuation_logs[
                    f"p_attenuation_{target_name}_score_drop"
                ] = (
                    predicted_drop[:, target_index] * target_weight
                ).sum() / denominator
                attenuation_logs[
                    f"p_attenuation_{target_name}_target_drop"
                ] = (
                    target_drop[:, target_index] * target_weight
                ).sum() / denominator
                attenuation_logs[
                    f"p_attenuation_{target_name}_required_margin"
                ] = (
                    required_margin[:, target_index] * target_weight
                ).sum() / denominator
                attenuation_logs[
                    f"p_attenuation_{target_name}_margin_accuracy"
                ] = (
                    (predicted_drop[:, target_index] >= required_margin[:, target_index])
                    .to(predicted_drop.dtype)
                    * target_weight
                ).sum() / denominator
            role_weight = (
                behavior_weight[attenuation_index, :3]
            )
            role_invariance_loss = weighted_smooth_l1(
                behavior_attenuation[:, :3],
                behavior_clean[attenuation_index, :3],
                role_weight,
            )
        else:
            p_attenuation_rank_loss = zero
            p_attenuation_pair_accuracy = zero
            p_attenuation_score_drop = zero
            p_attenuation_target_drop = zero
            p_attenuation_required_margin = zero
            p_attenuation_margin_accuracy = zero
            role_invariance_loss = zero

        prediction_consistency = 0.5 * (
            F.mse_loss(physical_clean, physical_augmented)
            + F.mse_loss(response_clean, response_augmented)
        )
        (
            regularization,
            embedding_std_min,
            embedding_std_mean,
            variance_floor_loss,
            covariance_loss,
        ) = embedding_regularization(
            embeddings["physical_clean"],
            embeddings["physical_augmented"],
            target_std=self.target_std,
            covariance_weight=self.reg_covariance_weight,
        )
        if self.lambda_anchor > 0.0:
            anchor_loss = F.smooth_l1_loss(
                embeddings["physical_clean"], batch["stage1_embedding"]
            )
        else:
            anchor_loss = embeddings["physical_clean"].new_tensor(0.0)

        warmup = min(1.0, max(0.0, epoch / max(self.warmup_epochs, 1)))
        total = warmup * (
            self.lambda_physical * physical_loss
            + self.lambda_response * response_loss
            + self.lambda_behavior * behavior_loss
            + self.lambda_transfer * transfer_loss
            + self.lambda_policy_mismatch * mismatch_loss
            + self.lambda_policy_rank * (
                policy_rank_loss + policy_drop_calibration_loss
            )
            + self.lambda_pseudo_rank * pseudo_rank_loss
            + self.lambda_hierarchy * hierarchy_total_loss
            + self.lambda_p_attenuation * p_attenuation_rank_loss
            + self.lambda_role_invariance * role_invariance_loss
        )
        total = (
            total
            + self.lambda_consistency * prediction_consistency
            + self.lambda_reg * regularization
            + self.lambda_anchor * anchor_loss
        )
        valid_transfer = (batch["transfer_weight"] > 0.0).to(transfer_score.dtype)
        valid_count = valid_transfer.sum()
        score_difference = transfer_score - mismatch_score
        policy_score_drop = (
            score_difference * valid_transfer
        ).sum() / (valid_count + 1e-7)
        policy_pair_accuracy = (
            (score_difference > 0.0).to(transfer_score.dtype) * valid_transfer
        ).sum() / (valid_count + 1e-7)
        logs = {
            "loss": total.detach(),
            "physical_loss": physical_loss.detach(),
            "response_loss": response_loss.detach(),
            "behavior_loss": behavior_loss.detach(),
            "transfer_loss": transfer_loss.detach(),
            "mismatch_loss": mismatch_loss.detach(),
            "policy_rank_loss": policy_rank_loss.detach(),
            "policy_drop_calibration_loss": policy_drop_calibration_loss.detach(),
            "policy_score_drop": policy_score_drop.detach(),
            "policy_pair_accuracy": policy_pair_accuracy.detach(),
            "pseudo_rank_loss": pseudo_rank_loss.detach(),
            "pseudo_score_drop": pseudo_score_drop.detach(),
            "pseudo_pair_accuracy": pseudo_pair_accuracy.detach(),
            "pseudo_shift_score_drop": pseudo_shift_drop.detach(),
            "pseudo_shift_pair_accuracy": pseudo_shift_pair.detach(),
            "pseudo_shift_sign_score_drop": pseudo_shift_sign_drop.detach(),
            "pseudo_shift_sign_pair_accuracy": pseudo_shift_sign_pair.detach(),
            "hierarchy_physical_loss": hierarchy_physical_loss.detach(),
            "hierarchy_behavior_loss": hierarchy_behavior_loss.detach(),
            "hierarchy_total_loss": hierarchy_total_loss.detach(),
            "evidence_formula_mae": evidence_formula_mae.detach(),
            "behavior_formula_mae": behavior_formula_mae.detach(),
            "p_attenuation_rank_loss": p_attenuation_rank_loss.detach(),
            "p_attenuation_pair_accuracy": p_attenuation_pair_accuracy.detach(),
            "p_attenuation_score_drop": p_attenuation_score_drop.detach(),
            "p_attenuation_target_drop": p_attenuation_target_drop.detach(),
            "p_attenuation_required_margin": p_attenuation_required_margin.detach(),
            "p_attenuation_margin_accuracy": p_attenuation_margin_accuracy.detach(),
            "role_invariance_loss": role_invariance_loss.detach(),
            "consistency_loss": prediction_consistency.detach(),
            "regularization": regularization.detach(),
            "variance_floor_loss": variance_floor_loss.detach(),
            "covariance_loss": covariance_loss.detach(),
            "anchor_loss": anchor_loss.detach(),
            "embedding_std_min": embedding_std_min.detach(),
            "embedding_std_mean": embedding_std_mean.detach(),
            "embedding_norm_mean": embeddings["physical_clean"].detach().norm(dim=1).mean(),
            "warmup": total.detach().new_tensor(warmup),
        }
        logs.update({name: value.detach() for name, value in attenuation_logs.items()})
        outputs = {
            "physical": physical_clean,
            "response": response_clean,
            "behavior": behavior_clean,
            "transfer_behavior": behavior_transfer,
            "mismatch_behavior": behavior_mismatch,
        }
        return total, logs, outputs


class Stage2Dataset(Dataset):
    def __init__(self, inputs, targets, transfer_inputs, transfer_targets,
                 mismatch_inputs, mismatch_targets, transfer_weights, loss_weights,
                 attenuation_pq_inputs=None, attenuation_target_drop=None,
                 attenuation_weights=None, attenuation_alphas=None,
                 edge_weights=None):
        self.inputs = _owned_float_tensor(inputs)
        self.targets = {
            key: _owned_float_tensor(targets[key]) for key in STAGE2_TARGET_KEYS
        }
        self.transfer_inputs = _owned_float_tensor(transfer_inputs)
        self.transfer_targets = {
            key: _owned_float_tensor(transfer_targets[key]) for key in STAGE2_TARGET_KEYS
        }
        self.mismatch_inputs = _owned_float_tensor(mismatch_inputs)
        self.mismatch_targets = {
            key: _owned_float_tensor(mismatch_targets[key])
            for key in ("behavior_targets", "behavior_weights")
        }
        self.transfer_weights = _owned_float_tensor(transfer_weights)
        self.loss_weights = _owned_float_tensor(loss_weights)
        sample_count = len(self.inputs)
        if attenuation_pq_inputs is None:
            attenuation_pq_inputs = self.inputs[:, :2, :]
        if attenuation_target_drop is None:
            attenuation_target_drop = torch.zeros(sample_count, 3)
        if attenuation_weights is None:
            attenuation_weights = torch.zeros(sample_count, 3)
        if attenuation_alphas is None:
            attenuation_alphas = torch.ones(sample_count)
        if edge_weights is None:
            edge_weights = torch.ones(sample_count)
        self.attenuation_pq_inputs = _owned_float_tensor(attenuation_pq_inputs)
        self.attenuation_target_drop = _owned_float_tensor(attenuation_target_drop)
        self.attenuation_weights = _owned_float_tensor(attenuation_weights)
        self.attenuation_alphas = _owned_float_tensor(attenuation_alphas)
        self.edge_weights = _owned_float_tensor(edge_weights)
        self.stage1_embeddings = None
        self.frozen_embeddings = {}

    def set_stage1_embeddings(self, embeddings):
        embeddings = _owned_float_tensor(embeddings)
        if len(embeddings) != len(self.inputs):
            raise ValueError("Stage-1 锚点数量与 Stage-2 样本数量不一致")
        self.stage1_embeddings = embeddings

    def set_frozen_embeddings(self, embeddings):
        if any(len(values) != len(self.inputs) for values in embeddings.values()):
            raise ValueError("冻结视图特征数量与 Stage-2 样本数量不一致")
        self.frozen_embeddings = {
            name: _owned_float_tensor(values) for name, values in embeddings.items()
        }

    @property
    def device(self):
        return self.inputs.device

    def storage_nbytes(self):
        tensors = [
            self.inputs,
            self.transfer_inputs,
            self.mismatch_inputs,
            self.transfer_weights,
            self.loss_weights,
            self.attenuation_pq_inputs,
            self.attenuation_target_drop,
            self.attenuation_weights,
            self.attenuation_alphas,
            self.edge_weights,
        ]
        tensors.extend(self.targets.values())
        tensors.extend(self.transfer_targets.values())
        tensors.extend(self.mismatch_targets.values())
        tensors.extend(self.frozen_embeddings.values())
        if self.stage1_embeddings is not None:
            tensors.append(self.stage1_embeddings)
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def to(self, device):
        if self.stage1_embeddings is None:
            raise RuntimeError("Stage-1 锚点尚未预计算")
        self.inputs = self.inputs.to(device)
        self.transfer_inputs = self.transfer_inputs.to(device)
        self.mismatch_inputs = self.mismatch_inputs.to(device)
        self.targets = {name: value.to(device) for name, value in self.targets.items()}
        self.transfer_targets = {
            name: value.to(device) for name, value in self.transfer_targets.items()
        }
        self.mismatch_targets = {
            name: value.to(device) for name, value in self.mismatch_targets.items()
        }
        self.transfer_weights = self.transfer_weights.to(device)
        self.loss_weights = self.loss_weights.to(device)
        self.attenuation_pq_inputs = self.attenuation_pq_inputs.to(device)
        self.attenuation_target_drop = self.attenuation_target_drop.to(device)
        self.attenuation_weights = self.attenuation_weights.to(device)
        self.attenuation_alphas = self.attenuation_alphas.to(device)
        self.edge_weights = self.edge_weights.to(device)
        self.stage1_embeddings = self.stage1_embeddings.to(device)
        self.frozen_embeddings = {
            name: value.to(device) for name, value in self.frozen_embeddings.items()
        }
        return self

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        if self.stage1_embeddings is None:
            raise RuntimeError("Stage-1 锚点尚未预计算")
        return self.get_batch(index)


    def get_batch(self, index):
        if self.stage1_embeddings is None:
            raise RuntimeError("Stage-1 锚点尚未预计算")
        batch = {
            "clean_inputs": self.inputs[index],
            "transfer_inputs": self.transfer_inputs[index],
            "mismatch_inputs": self.mismatch_inputs[index],
            "physical_target": self.targets["physical_targets"][index],
            "physical_weight": self.targets["physical_weights"][index],
            "response_target": self.targets["response_targets"][index],
            "response_weight": self.targets["response_weights"][index],
            "behavior_target": self.targets["behavior_targets"][index],
            "behavior_weight": self.targets["behavior_weights"][index],
            "transfer_physical_target": self.transfer_targets["physical_targets"][index],
            "transfer_physical_weight": self.transfer_targets["physical_weights"][index],
            "transfer_response_target": self.transfer_targets["response_targets"][index],
            "transfer_response_weight": self.transfer_targets["response_weights"][index],
            "transfer_behavior_target": self.transfer_targets["behavior_targets"][index],
            "transfer_behavior_weight": self.transfer_targets["behavior_weights"][index],
            "mismatch_behavior_target": self.mismatch_targets["behavior_targets"][index],
            "mismatch_behavior_weight": self.mismatch_targets["behavior_weights"][index],
            "transfer_weight": self.transfer_weights[index],
            "sample_weight": self.loss_weights[index],
            "stage1_embedding": self.stage1_embeddings[index],
            "attenuation_pq_inputs": self.attenuation_pq_inputs[index],
            "attenuation_target_drop": self.attenuation_target_drop[index],
            "attenuation_weight": self.attenuation_weights[index],
            "attenuation_alpha": self.attenuation_alphas[index],
            "edge_weight": self.edge_weights[index],
        }
        batch.update({
            f"frozen_{name}_embedding": values[index]
            for name, values in self.frozen_embeddings.items()
        })
        return batch


class TensorBatchLoader:
    def __init__(self, dataset, batch_size, sampler=None, drop_last=False):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.sampler = sampler
        self.drop_last = bool(drop_last)

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.sampler is None:
            indices = torch.arange(len(self.dataset), device=self.dataset.device)
        else:
            indices = torch.as_tensor(
                list(self.sampler), dtype=torch.long, device=self.dataset.device
            )
        sample_count = len(indices)
        if self.drop_last:
            sample_count -= sample_count % self.batch_size
        for start in range(0, sample_count, self.batch_size):
            yield self.dataset.get_batch(indices[start:start + self.batch_size])


def maybe_enable_gpu_resident_loaders(
    train_loader, val_loader, device, args, data_info
):
    data_info["loader_mode"] = "pytorch-dataloader"
    data_info["gpu_resident_bytes"] = 0
    if device.type != "cuda" or not args.gpu_data_cache:
        return train_loader, val_loader

    required_bytes = (
        train_loader.dataset.storage_nbytes()
        + val_loader.dataset.storage_nbytes()
    )
    free_bytes, _ = torch.cuda.mem_get_info(device)
    cache_budget = int(free_bytes * args.gpu_cache_max_fraction)
    if required_bytes > cache_budget:
        print(
            f"[Speed] GPU data cache skipped | required={required_bytes / 2**20:.1f} MB "
            f"budget={cache_budget / 2**20:.1f} MB"
        )
        return train_loader, val_loader

    train_dataset = train_loader.dataset.to(device)
    val_dataset = val_loader.dataset.to(device)
    resident_train_loader = TensorBatchLoader(
        train_dataset,
        train_loader.batch_size,
        sampler=train_loader.sampler,
        drop_last=train_loader.drop_last,
    )
    resident_val_loader = TensorBatchLoader(
        val_dataset,
        val_loader.batch_size,
        drop_last=val_loader.drop_last,
    )
    data_info["loader_mode"] = "gpu-resident-index-select"
    data_info["gpu_resident_bytes"] = int(required_bytes)
    gc.collect()
    torch.cuda.empty_cache()
    print(
        f"[Speed] GPU data cache enabled | size={required_bytes / 2**20:.1f} MB | "
        "batch=index_select"
    )
    return resident_train_loader, resident_val_loader


def _owned_float_tensor(values):
    if torch.is_tensor(values):
        return values.detach().clone().to(dtype=torch.float32)
    return torch.from_numpy(np.array(values, dtype=np.float32, copy=True))


def subset_array(values, indices):
    if torch.is_tensor(values):
        return values.index_select(0, torch.as_tensor(indices, dtype=torch.long))
    return np.asarray(values)[indices]


def save_rows_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_transfer_target_cache(path, data, transfer, target_config, transfer_config,
                                force=False):
    path = Path(path)
    metadata = {
        "cache_version": TRANSFER_TARGET_CACHE_VERSION,
        "target_pipeline_version": TARGET_PIPELINE_VERSION,
        "policy_transfer_version": POLICY_TRANSFER_VERSION,
        "data_pipeline_version": data["pipeline_version"],
        "data_sha256": data["data_sha256"],
        "split_manifest_sha256": data["split_manifest_sha256"],
        "sample_count": int(len(data["meta"])),
        "target_config": asdict(target_config),
        "transfer_config": asdict(transfer_config),
    }
    if path.is_file() and not force:
        with np.load(path, allow_pickle=False) as cache:
            cached_metadata = json.loads(str(cache["metadata_json"].item()))
            if cached_metadata == metadata:
                positive = {
                    key: cache[f"positive__{key}"].copy()
                    for key in STAGE2_TARGET_KEYS
                }
                mismatch = {
                    key: cache[f"mismatch__{key}"].copy()
                    for key in STAGE2_TARGET_KEYS
                }
                return positive, mismatch

    positive = compute_stage2_targets(
        transfer["positive_p_raw"],
        transfer["positive_q_raw"],
        transfer["target_role_seq"],
        config=target_config,
    )
    mismatch = compute_stage2_targets(
        transfer["positive_p_raw"],
        transfer["positive_q_raw"],
        transfer["mismatch_role_seq"],
        config=target_config,
    )
    arrays = {"metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False))}
    arrays.update({
        f"positive__{key}": positive[key] for key in STAGE2_TARGET_KEYS
    })
    arrays.update({
        f"mismatch__{key}": mismatch[key] for key in STAGE2_TARGET_KEYS
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary_path, **arrays)
    os.replace(temporary_path, path)
    return positive, mismatch


def build_p_attenuation_cache(path, data, targets, target_config, alpha_values,
                              seed, force=False):
    path = Path(path)
    metadata = {
        "cache_version": P_ATTENUATION_CACHE_VERSION,
        "data_sha256": data["data_sha256"],
        "split_manifest_sha256": data["split_manifest_sha256"],
        "target_pipeline_version": TARGET_PIPELINE_VERSION,
        "target_config": asdict(target_config),
        "q_global_scale": float(data["normalization"]["q_global_scale"]),
        "alpha_values": [float(value) for value in alpha_values],
        "seed": int(seed),
        "sample_count": int(len(data["P_raw"])),
    }
    if path.is_file() and not force:
        with np.load(path, allow_pickle=False) as payload:
            cached_metadata = json.loads(str(payload["metadata_json"].item()))
            if cached_metadata == metadata:
                return {
                    "pq_inputs": payload["pq_inputs"].copy(),
                    "target_drop": payload["target_drop"].copy(),
                    "constraint_weight": payload["constraint_weight"].copy(),
                    "alpha": payload["alpha"].copy(),
                    "metadata": cached_metadata,
                }

    rng = np.random.default_rng(int(seed) + 2718)
    alpha = rng.choice(
        np.asarray(alpha_values, dtype=np.float32),
        size=len(data["P_raw"]),
        replace=True,
    ).astype(np.float32)
    p_raw = np.asarray(data["P_raw"], dtype=np.float32)
    q_raw = np.asarray(data["Q_raw"], dtype=np.float32)
    p_attenuated = p_raw - q_raw + alpha[:, None] * q_raw
    q_global_scale = float(data["normalization"]["q_global_scale"])
    pq_inputs = np.empty((len(p_raw), 2, p_raw.shape[1]), dtype=np.float32)
    for sample_index in range(len(p_raw)):
        p_scaled, q_scaled, _ = scale_pq(
            p_attenuated[sample_index],
            q_raw[sample_index],
            q_global_scale=q_global_scale,
        )
        pq_inputs[sample_index, 0] = p_scaled
        pq_inputs[sample_index, 1] = q_scaled

    role_sequence = data["X"][:, 3, :].cpu().numpy()
    attenuated_targets = compute_stage2_targets(
        p_attenuated,
        q_raw,
        role_sequence,
        config=target_config,
    )
    target_drop = np.column_stack([
        targets["physical_targets"][:, PHYSICAL_TARGET_NAMES.index("storage_evidence")]
        - attenuated_targets["physical_targets"][:, PHYSICAL_TARGET_NAMES.index("storage_evidence")],
        targets["response_targets"][:, RESPONSE_TARGET_NAMES.index("response_confidence")]
        - attenuated_targets["response_targets"][:, RESPONSE_TARGET_NAMES.index("response_confidence")],
        targets["behavior_targets"][:, BEHAVIOR_TARGET_NAMES.index("storage_behavior")]
        - attenuated_targets["behavior_targets"][:, BEHAVIOR_TARGET_NAMES.index("storage_behavior")],
    ]).astype(np.float32)
    constraint_weight = np.column_stack([
        np.minimum(
            targets["physical_weights"][:, PHYSICAL_TARGET_NAMES.index("storage_evidence")],
            attenuated_targets["physical_weights"][:, PHYSICAL_TARGET_NAMES.index("storage_evidence")],
        ),
        np.minimum(
            targets["response_weights"][:, RESPONSE_TARGET_NAMES.index("response_confidence")],
            attenuated_targets["response_weights"][:, RESPONSE_TARGET_NAMES.index("response_confidence")],
        ),
        np.minimum(
            targets["behavior_weights"][:, BEHAVIOR_TARGET_NAMES.index("storage_behavior")],
            attenuated_targets["behavior_weights"][:, BEHAVIOR_TARGET_NAMES.index("storage_behavior")],
        ),
    ]).astype(np.float32)
    constraint_weight *= (target_drop > 1e-5).astype(np.float32)
    target_drop = np.clip(target_drop, 0.0, 1.0).astype(np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        temporary_path,
        pq_inputs=pq_inputs,
        target_drop=target_drop,
        constraint_weight=constraint_weight,
        alpha=alpha,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    os.replace(temporary_path, path)
    return {
        "pq_inputs": pq_inputs,
        "target_drop": target_drop,
        "constraint_weight": constraint_weight,
        "alpha": alpha,
        "metadata": metadata,
    }


def build_dataloaders_st2(args):
    data = load_storage_data(
        data_path=args.data_path,
        train_ratio=args.train_ratio,
        time_type=args.time_type,
        seed=args.seed,
        split_manifest_path=args.split_manifest,
        user_col=args.user_col,
        policy_name=args.policy_name,
    )
    if args.policy_name != "old":
        raise ValueError("Stage-2 真实 P/Q 必须按旧政策读取，迁移样本由同步重构生成")

    target_config = TeacherTargetConfig(
        activity_ratio=args.activity_ratio,
        min_activity_to_load_ratio=args.min_activity_to_load_ratio,
        alignment_max_lag=args.alignment_max_lag,
        response_noise_floor_ratio=args.response_noise_floor_ratio,
        min_run_points=args.min_run_points,
        edge_window_points=args.edge_window_points,
        strength_reference_ratio=args.strength_reference_ratio,
        role_smooth_radius=args.role_smooth_radius,
    )
    target_cache_path = args.save_dir / "teacher_targets_v5.npz"
    targets = build_stage2_target_cache(
        data,
        target_cache_path,
        config=target_config,
        force=args.rebuild_target_cache,
    )
    attenuation = build_p_attenuation_cache(
        args.save_dir / "p_attenuation_v1.npz",
        data,
        targets,
        target_config,
        alpha_values=getattr(args, "p_attenuation_alphas", [0.25, 0.50, 0.75]),
        seed=args.seed,
        force=(
            args.rebuild_target_cache
            or getattr(args, "rebuild_p_attenuation_cache", False)
        ),
    )
    transfer_config = PolicyTransferConfig(
        activity_ratio=args.activity_ratio,
        min_activity_to_load_ratio=args.min_activity_to_load_ratio,
        min_activity_coverage=args.transfer_min_activity_coverage,
        min_energy_retention=args.transfer_min_energy_retention,
        min_target_pattern=args.transfer_min_target_pattern,
        random_shift_probability=args.transfer_random_shift_probability,
        shift_step=args.transfer_shift_step,
        min_shift_points=args.transfer_min_shift_points,
    )
    transfer = build_policy_transfer_cache(
        data,
        args.save_dir / "policy_transfer_v3.npz",
        config=transfer_config,
        force=args.rebuild_transfer_cache,
        seed=args.seed,
    )
    transfer_targets, mismatch_targets = build_transfer_target_cache(
        args.save_dir / "teacher_transfer_targets_v1.npz",
        data,
        transfer,
        target_config,
        transfer_config,
        force=(
            args.rebuild_target_cache
            or args.rebuild_transfer_cache
            or args.rebuild_transfer_target_cache
        ),
    )
    if not np.allclose(
        transfer_targets["physical_targets"],
        mismatch_targets["physical_targets"],
        atol=1e-6,
    ):
        raise RuntimeError("仅替换政策角色时，物理目标不应变化")
    mismatch_inputs = transfer["positive_inputs"].copy()
    mismatch_inputs[:, 3, :] = transfer["mismatch_role_seq"]
    mismatch_inputs[:, 4, :] = transfer["mismatch_role_jump"]

    summary_rows = []
    for source, target_data in (("real_old", targets), ("relocated", transfer_targets)):
        for row in summarize_stage2_targets(target_data):
            summary_rows.append({"source": source, **row})
    save_rows_csv(args.save_dir / "teacher_target_summary.csv", summary_rows)

    train_indices = data["split_indices"]["train"]
    val_indices = data["split_indices"]["val"]
    if len(train_indices) == 0 or len(val_indices) == 0:
        raise ValueError("Stage-2 需要非空的 train 和 val 用户切分")

    metadata = data["meta"]
    train_sample_weight = metadata.iloc[train_indices]["sample_weight"].to_numpy(np.float32)
    val_sample_weight = metadata.iloc[val_indices]["sample_weight"].to_numpy(np.float32)
    sampling_weight = np.sqrt(np.maximum(train_sample_weight, 1e-6))
    train_loss_weight = np.sqrt(np.maximum(train_sample_weight, 1e-6))

    augment_config = Stage2AugmentConfig(
        scale_min=args.augment_scale_min,
        scale_max=args.augment_scale_max,
        smooth_prob=args.augment_smooth_prob,
        smooth_window=args.augment_smooth_window,
        noise_std=args.augment_noise_std,
    )
    def selected_targets(target_data, indices):
        return {
            key: subset_array(target_data[key], indices)
            for key in STAGE2_TARGET_KEYS
        }

    def build_dataset(indices, loss_weights):
        return Stage2Dataset(
            subset_array(data["X"], indices),
            selected_targets(targets, indices),
            subset_array(transfer["positive_inputs"], indices),
            selected_targets(transfer_targets, indices),
            subset_array(mismatch_inputs, indices),
            selected_targets(mismatch_targets, indices),
            subset_array(transfer["transfer_weight"], indices),
            loss_weights,
            attenuation_pq_inputs=subset_array(attenuation["pq_inputs"], indices),
            attenuation_target_drop=subset_array(attenuation["target_drop"], indices),
            attenuation_weights=subset_array(attenuation["constraint_weight"], indices),
            attenuation_alphas=subset_array(attenuation["alpha"], indices),
            edge_weights=subset_array(targets["auxiliary"]["edge_weight"], indices),
        )

    train_dataset = build_dataset(train_indices, train_loss_weight)
    val_dataset = build_dataset(val_indices, val_sample_weight)
    sampler = WeightedRandomSampler(
        torch.as_tensor(sampling_weight, dtype=torch.double),
        num_samples=len(train_dataset),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader_options = {
        "num_workers": args.num_workers,
        "pin_memory": args.cuda,
    }
    if args.num_workers > 0:
        loader_options.update({
            "persistent_workers": True,
            "prefetch_factor": args.prefetch_factor,
        })
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=len(train_dataset) >= args.batch_size,
        **loader_options,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **loader_options,
    )
    attenuation_target_names = (
        "storage_evidence",
        "response_confidence",
        "storage_behavior",
    )

    def attenuation_summary(indices):
        weights = attenuation["constraint_weight"][indices]
        drops = attenuation["target_drop"][indices]
        summary = {}
        for target_index, target_name in enumerate(attenuation_target_names):
            valid = weights[:, target_index] > 0.0
            summary[target_name] = {
                "valid_ratio": float(valid.mean()),
                "target_drop_mean": float(drops[valid, target_index].mean())
                if valid.any() else 0.0,
            }
        return summary

    data_info = {
        "pipeline_version": data["pipeline_version"],
        "target_pipeline_version": TARGET_PIPELINE_VERSION,
        "stage2_objective_version": STAGE2_OBJECTIVE_VERSION,
        "stage2_input_semantics": "physical-pq-only_behavior-pq-relative-role_no-season",
        "data_path": data["data_path"],
        "data_sha256": data["data_sha256"],
        "split_manifest_path": data["split_manifest_path"],
        "split_manifest_sha256": data["split_manifest_sha256"],
        "split_strategy": data["split_strategy"],
        "normalization": data["normalization"],
        "train_samples": int(len(train_indices)),
        "val_samples": int(len(val_indices)),
        "test_samples": int(len(data["split_indices"]["test"])),
        "train_users": int(metadata.iloc[train_indices]["user_id"].nunique()),
        "val_users": int(metadata.iloc[val_indices]["user_id"].nunique()),
        "test_users": int(metadata.iloc[data["split_indices"]["test"]]["user_id"].nunique()),
        "target_config": asdict(target_config),
        "augment_config": asdict(augment_config),
        "loader_workers": int(args.num_workers),
        "loader_prefetch_factor": int(args.prefetch_factor),
        "policy_transfer_version": POLICY_TRANSFER_VERSION,
        "transfer_target_cache_version": TRANSFER_TARGET_CACHE_VERSION,
        "p_attenuation_cache_version": P_ATTENUATION_CACHE_VERSION,
        "p_attenuation_alphas": [
            float(value) for value in getattr(
                args, "p_attenuation_alphas", [0.25, 0.50, 0.75]
            )
        ],
        "p_attenuation_valid_ratio_train": float(
            (attenuation["constraint_weight"][train_indices].sum(axis=1) > 0.0).mean()
        ),
        "p_attenuation_valid_ratio_val": float(
            (attenuation["constraint_weight"][val_indices].sum(axis=1) > 0.0).mean()
        ),
        "p_attenuation_target_summary_train": attenuation_summary(train_indices),
        "p_attenuation_target_summary_val": attenuation_summary(val_indices),
        "policy_transfer_config": asdict(transfer_config),
        "transfer_acceptance_train": float(transfer["accepted"][train_indices].mean()),
        "transfer_acceptance_val": float(transfer["accepted"][val_indices].mean()),
        "transfer_weight_mean_train": float(transfer["transfer_weight"][train_indices].mean()),
        "user_balance": "sqrt sampler × sqrt loss weight",
        "physical_policy_input": "season_flag, price_role_seq and price_role_jump masked to zero",
        "behavior_policy_input": "season_flag masked; relative price role and role jump retained",
        "policy_transfer": "B=P-Q; relocate Q segments; P_syn=B+Q_syn; normalize jointly",
        "pseudo_a": "Q shift/sign mismatch for teacher ranking only",
        "p_attenuation": "B=P-Q; attenuate only P response while retaining full Q",
    }
    return train_loader, val_loader, data_info


def model_kwargs_from_args(args):
    return {
        "input_channels": 5,
        "nhid": args.nhid,
        "levels": args.levels,
        "ksize": args.ksize,
        "dropout": args.dropout,
        "lstm_hidden": args.lstm_hidden,
        "lstm_layers": args.lstm_layers,
        "embedding_dim": args.embedding_dim,
        "proj_dim": args.proj_dim,
    }


def load_stage1_model(model, checkpoint_path, device):
    checkpoint_path = resolve_project_path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Stage-1 checkpoint 不存在：{checkpoint_path}")
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location=device)
    architecture_version = checkpoint.get("model_arch_version")
    if architecture_version != TCN_LSTM_ARCH_VERSION:
        raise RuntimeError(
            "Stage-1 checkpoint 架构不兼容，请使用当前代码重新训练 Stage-1："
            f"{architecture_version!r} != {TCN_LSTM_ARCH_VERSION!r}"
        )
    objective_version = checkpoint.get("objective_version")
    if objective_version != REQUIRED_STAGE1_OBJECTIVE_VERSION:
        raise RuntimeError(
            "Stage-1 checkpoint 训练版本不兼容，请使用当前代码重新训练 Stage-1："
            f"{objective_version!r} != {REQUIRED_STAGE1_OBJECTIVE_VERSION!r}"
        )
    saved_args = checkpoint.get("args", {}) or {}
    expected_config = {
        "nhid": model.backbone.tcn.network[0].conv1.out_channels,
        "levels": len(model.backbone.tcn.network),
        "lstm_hidden": model.backbone.lstm.hidden_size,
        "lstm_layers": model.backbone.lstm.num_layers,
        "embedding_dim": model.backbone.linear.out_features,
    }
    for name, expected_value in expected_config.items():
        saved_value = saved_args.get(name)
        if saved_value is not None and int(saved_value) != int(expected_value):
            raise RuntimeError(
                f"Stage-1 checkpoint 参数 {name}={saved_value}，"
                f"当前 Stage-2 需要 {expected_value}"
            )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected, skipped = load_compatible_state_dict(model, state_dict)
    if unexpected:
        raise RuntimeError(f"Stage-1 checkpoint 存在无法处理的键：{unexpected}")
    loaded_backbone = sum(1 for name in state_dict if name.startswith("backbone."))
    if loaded_backbone == 0:
        raise RuntimeError("Stage-1 checkpoint 未包含 backbone 权重")
    print(f"[Stage-1] loaded={checkpoint_path}")
    print(f"[Stage-1] missing={len(missing)}, compatible-adjustments={len(skipped)}")
    return checkpoint


def build_model_and_optimizer(args, device):
    model_kwargs = model_kwargs_from_args(args)
    model = EncoderWithHead(**model_kwargs).to(device)
    load_stage1_model(model, args.phase1_ckpt, device)
    model.backbone.lstm.flatten_parameters()

    stage1_teacher = EncoderWithHead(**model_kwargs).to(device)
    load_stage1_model(stage1_teacher, args.phase1_ckpt, device)
    stage1_teacher.backbone.lstm.flatten_parameters()
    stage1_teacher.eval()
    for parameter in stage1_teacher.parameters():
        parameter.requires_grad_(False)

    for parameter in model.proj_head.parameters():
        parameter.requires_grad_(False)

    backbone_parameters = list(model.backbone.parameters())
    head_parameters = (
        list(model.physical_head.parameters())
        + list(model.response_head.parameters())
        + list(model.behavior_head.parameters())
    )
    parameter_groups = [
        {"params": backbone_parameters, "lr": args.lr_backbone, "weight_decay": args.weight_decay},
        {"params": head_parameters, "lr": args.lr_head, "weight_decay": args.weight_decay},
    ]
    optimizer_options = {"fused": True} if args.fused_adam else {}
    try:
        optimizer = optim.Adam(parameter_groups, **optimizer_options)
    except (RuntimeError, TypeError) as error:
        if not args.fused_adam:
            raise
        args.fused_adam = False
        print(f"[Speed] fused Adam unavailable, fallback to default Adam: {error}")
        optimizer = optim.Adam(parameter_groups)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=args.lr_patience,
        threshold=args.best_min_delta,
        threshold_mode="abs",
    )
    criterion = Stage2TeacherLoss(
        model.physical_head,
        model.response_head,
        model.behavior_head,
        lambda_physical=args.lambda_physical,
        lambda_response=args.lambda_response,
        lambda_behavior=args.lambda_behavior,
        lambda_transfer=args.lambda_transfer,
        lambda_policy_mismatch=args.lambda_policy_mismatch,
        lambda_policy_rank=args.lambda_policy_rank,
        policy_margin=args.policy_margin,
        lambda_pseudo_rank=args.lambda_pseudo_rank,
        pseudo_margin=args.pseudo_margin,
        lambda_hierarchy=getattr(args, "lambda_hierarchy", 0.03),
        lambda_p_attenuation=getattr(args, "lambda_p_attenuation", 0.05),
        p_attenuation_margin=getattr(args, "p_attenuation_margin", 0.015),
        lambda_role_invariance=getattr(args, "lambda_role_invariance", 0.01),
        lambda_consistency=args.lambda_consistency,
        lambda_reg=args.lambda_reg,
        lambda_anchor=args.lambda_anchor,
        warmup_epochs=args.warmup_epochs,
        target_std=args.target_std,
        reg_covariance_weight=args.reg_covariance_weight,
    ).to(device)
    return model, criterion, optimizer, scheduler, model_kwargs, stage1_teacher


def move_batch_to_device(batch, device):
    if batch and next(iter(batch.values())).device == device:
        return batch
    return {
        name: value.to(device, non_blocking=True)
        for name, value in batch.items()
    }


def autocast_context(device, enabled):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def create_grad_scaler(enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.inference_mode()
def precompute_stage1_caches(
    stage1_teacher, dataset, device, batch_size, cache_anchor, cache_frozen_views,
    amp=False, cache_p_attenuation=False,
):
    stage1_teacher.eval()
    anchor_outputs = []
    frozen_names = ["clean", "transfer", "pseudo_shift", "pseudo_shift_sign"]
    if cache_p_attenuation:
        frozen_names.append("p_attenuation")
    frozen_outputs = {name: [] for name in frozen_names}
    for start in range(0, len(dataset), batch_size):
        inputs = dataset.inputs[start:start + batch_size].to(
            device, non_blocking=device.type == "cuda"
        )
        clean_inputs = mask_physical_context(inputs)
        if cache_anchor:
            anchor_outputs.append(
                stage1_teacher.encode(clean_inputs).float().cpu()
            )
        if cache_frozen_views:
            transfer_inputs = dataset.transfer_inputs[start:start + batch_size].to(
                device, non_blocking=device.type == "cuda"
            )
            views = [
                clean_inputs,
                mask_physical_context(transfer_inputs),
                mask_physical_context(make_pseudo_a_variant(inputs, sign_flip=False)),
                mask_physical_context(make_pseudo_a_variant(inputs, sign_flip=True)),
            ]
            if cache_p_attenuation:
                attenuation_pq_inputs = dataset.attenuation_pq_inputs[
                    start:start + batch_size
                ].to(device, non_blocking=device.type == "cuda")
                views.append(mask_physical_context(build_p_attenuation_inputs(
                    inputs, attenuation_pq_inputs
                )))
            with autocast_context(device, amp):
                embeddings = stage1_teacher.encode(torch.cat(views, dim=0)).float()
            chunks = torch.split(
                embeddings, [view.size(0) for view in views], dim=0
            )
            for name, values in zip(frozen_outputs, chunks):
                frozen_outputs[name].append(values.cpu())

    if cache_anchor:
        dataset.set_stage1_embeddings(torch.cat(anchor_outputs, dim=0))
    else:
        dataset.set_stage1_embeddings(
            torch.zeros(len(dataset), stage1_teacher.backbone.linear.out_features)
        )
    if cache_frozen_views:
        dataset.set_frozen_embeddings({
            name: torch.cat(values, dim=0)
            for name, values in frozen_outputs.items()
        })


def prepare_stage1_anchor_cache(stage1_teacher, train_loader, val_loader, device, args):
    cache_anchor = args.lambda_anchor > 0.0
    cache_frozen_views = (
        args.cache_frozen_views and args.freeze_backbone_epochs > 0
    )
    cache_p_attenuation = (
        getattr(args, "lambda_p_attenuation", 0.05) > 0.0
        or getattr(args, "lambda_role_invariance", 0.01) > 0.0
    )
    cache_multiplier = 2 if cache_frozen_views else 4
    cache_batch_size = max(args.batch_size * cache_multiplier, args.batch_size)
    for loader in (train_loader, val_loader):
        precompute_stage1_caches(
            stage1_teacher,
            loader.dataset,
            device,
            cache_batch_size,
            cache_anchor=cache_anchor,
            cache_frozen_views=cache_frozen_views,
            amp=args.amp,
            cache_p_attenuation=cache_p_attenuation,
        )
    print(
        f"[Speed] Stage-1 cache ready | anchor={cache_anchor} | "
        f"frozen_views={cache_frozen_views} | batch_size={cache_batch_size}"
    )


def augment_stage2_batch(batch, config):
    batch["augmented_inputs"] = augment_sync_intensity_torch(
        batch["clean_inputs"],
        scale_min=config.scale_min,
        scale_max=config.scale_max,
        smooth_prob=config.smooth_prob,
        smooth_window=config.smooth_window,
        noise_std=config.noise_std,
    )
    return batch


def build_p_attenuation_inputs(clean_inputs, attenuation_pq_inputs, index=None):
    selected = clean_inputs if index is None else clean_inputs[index]
    selected_pq = (
        attenuation_pq_inputs
        if index is None
        else attenuation_pq_inputs[index]
    )
    attenuated = selected.clone()
    attenuated[:, :2, :] = selected_pq
    return attenuated


def select_constraint_indices(batch_size, fraction, device):
    fraction = float(fraction)
    if fraction <= 0.0 or batch_size <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if fraction >= 1.0:
        return torch.arange(batch_size, dtype=torch.long, device=device)
    selected_count = max(1, int(round(batch_size * fraction)))
    return torch.randperm(batch_size, device=device)[:selected_count]


def encode_stage2_views(
    model, batch, include_augmented=True, use_frozen_cache=False,
    p_attenuation_fraction=0.0,
):
    attenuation_index = select_constraint_indices(
        batch["clean_inputs"].size(0),
        p_attenuation_fraction,
        batch["clean_inputs"].device,
    )
    if len(attenuation_index) > 0:
        batch["p_attenuation_index"] = attenuation_index
    if use_frozen_cache:
        required = (
            "frozen_clean_embedding",
            "frozen_transfer_embedding",
            "frozen_pseudo_shift_embedding",
            "frozen_pseudo_shift_sign_embedding",
        )
        missing = [name for name in required if name not in batch]
        if missing:
            raise RuntimeError(f"冻结视图缓存缺失：{missing}")
        batch_size = batch["clean_inputs"].size(0)
        midpoint = batch_size // 2
        pseudo_type = torch.zeros(
            batch_size, device=batch["clean_inputs"].device, dtype=torch.long
        )
        pseudo_type[midpoint:] = 1
        batch["pseudo_a_type"] = pseudo_type
        pseudo_embedding = batch["frozen_pseudo_shift_embedding"].clone()
        pseudo_embedding[midpoint:] = batch[
            "frozen_pseudo_shift_sign_embedding"
        ][midpoint:]
        clean_embedding = batch["frozen_clean_embedding"]
        transfer_embedding = batch["frozen_transfer_embedding"]
        embeddings = {
            "physical_clean": clean_embedding,
            "physical_transfer": transfer_embedding,
            "context_clean": clean_embedding,
            "context_transfer": transfer_embedding,
            "context_mismatch": transfer_embedding,
            "pseudo": pseudo_embedding,
        }
        if len(attenuation_index) > 0:
            if "frozen_p_attenuation_embedding" not in batch:
                raise RuntimeError("冻结视图缓存缺少 P 响应衰减特征")
            attenuation_embedding = batch[
                "frozen_p_attenuation_embedding"
            ][attenuation_index]
            embeddings["physical_attenuation"] = attenuation_embedding
            embeddings["context_attenuation"] = attenuation_embedding
        if include_augmented:
            embeddings["physical_augmented"] = model.encode(
                mask_physical_context(batch["augmented_inputs"])
            )
        else:
            embeddings["physical_augmented"] = clean_embedding
        return embeddings

    physical_inputs = mask_physical_context(batch["clean_inputs"])
    physical_transfer_inputs = mask_physical_context(batch["transfer_inputs"])
    pseudo_raw, pseudo_type = make_pseudo_a_inputs(
        batch["clean_inputs"], return_type=True
    )
    batch["pseudo_a_type"] = pseudo_type
    pseudo_inputs = mask_physical_context(pseudo_raw)
    names = ["physical_clean"]
    views = [physical_inputs]
    if include_augmented:
        names.append("physical_augmented")
        views.append(mask_physical_context(batch["augmented_inputs"]))
    names.extend((
        "physical_transfer",
        "context_clean",
        "context_transfer",
        "context_mismatch",
        "pseudo",
    ))
    views.extend((
        physical_transfer_inputs,
        mask_behavior_season(batch["clean_inputs"]),
        mask_behavior_season(batch["transfer_inputs"]),
        mask_behavior_season(batch["mismatch_inputs"]),
        pseudo_inputs,
    ))
    if len(attenuation_index) > 0:
        attenuation_inputs = build_p_attenuation_inputs(
            batch["clean_inputs"],
            batch["attenuation_pq_inputs"],
            attenuation_index,
        )
        names.extend(("physical_attenuation", "context_attenuation"))
        views.extend((
            mask_physical_context(attenuation_inputs),
            mask_behavior_season(attenuation_inputs),
        ))
    all_embeddings = model.encode(torch.cat(views, dim=0))
    embeddings = dict(zip(
        names,
        torch.split(all_embeddings, [view.size(0) for view in views], dim=0),
    ))
    if not include_augmented:
        embeddings["physical_augmented"] = embeddings["physical_clean"]
    return embeddings


def weighted_corr(values, targets, weights):
    values = np.asarray(values, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weight_sum = weights.sum()
    if weight_sum <= 1e-12:
        return 0.0
    value_mean = (weights * values).sum() / weight_sum
    target_mean = (weights * targets).sum() / weight_sum
    value_centered = values - value_mean
    target_centered = targets - target_mean
    covariance = (weights * value_centered * target_centered).sum() / weight_sum
    value_var = (weights * np.square(value_centered)).sum() / weight_sum
    target_var = (weights * np.square(target_centered)).sum() / weight_sum
    if value_var <= 1e-12 or target_var <= 1e-12:
        return 0.0
    return float(covariance / np.sqrt(value_var * target_var))


def prediction_metrics(prediction, target, target_weight, sample_weight, names, prefix):
    metrics = {}
    effective = target_weight * sample_weight[:, None]
    correlations = []
    errors = []
    prediction_stds = []
    target_stds = []
    for index, name in enumerate(names):
        weight = effective[:, index]
        correlation = weighted_corr(prediction[:, index], target[:, index], weight)
        error = float(
            (np.abs(prediction[:, index] - target[:, index]) * weight).sum()
            / (weight.sum() + 1e-12)
        )
        metrics[f"{prefix}_{name}_corr"] = correlation
        metrics[f"{prefix}_{name}_mae"] = error
        prediction_std = float(prediction[:, index].std())
        target_std = float(target[:, index].std())
        metrics[f"{prefix}_{name}_pred_std"] = prediction_std
        metrics[f"{prefix}_{name}_target_std"] = target_std
        metrics[f"{prefix}_{name}_std_ratio"] = prediction_std / (target_std + 1e-12)
        correlations.append(correlation)
        errors.append(error)
        prediction_stds.append(prediction_std)
        target_stds.append(target_std)
    metrics[f"{prefix}_corr_mean"] = float(np.mean(correlations))
    metrics[f"{prefix}_corr_min"] = float(np.min(correlations))
    metrics[f"{prefix}_mae_mean"] = float(np.mean(errors))
    metrics[f"{prefix}_pred_std_min"] = float(np.min(prediction_stds))
    metrics[f"{prefix}_pred_std_mean"] = float(np.mean(prediction_stds))
    metrics[f"{prefix}_target_std_min"] = float(np.min(target_stds))
    metrics[f"{prefix}_target_std_mean"] = float(np.mean(target_stds))
    metrics[f"{prefix}_output_effective_rank"] = effective_rank(prediction)
    metrics[f"{prefix}_target_effective_rank"] = effective_rank(target)
    return metrics


def embedding_structure_metrics(values, prefix):
    dimension_std = np.asarray(values, dtype=np.float64).std(axis=0)
    normalized = values / (
        np.linalg.norm(values, axis=1, keepdims=True) + 1e-12
    )
    normalized_std = np.asarray(normalized, dtype=np.float64).std(axis=0)
    centered = np.asarray(values, dtype=np.float64) - np.asarray(
        values, dtype=np.float64
    ).mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    energy = np.square(singular_values)
    top1_energy_ratio = float(energy[0] / (energy.sum() + 1e-12))
    return {
        f"{prefix}_std_min": float(dimension_std.min()),
        f"{prefix}_std_mean": float(dimension_std.mean()),
        f"{prefix}_normalized_std_min": float(normalized_std.min()),
        f"{prefix}_normalized_std_mean": float(normalized_std.mean()),
        f"{prefix}_effective_rank": effective_rank(values),
        f"{prefix}_top1_energy_ratio": top1_energy_ratio,
    }


def pair_correlation(values, left_index, right_index):
    weights = np.ones(values.shape[0], dtype=np.float64)
    return weighted_corr(
        values[:, left_index], values[:, right_index], weights
    )


def train_one_epoch(model, criterion, loader, optimizer, scaler, device, epoch, args):
    model.train()
    use_frozen_cache = (
        args.cache_frozen_views and epoch <= args.freeze_backbone_epochs
    )
    if use_frozen_cache:
        model.backbone.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()
    totals = {}
    sample_count = 0
    augment_config = Stage2AugmentConfig(
        scale_min=args.augment_scale_min,
        scale_max=args.augment_scale_max,
        smooth_prob=args.augment_smooth_prob,
        smooth_window=args.augment_smooth_window,
        noise_std=args.augment_noise_std,
    )
    for batch_index, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)
        batch = augment_stage2_batch(batch, augment_config)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, args.amp):
            encoding_context = torch.no_grad() if use_frozen_cache else nullcontext()
            with encoding_context:
                embeddings = encode_stage2_views(
                    model,
                    batch,
                    use_frozen_cache=use_frozen_cache,
                    p_attenuation_fraction=getattr(
                        args, "p_attenuation_fraction", 0.25
                    ),
                )
        embeddings = {name: value.float() for name, value in embeddings.items()}
        loss, logs, _ = criterion(embeddings, batch, epoch)
        if (
            batch_index % args.finite_check_interval == 0
            and not torch.isfinite(loss.detach()).item()
        ):
            raise FloatingPointError(f"epoch={epoch}, batch={batch_index} 出现非有限损失")
        scale_before = float(scaler.get_scale())
        scaler.scale(loss).backward()
        if args.clip > 0:
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.clip
            )
            gradient_is_finite = torch.isfinite(gradient_norm.detach())
            if not scaler.is_enabled() and not gradient_is_finite.item():
                raise FloatingPointError(
                    f"epoch={epoch}, batch={batch_index} 梯度范数出现NaN/Inf"
                )
            gradient_norm_for_log = torch.where(
                gradient_is_finite,
                gradient_norm.detach(),
                gradient_norm.detach().new_zeros(()),
            )
            gradient_clipped = (
                gradient_is_finite & (gradient_norm.detach() > args.clip)
            ).to(loss.dtype)
            gradient_nonfinite = (~gradient_is_finite).to(loss.dtype)
        else:
            gradient_norm = loss.detach().new_zeros(())
            gradient_norm_for_log = gradient_norm
            gradient_clipped = loss.detach().new_zeros(())
            gradient_nonfinite = loss.detach().new_zeros(())
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        amp_overflow = loss.detach().new_tensor(scale_after < scale_before)
        batch_size = batch["clean_inputs"].size(0)
        for name, value in logs.items():
            totals[name] = totals.get(name, 0.0) + value * batch_size
        totals["gradient_norm"] = (
            totals.get("gradient_norm", 0.0) + gradient_norm_for_log * batch_size
        )
        totals["gradient_clipped_fraction"] = (
            totals.get("gradient_clipped_fraction", 0.0)
            + gradient_clipped.detach() * batch_size
        )
        totals["amp_overflow_fraction"] = (
            totals.get("amp_overflow_fraction", 0.0)
            + amp_overflow * batch_size
        )
        totals["gradient_nonfinite_fraction"] = (
            totals.get("gradient_nonfinite_fraction", 0.0)
            + gradient_nonfinite.detach() * batch_size
        )
        sample_count += batch_size
    if sample_count == 0:
        raise RuntimeError("训练 DataLoader 为空")
    metrics = {
        name: float((value / sample_count).item())
        for name, value in totals.items()
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - start_time
    metrics["wall_seconds"] = float(wall_seconds)
    metrics["samples_per_second"] = float(sample_count / max(wall_seconds, 1e-7))
    metrics["amp_scale"] = float(scaler.get_scale())
    non_finite = [
        name for name, value in metrics.items()
        if not np.isfinite(value)
    ]
    if non_finite:
        raise FloatingPointError(
            f"训练指标出现非有限值：{non_finite[:8]}"
        )
    return metrics


@torch.inference_mode()
def evaluate(model, criterion, loader, device, epoch, args):
    model.eval()
    use_frozen_cache = (
        args.cache_frozen_views and epoch <= args.freeze_backbone_epochs
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()
    totals = {}
    sample_count = 0
    physical_predictions = []
    physical_targets = []
    physical_weights = []
    response_predictions = []
    response_targets = []
    response_weights = []
    behavior_predictions = []
    behavior_targets = []
    behavior_weights = []
    transfer_behavior_predictions = []
    transfer_behavior_targets = []
    transfer_behavior_weights = []
    mismatch_behavior_predictions = []
    mismatch_behavior_targets = []
    transfer_weights = []
    sample_weights = []
    physical_embeddings = []
    context_embeddings = []
    transfer_context_embeddings = []
    stage1_embeddings = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with autocast_context(device, args.amp):
            batch_embeddings = encode_stage2_views(
                model,
                batch,
                include_augmented=False,
                use_frozen_cache=use_frozen_cache,
                p_attenuation_fraction=getattr(
                    args, "p_attenuation_eval_fraction", 1.0
                ),
            )
        batch_embeddings = {
            name: value.float() for name, value in batch_embeddings.items()
        }
        loss, logs, outputs = criterion(batch_embeddings, batch, epoch)
        batch_size = batch["clean_inputs"].size(0)
        for name, value in logs.items():
            totals[name] = totals.get(name, 0.0) + value * batch_size
        sample_count += batch_size
        physical_predictions.append(outputs["physical"])
        physical_targets.append(batch["physical_target"])
        physical_weights.append(batch["physical_weight"])
        response_predictions.append(outputs["response"])
        response_targets.append(batch["response_target"])
        response_weights.append(batch["response_weight"])
        behavior_predictions.append(outputs["behavior"])
        behavior_targets.append(batch["behavior_target"])
        behavior_weights.append(batch["behavior_weight"])
        transfer_behavior_predictions.append(outputs["transfer_behavior"])
        transfer_behavior_targets.append(batch["transfer_behavior_target"])
        transfer_behavior_weights.append(batch["transfer_behavior_weight"])
        mismatch_behavior_predictions.append(outputs["mismatch_behavior"])
        mismatch_behavior_targets.append(batch["mismatch_behavior_target"])
        transfer_weights.append(batch["transfer_weight"])
        sample_weights.append(batch["sample_weight"])
        physical_embeddings.append(batch_embeddings["physical_clean"])
        context_embeddings.append(batch_embeddings["context_clean"])
        transfer_context_embeddings.append(batch_embeddings["context_transfer"])
        stage1_embeddings.append(batch["stage1_embedding"])

    metrics = {
        name: float((value / max(sample_count, 1)).item())
        for name, value in totals.items()
    }
    def concatenate_to_numpy(values):
        return torch.cat(values, dim=0).float().cpu().numpy()

    physical_prediction = concatenate_to_numpy(physical_predictions)
    physical_target = concatenate_to_numpy(physical_targets)
    physical_weight = concatenate_to_numpy(physical_weights)
    response_prediction = concatenate_to_numpy(response_predictions)
    response_target = concatenate_to_numpy(response_targets)
    response_weight = concatenate_to_numpy(response_weights)
    behavior_prediction = concatenate_to_numpy(behavior_predictions)
    behavior_target = concatenate_to_numpy(behavior_targets)
    behavior_weight = concatenate_to_numpy(behavior_weights)
    transfer_behavior_prediction = concatenate_to_numpy(transfer_behavior_predictions)
    transfer_behavior_target = concatenate_to_numpy(transfer_behavior_targets)
    transfer_behavior_weight = concatenate_to_numpy(transfer_behavior_weights)
    mismatch_behavior_prediction = concatenate_to_numpy(mismatch_behavior_predictions)
    mismatch_behavior_target = concatenate_to_numpy(mismatch_behavior_targets)
    transfer_weight = concatenate_to_numpy(transfer_weights)
    sample_weight = concatenate_to_numpy(sample_weights)
    physical_embedding = concatenate_to_numpy(physical_embeddings)
    context_embedding = concatenate_to_numpy(context_embeddings)
    transfer_context_embedding = concatenate_to_numpy(transfer_context_embeddings)
    stage1_embedding = concatenate_to_numpy(stage1_embeddings)
    metrics.update(prediction_metrics(
        physical_prediction,
        physical_target,
        physical_weight,
        sample_weight,
        PHYSICAL_TARGET_NAMES,
        "physical",
    ))
    metrics.update(prediction_metrics(
        response_prediction,
        response_target,
        response_weight,
        sample_weight,
        RESPONSE_TARGET_NAMES,
        "response",
    ))
    metrics.update(prediction_metrics(
        behavior_prediction,
        behavior_target,
        behavior_weight,
        sample_weight,
        BEHAVIOR_TARGET_NAMES,
        "behavior",
    ))
    metrics.update(prediction_metrics(
        transfer_behavior_prediction,
        transfer_behavior_target,
        transfer_behavior_weight,
        sample_weight * transfer_weight,
        BEHAVIOR_TARGET_NAMES,
        "transfer_behavior",
    ))
    metrics.update(embedding_structure_metrics(
        physical_embedding, "physical_embedding"
    ))
    metrics.update(embedding_structure_metrics(
        context_embedding, "context_embedding"
    ))
    metrics.update(embedding_structure_metrics(
        transfer_context_embedding, "transfer_context_embedding"
    ))
    metrics.update(embedding_structure_metrics(
        stage1_embedding, "stage1_anchor_embedding"
    ))
    metrics["embedding_rank_retention"] = float(
        metrics["physical_embedding_effective_rank"]
        / max(metrics["stage1_anchor_embedding_effective_rank"], 1e-12)
    )
    physical_score_index = PHYSICAL_TARGET_NAMES.index("storage_evidence")
    response_score_index = RESPONSE_TARGET_NAMES.index("response_confidence")
    behavior_score_index = BEHAVIOR_TARGET_NAMES.index("storage_behavior")
    core_prediction = np.column_stack([
        physical_prediction[:, physical_score_index],
        response_prediction[:, response_score_index],
        behavior_prediction[:, behavior_score_index],
    ])
    core_target = np.column_stack([
        physical_target[:, physical_score_index],
        response_target[:, response_score_index],
        behavior_target[:, behavior_score_index],
    ])
    core_pairs = (
        ("physical_response", 0, 1),
        ("physical_behavior", 0, 2),
        ("response_behavior", 1, 2),
    )
    correlation_excess = []
    for pair_name, left_index, right_index in core_pairs:
        prediction_correlation = pair_correlation(
            core_prediction, left_index, right_index
        )
        target_correlation = pair_correlation(core_target, left_index, right_index)
        metrics[f"core_pred_corr_{pair_name}"] = prediction_correlation
        metrics[f"core_target_corr_{pair_name}"] = target_correlation
        correlation_excess.append(
            abs(prediction_correlation) - abs(target_correlation)
        )
    metrics["core_corr_excess_max"] = float(max(correlation_excess))
    metrics["core_pred_effective_rank"] = effective_rank(core_prediction)
    metrics["core_target_effective_rank"] = effective_rank(core_target)
    score_index = BEHAVIOR_TARGET_NAMES.index("storage_behavior")
    valid = transfer_weight > 0.0
    if valid.any():
        predicted_drop = (
            transfer_behavior_prediction[valid, score_index]
            - mismatch_behavior_prediction[valid, score_index]
        )
        target_drop = (
            transfer_behavior_target[valid, score_index]
            - mismatch_behavior_target[valid, score_index]
        )
        metrics["policy_score_drop"] = float(predicted_drop.mean())
        metrics["policy_target_drop"] = float(target_drop.mean())
        metrics["policy_pair_accuracy"] = float((predicted_drop > 0.0).mean())
        metrics["policy_drop_corr"] = weighted_corr(
            predicted_drop,
            target_drop,
            sample_weight[valid] * transfer_weight[valid],
        )
    else:
        metrics.update({
            "policy_score_drop": 0.0,
            "policy_target_drop": 0.0,
            "policy_pair_accuracy": 0.0,
            "policy_drop_corr": 0.0,
        })
    metrics["embedding_effective_rank"] = metrics[
        "physical_embedding_effective_rank"
    ]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - start_time
    metrics["wall_seconds"] = float(wall_seconds)
    metrics["samples_per_second"] = float(sample_count / max(wall_seconds, 1e-7))
    non_finite = [
        name for name, value in metrics.items()
        if not np.isfinite(value)
    ]
    if non_finite:
        raise FloatingPointError(
            f"验证指标出现非有限值：{non_finite[:8]}"
        )
    return metrics


def effective_rank(embedding):
    centered = embedding - embedding.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    power = np.square(singular_values)
    probability = power / (power.sum() + 1e-12)
    probability = np.clip(probability, 1e-12, 1.0)
    return float(np.exp(-(probability * np.log(probability)).sum()))


def composite_score(metrics, target_std):
    physical_corr = np.clip(metrics["physical_corr_mean"], 0.0, 1.0)
    response_corr = np.clip(metrics["response_corr_mean"], 0.0, 1.0)
    behavior_corr = np.clip(metrics["behavior_corr_mean"], 0.0, 1.0)
    transfer_corr = np.clip(metrics["transfer_behavior_corr_mean"], 0.0, 1.0)
    weakest_corr = np.clip(
        min(
            metrics["physical_corr_min"],
            metrics["response_corr_min"],
            metrics["behavior_corr_min"],
        ),
        0.0,
        1.0,
    )
    mean_mae = np.mean([
        metrics["physical_mae_mean"],
        metrics["response_mae_mean"],
        metrics["behavior_mae_mean"],
    ])
    accuracy = np.clip(1.0 - mean_mae, 0.0, 1.0)
    cloud = np.clip(
        metrics["physical_embedding_normalized_std_min"]
        / max(target_std, 1e-6),
        0.0,
        1.0,
    )
    rank_health = np.clip(metrics["embedding_rank_retention"] / 0.50, 0.0, 1.0)
    pseudo_separation = np.clip(metrics["pseudo_score_drop"] / 0.15, 0.0, 1.0)
    target_drop = max(metrics["policy_target_drop"], 0.03)
    policy_drop_error = abs(metrics["policy_score_drop"] - metrics["policy_target_drop"])
    policy_calibration = np.clip(1.0 - policy_drop_error / target_drop, 0.0, 1.0)
    policy_separation = 0.40 * np.clip(metrics["policy_pair_accuracy"], 0.0, 1.0)
    policy_separation += 0.20 * np.clip(
        metrics["policy_score_drop"] / target_drop, 0.0, 1.0
    )
    policy_separation += 0.20 * policy_calibration
    policy_separation += 0.20 * np.clip(metrics["policy_drop_corr"], 0.0, 1.0)
    return float(
        0.22 * physical_corr
        + 0.17 * response_corr
        + 0.23 * behavior_corr
        + 0.10 * transfer_corr
        + 0.05 * weakest_corr
        + 0.04 * accuracy
        + 0.03 * cloud
        + 0.03 * rank_health
        + 0.05 * pseudo_separation
        + 0.08 * policy_separation
    )


def set_backbone_phase(model, optimizer, epoch, args):
    head_lr_scale = optimizer.param_groups[1]["lr"] / max(args.lr_head, 1e-12)
    if epoch <= args.freeze_backbone_epochs:
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
        optimizer.param_groups[0]["lr"] = 0.0
        return "head-only"

    for parameter in model.backbone.parameters():
        parameter.requires_grad_(True)
    ramp_epoch = epoch - args.freeze_backbone_epochs
    ramp_ratio = min(1.0, ramp_epoch / max(args.unfreeze_lr_ramp_epochs, 1))
    optimizer.param_groups[0]["lr"] = (
        args.lr_backbone * head_lr_scale * ramp_ratio
    )
    return "joint"


def save_checkpoint(path, model, args, model_kwargs, data_info, epoch, score, metrics):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "model_kwargs": model_kwargs,
        "model_arch_version": TCN_LSTM_ARCH_VERSION,
        "data_info": data_info,
        "epoch": int(epoch),
        "best_epoch": int(epoch),
        "best_score": float(score),
        "validation_metrics": metrics,
        "physical_target_names": list(PHYSICAL_TARGET_NAMES),
        "response_target_names": list(RESPONSE_TARGET_NAMES),
        "behavior_target_names": list(BEHAVIOR_TARGET_NAMES),
        "target_pipeline_version": TARGET_PIPELINE_VERSION,
        "stage2_objective_version": STAGE2_OBJECTIVE_VERSION,
        "physical_head_semantics": "policy-invariant storage evidence",
        "behavior_head_semantics": "role-relative storage behavior under current policy",
        "teacher_score": "behavior.storage_behavior",
        "physical_policy_masked": True,
        "physical_season_masked": True,
        "behavior_season_masked": True,
        "behavior_policy_role_retained": True,
        "policy_counterfactual_training": True,
        "hierarchy_consistency_training": bool(args.lambda_hierarchy > 0.0),
        "p_attenuation_training": bool(args.lambda_p_attenuation > 0.0),
        "p_attenuation_role_invariance": bool(args.lambda_role_invariance > 0.0),
        "training_schedule": (
            f"freeze-{args.freeze_backbone_epochs}-full-unfreeze-"
            f"linear-ramp-{args.unfreeze_lr_ramp_epochs}"
        ),
        "gpu_resident_training": data_info.get("loader_mode") == "gpu-resident-index-select",
        "cudnn_deterministic": bool(args.deterministic),
        "tf32": bool(args.tf32),
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)


def print_detailed_metrics(epoch, metrics):
    physical = " | ".join(
        f"{name}: r={metrics[f'physical_{name}_corr']:.3f}, mae={metrics[f'physical_{name}_mae']:.3f}"
        for name in PHYSICAL_TARGET_NAMES
    )
    response = " | ".join(
        f"{name}: r={metrics[f'response_{name}_corr']:.3f}, mae={metrics[f'response_{name}_mae']:.3f}"
        for name in RESPONSE_TARGET_NAMES
    )
    behavior = " | ".join(
        f"{name}: r={metrics[f'behavior_{name}_corr']:.3f}, mae={metrics[f'behavior_{name}_mae']:.3f}"
        for name in BEHAVIOR_TARGET_NAMES
    )
    print(f"[ValPhysical] epoch={epoch} | {physical}")
    print(f"[ValResponse] epoch={epoch} | {response}")
    print(f"[ValBehavior] epoch={epoch} | {behavior}")
    print(
        f"[ValPolicy] epoch={epoch} | drop={metrics['policy_score_drop']:.3f}/"
        f"{metrics['policy_target_drop']:.3f} | pair={metrics['policy_pair_accuracy']:.3f} | "
        f"drop_r={metrics['policy_drop_corr']:.3f} | "
        f"calib_mae={abs(metrics['policy_score_drop'] - metrics['policy_target_drop']):.3f}"
    )
    print(
        f"[ValPseudoA] epoch={epoch} | all={metrics['pseudo_score_drop']:.3f}/"
        f"{metrics['pseudo_pair_accuracy']:.3f} | "
        f"shift={metrics['pseudo_shift_score_drop']:.3f}/"
        f"{metrics['pseudo_shift_pair_accuracy']:.3f} | "
        f"shift_sign={metrics['pseudo_shift_sign_score_drop']:.3f}/"
        f"{metrics['pseudo_shift_sign_pair_accuracy']:.3f}"
    )
    print(
        f"[ValConstraints] epoch={epoch} | hierarchy="
        f"{metrics['hierarchy_total_loss']:.5f} "
        f"({metrics['hierarchy_physical_loss']:.5f}/"
        f"{metrics['hierarchy_behavior_loss']:.5f}) | formula_mae="
        f"{metrics['evidence_formula_mae']:.4f}/"
        f"{metrics['behavior_formula_mae']:.4f} | role_inv="
        f"{metrics['role_invariance_loss']:.5f}"
    )
    print(
        f"[ValPAtt] epoch={epoch} | pred/target/margin="
        f"{metrics['p_attenuation_score_drop']:.4f}/"
        f"{metrics['p_attenuation_target_drop']:.4f}/"
        f"{metrics['p_attenuation_required_margin']:.4f} | "
        f"pair/margin_ok={metrics['p_attenuation_pair_accuracy']:.3f}/"
        f"{metrics['p_attenuation_margin_accuracy']:.3f} | "
        f"rank_loss={metrics['p_attenuation_rank_loss']:.5f}"
    )
    attenuation_labels = (
        ("evidence", "storage_evidence"),
        ("response", "response_confidence"),
        ("behavior", "storage_behavior"),
    )
    attenuation_detail = " | ".join(
        f"{display}: drop={metrics[f'p_attenuation_{name}_score_drop']:.4f}/"
        f"{metrics[f'p_attenuation_{name}_target_drop']:.4f}, "
        f"pair={metrics[f'p_attenuation_{name}_pair_accuracy']:.3f}, "
        f"margin={metrics[f'p_attenuation_{name}_margin_accuracy']:.3f}"
        for display, name in attenuation_labels
    )
    print(
        f"[ValPAttCore] epoch={epoch} | {attenuation_detail}"
    )
    print(
        f"[ValCollapse] epoch={epoch} | "
        f"Hphys={metrics['physical_embedding_std_min']:.4f}/"
        f"{metrics['physical_embedding_normalized_std_min']:.4f}/"
        f"{metrics['physical_embedding_effective_rank']:.2f} | "
        f"Hctx={metrics['context_embedding_std_min']:.4f}/"
        f"{metrics['context_embedding_normalized_std_min']:.4f}/"
        f"{metrics['context_embedding_effective_rank']:.2f} | "
        f"rank_keep={metrics['embedding_rank_retention']:.3f} | "
        f"head_std={metrics['physical_pred_std_min']:.4f}/"
        f"{metrics['response_pred_std_min']:.4f}/"
        f"{metrics['behavior_pred_std_min']:.4f} | head_erank="
        f"{metrics['physical_output_effective_rank']:.2f}/"
        f"{metrics['physical_target_effective_rank']:.2f},"
        f"{metrics['response_output_effective_rank']:.2f}/"
        f"{metrics['response_target_effective_rank']:.2f},"
        f"{metrics['behavior_output_effective_rank']:.2f}/"
        f"{metrics['behavior_target_effective_rank']:.2f}"
    )
    print(
        f"[ValCoreCorr] epoch={epoch} | pred/target "
        f"P-R={metrics['core_pred_corr_physical_response']:.3f}/"
        f"{metrics['core_target_corr_physical_response']:.3f} | "
        f"P-B={metrics['core_pred_corr_physical_behavior']:.3f}/"
        f"{metrics['core_target_corr_physical_behavior']:.3f} | "
        f"R-B={metrics['core_pred_corr_response_behavior']:.3f}/"
        f"{metrics['core_target_corr_response_behavior']:.3f} | "
        f"core_erank={metrics['core_pred_effective_rank']:.2f}/"
        f"{metrics['core_target_effective_rank']:.2f} | "
        f"core_std_ratio="
        f"{metrics['physical_storage_evidence_std_ratio']:.2f}/"
        f"{metrics['response_response_confidence_std_ratio']:.2f}/"
        f"{metrics['behavior_storage_behavior_std_ratio']:.2f} | "
        f"corr_excess={metrics['core_corr_excess_max']:.3f}"
    )


def save_history(path, history):
    save_rows_csv(path, history)


def save_training_summary(path, data_info, best_epoch, best_score, best_metrics):
    weakest_physical = min(
        best_metrics[f"physical_{name}_corr"] for name in PHYSICAL_TARGET_NAMES
    )
    weakest_response = min(
        best_metrics[f"response_{name}_corr"] for name in RESPONSE_TARGET_NAMES
    )
    weakest_behavior = min(
        best_metrics[f"behavior_{name}_corr"] for name in BEHAVIOR_TARGET_NAMES
    )
    reasons = []
    warnings = []
    if weakest_physical < 0.50:
        reasons.append(f"最弱物理目标相关系数不足：{weakest_physical:.3f}")
    if weakest_response < 0.50:
        reasons.append(f"最弱响应相关系数不足：{weakest_response:.3f}")
    if weakest_behavior < 0.45:
        reasons.append(f"最弱行为相关系数不足：{weakest_behavior:.3f}")
    if best_metrics["embedding_std_min"] < 0.012:
        reasons.append(f"嵌入最小标准差过低：{best_metrics['embedding_std_min']:.4f}")
    elif best_metrics["embedding_std_min"] < 0.02:
        warnings.append(f"嵌入最小标准差偏低：{best_metrics['embedding_std_min']:.4f}")
    for embedding_name in ("physical_embedding", "context_embedding"):
        std_min = best_metrics[f"{embedding_name}_std_min"]
        normalized_std_min = best_metrics[f"{embedding_name}_normalized_std_min"]
        rank = best_metrics[f"{embedding_name}_effective_rank"]
        if std_min < 0.005:
            reasons.append(f"{embedding_name} 标准差接近塌缩：{std_min:.4f}")
        elif std_min < 0.01:
            warnings.append(f"{embedding_name} 最小标准差偏低：{std_min:.4f}")
        if rank < 6.0:
            warnings.append(f"{embedding_name} 有效秩偏低：{rank:.2f}")
        if normalized_std_min < 0.012 and rank < 4.0:
            reasons.append(
                f"{embedding_name} 归一化后标准差接近塌缩：{normalized_std_min:.4f}"
            )
        elif normalized_std_min < 0.02:
            warnings.append(
                f"{embedding_name} 归一化后最小标准差偏低：{normalized_std_min:.4f}"
            )
    rank_retention = best_metrics["embedding_rank_retention"]
    if rank_retention < 0.25:
        reasons.append(f"相对 Stage-1 的有效秩保留不足：{rank_retention:.3f}")
    elif rank_retention < 0.45:
        warnings.append(f"相对 Stage-1 的有效秩保留偏低：{rank_retention:.3f}")
    for head_name in ("physical", "response", "behavior"):
        pred_std_min = best_metrics[f"{head_name}_pred_std_min"]
        if pred_std_min < 0.005:
            reasons.append(f"{head_name} 教师头输出接近常数：{pred_std_min:.4f}")
        elif pred_std_min < 0.01:
            warnings.append(f"{head_name} 教师头输出方差偏低：{pred_std_min:.4f}")
    if best_metrics["pseudo_score_drop"] < 0.08:
        reasons.append(f"A类错配分数下降不足：{best_metrics['pseudo_score_drop']:.4f}")
    if best_metrics["pseudo_pair_accuracy"] < 0.70:
        reasons.append(
            f"A类错配总体排序准确率不足：{best_metrics['pseudo_pair_accuracy']:.3f}"
        )
    for pseudo_name, display_name in (
        ("pseudo_shift", "A类平移错配"),
        ("pseudo_shift_sign", "A类平移反号错配"),
    ):
        pair_accuracy = best_metrics[f"{pseudo_name}_pair_accuracy"]
        score_drop = best_metrics[f"{pseudo_name}_score_drop"]
        if pair_accuracy < 0.60:
            reasons.append(f"{display_name}排序准确率不足：{pair_accuracy:.3f}")
        if score_drop < 0.03:
            warnings.append(f"{display_name}分数下降偏低：{score_drop:.4f}")
    if best_metrics["p_attenuation_pair_accuracy"] < 0.80:
        warnings.append(
            "P 响应衰减反事实排序偏弱："
            f"{best_metrics['p_attenuation_pair_accuracy']:.3f}"
        )
    if best_metrics["p_attenuation_score_drop"] < 0.005:
        warnings.append(
            "P 响应衰减后的核心分数下降偏低："
            f"{best_metrics['p_attenuation_score_drop']:.4f}"
        )
    if best_metrics["p_attenuation_margin_accuracy"] < 0.65:
        warnings.append(
            "P衰减约束总体margin达标率偏低："
            f"{best_metrics['p_attenuation_margin_accuracy']:.3f}"
        )
    for target_name in (
        "storage_evidence",
        "response_confidence",
        "storage_behavior",
    ):
        margin_accuracy = best_metrics[
            f"p_attenuation_{target_name}_margin_accuracy"
        ]
        if margin_accuracy < 0.55:
            warnings.append(
                f"P衰减约束{target_name}的margin达标率偏低："
                f"{margin_accuracy:.3f}"
            )
    if max(
        best_metrics["evidence_formula_mae"],
        best_metrics["behavior_formula_mae"],
    ) > 0.08:
        warnings.append(
            "教师层级公式一致性偏弱："
            f"{best_metrics['evidence_formula_mae']:.4f}/"
            f"{best_metrics['behavior_formula_mae']:.4f}"
        )
    if (
        best_metrics["core_corr_excess_max"] > 0.20
        and best_metrics["core_pred_effective_rank"]
        < 0.75 * best_metrics["core_target_effective_rank"]
    ):
        warnings.append(
            "三个核心教师分数的相关性明显高于目标，可能存在多头复制"
        )
    minimum_policy_drop = max(0.03, 0.5 * best_metrics["policy_target_drop"])
    if best_metrics["policy_score_drop"] < minimum_policy_drop:
        reasons.append(
            f"政策错配分数下降不足：{best_metrics['policy_score_drop']:.4f}"
        )
    if best_metrics["policy_pair_accuracy"] < 0.70:
        reasons.append(
            f"同步迁移高于政策错配的比例不足：{best_metrics['policy_pair_accuracy']:.3f}"
        )
    if best_metrics["policy_drop_corr"] < 0.0:
        reasons.append(
            f"政策响应强弱相关系数为负：{best_metrics['policy_drop_corr']:.3f}"
        )
    elif best_metrics["policy_drop_corr"] < 0.20:
        warnings.append(
            f"政策响应强弱相关系数偏低：{best_metrics['policy_drop_corr']:.3f}"
        )
    if data_info["transfer_acceptance_val"] < 0.30:
        reasons.append(
            f"验证集政策迁移可用率过低：{data_info['transfer_acceptance_val']:.3f}"
        )
    summary = {
        "verdict": "PASS" if not reasons else "REVIEW",
        "reasons": reasons,
        "warnings": warnings,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_validation_metrics": best_metrics,
        "data_info": data_info,
        "physical_target_names": list(PHYSICAL_TARGET_NAMES),
        "response_target_names": list(RESPONSE_TARGET_NAMES),
        "behavior_target_names": list(BEHAVIOR_TARGET_NAMES),
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def train_phase2(args, device):
    set_seed(
        args.seed,
        cudnn_benchmark=args.cudnn_benchmark,
        deterministic=args.deterministic,
    )
    args.save_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, data_info = build_dataloaders_st2(args)
    model, criterion, optimizer, scheduler, model_kwargs, stage1_teacher = (
        build_model_and_optimizer(args, device)
    )
    prepare_stage1_anchor_cache(
        stage1_teacher, train_loader, val_loader, device, args
    )
    del stage1_teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()
    train_loader, val_loader = maybe_enable_gpu_resident_loaders(
        train_loader, val_loader, device, args, data_info
    )
    data_info["frozen_view_cache"] = bool(args.cache_frozen_views)
    data_info["fused_adam"] = bool(args.fused_adam)
    data_info["cudnn_benchmark"] = bool(args.cudnn_benchmark)
    data_info["cudnn_deterministic"] = bool(args.deterministic)
    data_info["tf32"] = bool(args.tf32)
    data_info["freeze_backbone_epochs"] = int(args.freeze_backbone_epochs)
    data_info["unfreeze_lr_ramp_epochs"] = int(args.unfreeze_lr_ramp_epochs)
    scaler = create_grad_scaler(args.amp)
    print(
        f"[Data] train={data_info['train_samples']} ({data_info['train_users']} users), "
        f"val={data_info['val_samples']} ({data_info['val_users']} users), "
        f"test={data_info['test_samples']} ({data_info['test_users']} users)"
    )
    print(
        f"[Target] {TARGET_PIPELINE_VERSION} | objective={STAGE2_OBJECTIVE_VERSION} | "
        f"user balance={data_info['user_balance']}"
    )
    print(
        f"[Target] physical={list(PHYSICAL_TARGET_NAMES)} | "
        "season/policy channels masked"
    )
    print(f"[Target] response={list(RESPONSE_TARGET_NAMES)}")
    print(
        f"[Target] behavior_core={list(BEHAVIOR_TARGET_NAMES)} | "
        "season masked, relative policy role retained"
    )
    print(
        f"[Speed] amp={args.amp} | fused_adam={args.fused_adam} | "
        f"cudnn_benchmark={args.cudnn_benchmark} | deterministic={args.deterministic} | "
        f"tf32={args.tf32} | loader={data_info['loader_mode']} | "
        f"frozen_cache={args.cache_frozen_views}"
    )
    print(
        f"[Schedule] freeze={args.freeze_backbone_epochs} | "
        f"lr_ramp={args.unfreeze_lr_ramp_epochs} | warmup={args.warmup_epochs}"
    )
    print(
        f"[Transfer] {POLICY_TRANSFER_VERSION} | "
        f"accept(train/val)={data_info['transfer_acceptance_train']:.3f}/"
        f"{data_info['transfer_acceptance_val']:.3f}"
    )
    print(
        f"[Constraints] hierarchy={args.lambda_hierarchy:.3f} | "
        f"p_attenuation={args.lambda_p_attenuation:.3f} "
        f"(train/eval={args.p_attenuation_fraction:.2f}/"
        f"{args.p_attenuation_eval_fraction:.2f}, margin={args.p_attenuation_margin:.3f}) | "
        f"role_invariance={args.lambda_role_invariance:.3f}"
    )
    attenuation_train = data_info["p_attenuation_target_summary_train"]
    attenuation_val = data_info["p_attenuation_target_summary_val"]
    attenuation_labels = (
        ("evidence", "storage_evidence"),
        ("response", "response_confidence"),
        ("behavior", "storage_behavior"),
    )
    attenuation_data_text = " | ".join(
        f"{display}: valid={attenuation_train[name]['valid_ratio']:.3f}/"
        f"{attenuation_val[name]['valid_ratio']:.3f}, "
        f"val_drop={attenuation_val[name]['target_drop_mean']:.4f}"
        for display, name in attenuation_labels
    )
    print(
        f"[ConstraintData] p_att_valid(train/val)="
        f"{data_info['p_attenuation_valid_ratio_train']:.3f}/"
        f"{data_info['p_attenuation_valid_ratio_val']:.3f} | "
        f"{attenuation_data_text}"
    )

    checkpoint_path = args.save_dir / args.save_name
    history_path = args.save_dir / "phase2_training_history.csv"
    best_score = -float("inf")
    best_epoch = 0
    best_metrics = None
    no_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        phase = set_backbone_phase(model, optimizer, epoch, args)
        train_metrics = train_one_epoch(
            model, criterion, train_loader, optimizer, scaler, device, epoch, args
        )
        val_metrics = evaluate(model, criterion, val_loader, device, epoch, args)
        score = composite_score(val_metrics, args.target_std)
        scheduler.step(score)
        row = {
            "epoch": epoch,
            "phase": phase,
            "score": score,
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[1]["lr"],
        }
        row.update({f"train_{name}": value for name, value in train_metrics.items()})
        row.update({f"val_{name}": value for name, value in val_metrics.items()})
        history.append(row)
        save_history(history_path, history)

        print(
            f"Epoch [{epoch}/{args.epochs}] | {phase} | score={score:.4f} | "
            f"train={train_metrics['loss']:.5f} | val={val_metrics['loss']:.5f} | "
            f"phys_r={val_metrics['physical_corr_mean']:.3f} | "
            f"resp_r={val_metrics['response_corr_mean']:.3f} | "
            f"behavior_r={val_metrics['behavior_corr_mean']:.3f} | "
            f"policy_pair={val_metrics['policy_pair_accuracy']:.3f} | "
            f"pseudo={val_metrics['pseudo_score_drop']:.3f}/"
            f"{val_metrics['pseudo_pair_accuracy']:.3f} | "
            f"p_att={val_metrics['p_attenuation_score_drop']:.3f}/"
            f"{val_metrics['p_attenuation_pair_accuracy']:.3f} | "
            f"H_std={val_metrics['embedding_std_mean']:.4f}/{val_metrics['embedding_std_min']:.4f} | "
            f"erank={val_metrics['embedding_effective_rank']:.2f}/"
            f"{val_metrics['stage1_anchor_embedding_effective_rank']:.2f} | "
            f"time={train_metrics['wall_seconds']:.1f}s+{val_metrics['wall_seconds']:.1f}s | "
            f"throughput={train_metrics['samples_per_second']:.1f}/s"
        )
        print(
            f"[TrainHealth] epoch={epoch} | lr="
            f"{optimizer.param_groups[0]['lr']:.2e}/"
            f"{optimizer.param_groups[1]['lr']:.2e} | grad="
            f"{train_metrics['gradient_norm']:.3f} | clipped="
            f"{train_metrics['gradient_clipped_fraction']:.3f} | "
            f"grad_nonfinite={train_metrics['gradient_nonfinite_fraction']:.3f} | "
            f"amp_scale={train_metrics['amp_scale']:.0f} | "
            f"amp_overflow={train_metrics['amp_overflow_fraction']:.3f}"
        )
        if epoch == 1 or epoch % args.detail_interval == 0:
            print_detailed_metrics(epoch, val_metrics)

        if score > best_score + args.best_min_delta:
            best_score = score
            best_epoch = epoch
            best_metrics = val_metrics
            no_improvement = 0
            save_checkpoint(
                checkpoint_path,
                model,
                args,
                model_kwargs,
                data_info,
                epoch,
                score,
                val_metrics,
            )
            print(f"[Checkpoint] saved={checkpoint_path}")
        else:
            no_improvement += 1

        if epoch >= args.min_train_epochs and no_improvement >= args.early_stop_patience:
            print(f"[EarlyStop] {no_improvement} epochs without improvement")
            break

    if best_metrics is None:
        raise RuntimeError("Stage-2 未生成有效 checkpoint")
    save_training_summary(
        args.save_dir / "phase2_run_summary.json",
        data_info,
        best_epoch,
        best_score,
        best_metrics,
    )
    print(f"[Done] best_epoch={best_epoch}, best_score={best_score:.4f}")
    print_detailed_metrics(best_epoch, best_metrics)
    return checkpoint_path


def parse_args():
    parser = argparse.ArgumentParser(description="Stage-2 相对电价行为教师训练")
    parser.add_argument("--data_path", type=str, default="processed_result.xlsx")
    parser.add_argument("--split_manifest", type=str, default="data_split/user_split.csv")
    parser.add_argument("--user_col", type=str, default="关联用户号")
    parser.add_argument("--policy_name", type=str, default="old", choices=["old", "new"])
    parser.add_argument("--time_type", type=str, default=None)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1111)
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument("--cuda", action="store_true", help="强制使用 CUDA")
    device_group.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    parser.add_argument("--no_amp", action="store_true", help="关闭 CUDA 混合精度")
    parser.add_argument("--no_gpu_data_cache", action="store_true")
    parser.add_argument("--gpu_cache_max_fraction", type=float, default=0.20)
    parser.add_argument("--no_frozen_view_cache", action="store_true")
    parser.add_argument("--no_fused_adam", action="store_true")
    parser.add_argument("--no_cudnn_benchmark", action="store_true")
    parser.add_argument("--deterministic", action="store_true", help="启用较慢的确定性卷积")
    parser.add_argument("--no_tf32", action="store_true", help="关闭 CUDA TF32")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--finite_check_interval", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--nhid", type=int, default=64)
    parser.add_argument("--levels", type=int, default=DEFAULT_TCN_LEVELS)
    parser.add_argument("--ksize", type=int, default=7)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lstm_hidden", type=int, default=64)
    parser.add_argument("--lstm_layers", type=int, default=1)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--lr_backbone", type=float, default=2e-5)
    parser.add_argument("--lr_head", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--lambda_physical", type=float, default=1.2)
    parser.add_argument("--lambda_response", type=float, default=1.0)
    parser.add_argument("--lambda_behavior", type=float, default=1.0)
    parser.add_argument("--lambda_transfer", type=float, default=0.6)
    parser.add_argument("--lambda_policy_mismatch", type=float, default=0.4)
    parser.add_argument("--lambda_policy_rank", type=float, default=0.25)
    parser.add_argument("--policy_margin", type=float, default=0.10)
    parser.add_argument("--lambda_pseudo_rank", type=float, default=0.2)
    parser.add_argument("--pseudo_margin", type=float, default=0.15)
    parser.add_argument("--lambda_hierarchy", type=float, default=0.03)
    parser.add_argument("--lambda_p_attenuation", type=float, default=0.05)
    parser.add_argument("--p_attenuation_margin", type=float, default=0.015)
    parser.add_argument("--lambda_role_invariance", type=float, default=0.01)
    parser.add_argument("--p_attenuation_fraction", type=float, default=0.25)
    parser.add_argument("--p_attenuation_eval_fraction", type=float, default=1.0)
    parser.add_argument(
        "--p_attenuation_alphas",
        type=float,
        nargs="+",
        default=[0.25, 0.50, 0.75],
    )
    parser.add_argument("--lambda_consistency", type=float, default=0.15)
    parser.add_argument("--lambda_reg", type=float, default=0.05)
    parser.add_argument("--lambda_anchor", type=float, default=0.05)
    parser.add_argument("--target_std", type=float, default=0.08)
    parser.add_argument("--reg_covariance_weight", type=float, default=5.0)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=2)
    parser.add_argument("--unfreeze_lr_ramp_epochs", type=int, default=3)
    parser.add_argument("--lr_patience", type=int, default=8)
    parser.add_argument("--min_train_epochs", type=int, default=30)
    parser.add_argument("--early_stop_patience", type=int, default=15)
    parser.add_argument("--best_min_delta", type=float, default=1e-4)
    parser.add_argument("--detail_interval", type=int, default=5)
    parser.add_argument("--activity_ratio", type=float, default=0.02)
    parser.add_argument("--min_activity_to_load_ratio", type=float, default=0.002)
    parser.add_argument("--alignment_max_lag", type=int, default=2)
    parser.add_argument("--min_run_points", type=float, default=2.0)
    parser.add_argument("--edge_window_points", type=int, default=2)
    parser.add_argument("--strength_reference_ratio", type=float, default=0.03)
    parser.add_argument("--role_smooth_radius", type=int, default=2)
    parser.add_argument("--response_noise_floor_ratio", type=float, default=0.03)
    parser.add_argument("--augment_scale_min", type=float, default=0.95)
    parser.add_argument("--augment_scale_max", type=float, default=1.05)
    parser.add_argument("--augment_smooth_prob", type=float, default=0.20)
    parser.add_argument("--augment_smooth_window", type=int, default=3)
    parser.add_argument("--augment_noise_std", type=float, default=0.002)
    parser.add_argument("--transfer_min_activity_coverage", type=float, default=0.70)
    parser.add_argument("--transfer_min_energy_retention", type=float, default=0.70)
    parser.add_argument("--transfer_min_target_pattern", type=float, default=0.60)
    parser.add_argument("--transfer_random_shift_probability", type=float, default=0.50)
    parser.add_argument("--transfer_shift_step", type=int, default=4)
    parser.add_argument("--transfer_min_shift_points", type=int, default=8)
    parser.add_argument("--phase1_ckpt", type=str, default="step_1/best_model_st1.pt")
    parser.add_argument("--save_dir", type=str, default="step_2")
    parser.add_argument("--save_name", type=str, default="best_model_st2.pt")
    parser.add_argument("--rebuild_target_cache", action="store_true")
    parser.add_argument("--rebuild_transfer_cache", action="store_true")
    parser.add_argument("--rebuild_transfer_target_cache", action="store_true")
    parser.add_argument("--rebuild_p_attenuation_cache", action="store_true")
    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error("--num_workers 不能为负数")
    if args.prefetch_factor <= 0:
        parser.error("--prefetch_factor 必须为正数")
    if args.finite_check_interval <= 0:
        parser.error("--finite_check_interval 必须为正数")
    if not 0.0 < args.gpu_cache_max_fraction <= 0.5:
        parser.error("--gpu_cache_max_fraction 必须位于 (0, 0.5]")
    if args.freeze_backbone_epochs < 0:
        parser.error("--freeze_backbone_epochs 不能为负数")
    if args.unfreeze_lr_ramp_epochs <= 0 or args.warmup_epochs <= 0:
        parser.error("解冻学习率爬坡和损失预热轮数必须为正数")
    if args.reg_covariance_weight < 0.0:
        parser.error("--reg_covariance_weight 不能为负数")
    if not 0.0 <= args.p_attenuation_fraction <= 1.0:
        parser.error("--p_attenuation_fraction 必须位于 [0, 1]")
    if not 0.0 <= args.p_attenuation_eval_fraction <= 1.0:
        parser.error("--p_attenuation_eval_fraction 必须位于 [0, 1]")
    if not args.p_attenuation_alphas or any(
        value <= 0.0 or value >= 1.0 for value in args.p_attenuation_alphas
    ):
        parser.error("--p_attenuation_alphas 必须位于 (0, 1)")
    for name in (
        "lambda_hierarchy",
        "lambda_p_attenuation",
        "p_attenuation_margin",
        "lambda_role_invariance",
    ):
        if getattr(args, name) < 0.0:
            parser.error(f"--{name} 不能为负数")
    args.save_dir = resolve_project_path(args.save_dir)
    return args


def main():
    args = parse_args()
    log_path, log_file, original_stdout, original_stderr = setup_console_logging(args.save_dir)
    try:
        cuda_available = torch.cuda.is_available()
        if args.cuda and not cuda_available:
            raise RuntimeError("已要求使用 CUDA，但当前 PyTorch 未检测到可用 GPU")
        use_cuda = cuda_available and not args.cpu
        device = torch.device("cuda" if use_cuda else "cpu")
        args.cuda = use_cuda
        args.amp = use_cuda and not args.no_amp
        args.gpu_data_cache = use_cuda and not args.no_gpu_data_cache
        args.cache_frozen_views = not args.no_frozen_view_cache
        args.fused_adam = use_cuda and not args.no_fused_adam
        args.tf32 = use_cuda and not args.no_tf32
        args.cudnn_benchmark = (
            use_cuda and not args.no_cudnn_benchmark and not args.deterministic
        )
        if use_cuda:
            torch.set_float32_matmul_precision("high" if args.tf32 else "highest")
            torch.backends.cuda.matmul.allow_tf32 = args.tf32
            torch.backends.cudnn.allow_tf32 = args.tf32
        print(f"[Device] {device}")
        if use_cuda:
            print(f"[Device] gpu={torch.cuda.get_device_name(0)}")
        print(f"[Log] {log_path}")
        train_phase2(args, device)
    finally:
        close_console_logging(log_file, original_stdout, original_stderr)


if __name__ == "__main__":
    main()
