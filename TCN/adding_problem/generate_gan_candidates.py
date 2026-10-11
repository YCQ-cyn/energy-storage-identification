# -*- coding: utf-8 -*-
"""生成离线 GAN 弱储能候选数据，不接入正式训练。"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw

try:
    from .teacher_targets import (
        BEHAVIOR_TARGET_NAMES,
        PHYSICAL_TARGET_NAMES,
        RESPONSE_TARGET_NAMES,
        TARGET_PIPELINE_VERSION,
        TeacherTargetConfig,
        compute_stage2_targets,
    )
    from .utils import load_storage_data, scale_pq
except ImportError:
    from teacher_targets import (
        BEHAVIOR_TARGET_NAMES,
        PHYSICAL_TARGET_NAMES,
        RESPONSE_TARGET_NAMES,
        TARGET_PIPELINE_VERSION,
        TeacherTargetConfig,
        compute_stage2_targets,
    )
    from utils import load_storage_data, scale_pq


GAN_CANDIDATE_VERSION = "physics-guided-residual-wgan-gp-v1"
ROOT = Path(__file__).resolve().parent


class ResidualGenerator(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.project = nn.Linear(self.latent_dim, 128 * 12)
        self.network = nn.Sequential(
            nn.ConvTranspose1d(128, 96, 4, 2, 1),
            nn.BatchNorm1d(96),
            nn.SiLU(),
            nn.ConvTranspose1d(96, 64, 4, 2, 1),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.ConvTranspose1d(64, 32, 4, 2, 1),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Conv1d(32, 1, 5, padding=2),
            nn.Tanh(),
        )

    def forward(self, noise):
        hidden = self.project(noise).reshape(len(noise), 128, 12)
        return self.network(hidden).squeeze(1)


class ResidualCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 32, 7, 2, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, 7, 2, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 96, 5, 2, 2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(96, 128, 5, 2, 2),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 6, 1),
        )

    def forward(self, curves):
        return self.network(curves[:, None]).squeeze(1)


def resolve_path(path):
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def moving_average(curves, window=9):
    padding = window // 2
    padded = np.pad(curves, ((0, 0), (padding, padding)), mode="edge")
    prefix = np.pad(np.cumsum(padded, axis=1), ((0, 0), (1, 0)))
    return ((prefix[:, window:] - prefix[:, :-window]) / float(window)).astype(
        np.float32
    )


def gradient_penalty(critic, real_curves, fake_curves):
    ratio = torch.rand(len(real_curves), 1, device=real_curves.device)
    mixed = ratio * real_curves + (1.0 - ratio) * fake_curves
    mixed.requires_grad_(True)
    scores = critic(mixed)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=mixed,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return ((gradients.flatten(1).norm(2, dim=1) - 1.0) ** 2).mean()


def train_wgan(residuals, args, device):
    generator = ResidualGenerator(args.latent_dim).to(device)
    critic = ResidualCritic().to(device)
    generator_optimizer = torch.optim.Adam(
        generator.parameters(), lr=args.generator_lr, betas=(0.0, 0.9)
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=args.critic_lr, betas=(0.0, 0.9)
    )
    residual_tensor = torch.from_numpy(residuals.astype(np.float32))
    rng = np.random.default_rng(args.seed)
    history = []
    for epoch in range(1, args.gan_epochs + 1):
        order = rng.permutation(len(residual_tensor))
        critic_losses = []
        generator_losses = []
        for start in range(0, len(order), args.gan_batch_size):
            indices = order[start:start + args.gan_batch_size]
            if len(indices) < 8:
                continue
            real_curves = residual_tensor[indices].to(device)
            for _ in range(args.critic_steps):
                noise = torch.randn(
                    len(indices), generator.latent_dim, device=device
                )
                fake_curves = generator(noise).detach()
                critic_loss = (
                    critic(fake_curves).mean()
                    - critic(real_curves).mean()
                    + args.gradient_penalty * gradient_penalty(
                        critic, real_curves, fake_curves
                    )
                )
                critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                critic_optimizer.step()
            noise = torch.randn(len(indices), generator.latent_dim, device=device)
            generator_loss = -critic(generator(noise)).mean()
            generator_optimizer.zero_grad(set_to_none=True)
            generator_loss.backward()
            generator_optimizer.step()
            critic_losses.append(float(critic_loss.detach()))
            generator_losses.append(float(generator_loss.detach()))
        row = {
            "epoch": epoch,
            "critic_loss": float(np.mean(critic_losses)),
            "generator_loss": float(np.mean(generator_losses)),
        }
        history.append(row)
        print(
            f"[GAN] epoch={epoch:02d} critic={row['critic_loss']:.4f} "
            f"generator={row['generator_loss']:.4f}"
        )
    return generator.eval(), history


@torch.inference_mode()
def generate_residuals(generator, count, device, seed):
    random = np.random.default_rng(seed)
    generated = []
    for start in range(0, count, 512):
        current = min(512, count - start)
        noise = torch.from_numpy(
            random.standard_normal((current, generator.latent_dim)).astype(np.float32)
        ).to(device)
        generated.append(generator(noise).cpu().numpy())
    return np.concatenate(generated).astype(np.float32)


def target_config_from_checkpoint(path):
    if not path.is_file():
        return TeacherTargetConfig(), {}
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved = checkpoint.get("args", {})
    supported = {
        name: saved[name]
        for name in TeacherTargetConfig.__dataclass_fields__
        if name in saved
    }
    return TeacherTargetConfig(**supported), {
        "checkpoint": str(path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "stage2_objective_version": checkpoint.get("stage2_objective_version", ""),
    }


def target_indices():
    return {
        "evidence": PHYSICAL_TARGET_NAMES.index("storage_evidence"),
        "response": RESPONSE_TARGET_NAMES.index("response_confidence"),
        "behavior": BEHAVIOR_TARGET_NAMES.index("storage_behavior"),
    }


def energy_balance(q_curves):
    absolute = np.abs(q_curves).sum(axis=1)
    return (1.0 - np.abs(q_curves.sum(axis=1)) / np.maximum(absolute, 1e-6)).astype(
        np.float32
    )


def sign_transitions(q_curves):
    threshold = np.maximum(np.percentile(np.abs(q_curves), 95, axis=1) * 0.02, 1e-6)
    sign = np.where(q_curves > threshold[:, None], 1, 0)
    sign = np.where(q_curves < -threshold[:, None], -1, sign)
    transitions = np.zeros(len(q_curves), dtype=np.float32)
    for index, row in enumerate(sign):
        active = row[row != 0]
        transitions[index] = float(np.count_nonzero(active[1:] != active[:-1]))
    return transitions


def summarize_curves(p_curves, q_curves):
    return {
        "samples": int(len(p_curves)),
        "p_mean": float(p_curves.mean()),
        "p_p95": float(np.percentile(p_curves, 95)),
        "q_abs_mean": float(np.abs(q_curves).mean()),
        "q_sign_transitions_mean": float(sign_transitions(q_curves).mean()),
        "q_energy_balance_mean": float(energy_balance(q_curves).mean()),
        "q_energy_balance_p10": float(np.quantile(energy_balance(q_curves), 0.10)),
    }


def normalize_inputs(p_raw, q_raw, data, anchor_global):
    q_global_scale = float(data["normalization"]["q_global_scale"])
    inputs = np.empty((len(p_raw), 5, 96), dtype=np.float32)
    for output_index, source_index in enumerate(anchor_global):
        power, storage, _ = scale_pq(
            p_raw[output_index], q_raw[output_index], q_global_scale=q_global_scale
        )
        inputs[output_index] = np.stack((
            power,
            storage,
            data["X"][source_index, 2].numpy(),
            data["X"][source_index, 3].numpy(),
            data["X"][source_index, 4].numpy(),
        ))
    return inputs


def build_candidates(data, config, generator, residual_scale, args, device):
    train_indices = data["split_indices"]["train"]
    p_train = data["P_raw"][train_indices].astype(np.float32)
    q_train = data["Q_raw"][train_indices].astype(np.float32)
    role_train = data["X"][train_indices, 3].numpy().astype(np.float32)
    targets = compute_stage2_targets(p_train, q_train, role_train, config=config)
    indices = target_indices()
    physical = targets["physical_targets"]
    response = targets["response_targets"]
    behavior = targets["behavior_targets"]
    thresholds = {
        "behavior_q05": float(np.quantile(behavior[:, indices["behavior"]], 0.05)),
        "behavior_q30": float(np.quantile(behavior[:, indices["behavior"]], 0.30)),
        "behavior_q40": float(np.quantile(behavior[:, indices["behavior"]], 0.40)),
        "evidence_q10": float(np.quantile(physical[:, indices["evidence"]], 0.10)),
        "evidence_q30": float(np.quantile(physical[:, indices["evidence"]], 0.30)),
        "response_q10": float(np.quantile(response[:, indices["response"]], 0.10)),
        "response_q30": float(np.quantile(response[:, indices["response"]], 0.30)),
    }
    clear = (
        (behavior[:, indices["behavior"]] >= thresholds["behavior_q40"])
        & (physical[:, indices["evidence"]] >= thresholds["evidence_q30"])
        & (response[:, indices["response"]] >= thresholds["response_q30"])
        & (q_train > 0.0).any(axis=1)
        & (q_train < 0.0).any(axis=1)
    )
    clear_local = np.flatnonzero(clear)
    if not len(clear_local):
        raise RuntimeError("训练集中没有满足条件的清晰储能锚点")

    random = np.random.default_rng(args.seed + 101)
    anchor_local = random.choice(clear_local, size=args.candidate_count, replace=True)
    anchor_global = train_indices[anchor_local]
    p_anchor = p_train[anchor_local]
    q_anchor = q_train[anchor_local]
    p_reference = np.maximum(np.percentile(p_anchor, 95, axis=1), 1e-6)
    base_relative = (p_anchor - q_anchor) / p_reference[:, None]
    base_trend = moving_average(base_relative, args.trend_window)
    original_residual = base_relative - base_trend
    generated_residual = generate_residuals(
        generator, args.candidate_count, device, args.seed + 202
    ) * residual_scale
    mixed_residual = (
        args.real_residual_weight * original_residual
        + (1.0 - args.real_residual_weight) * generated_residual
    )
    base_synthetic = np.maximum(
        (base_trend + mixed_residual) * p_reference[:, None], 0.0
    ).astype(np.float32)
    alpha = random.uniform(
        args.alpha_min, args.alpha_max, size=args.candidate_count
    ).astype(np.float32)
    q_synthetic = (q_anchor * alpha[:, None]).astype(np.float32)
    p_unclipped = base_synthetic + q_synthetic
    clipped_fraction = (p_unclipped < 0.0).mean(axis=1).astype(np.float32)
    p_synthetic = np.maximum(p_unclipped, 0.0).astype(np.float32)
    role_synthetic = data["X"][anchor_global, 3].numpy().astype(np.float32)
    synthetic_targets = compute_stage2_targets(
        p_synthetic, q_synthetic, role_synthetic, config=config
    )
    physical = synthetic_targets["physical_targets"]
    response = synthetic_targets["response_targets"]
    behavior = synthetic_targets["behavior_targets"]
    physical_weight = synthetic_targets["physical_weights"]
    behavior_weight = synthetic_targets["behavior_weights"]
    accepted = (
        np.isfinite(p_synthetic).all(axis=1)
        & np.isfinite(q_synthetic).all(axis=1)
        & (clipped_fraction <= args.max_clipped_fraction)
        & (behavior[:, indices["behavior"]] >= thresholds["behavior_q05"])
        & (behavior[:, indices["behavior"]] <= thresholds["behavior_q30"])
        & (physical[:, indices["evidence"]] >= thresholds["evidence_q10"])
        & (response[:, indices["response"]] >= thresholds["response_q10"])
        & (physical_weight[:, indices["evidence"]] >= 0.35)
        & (behavior_weight[:, indices["behavior"]] >= 0.35)
    )
    selected = np.flatnonzero(accepted)
    if len(selected) > args.output_count:
        selected = random.choice(selected, size=args.output_count, replace=False)
    selected = np.sort(selected)
    selected_anchor_global = anchor_global[selected]
    result = {
        "inputs": normalize_inputs(
            p_synthetic[selected], q_synthetic[selected], data, selected_anchor_global
        ),
        "p_raw": p_synthetic[selected],
        "q_raw": q_synthetic[selected],
        "base_load_raw": base_synthetic[selected],
        "alpha": alpha[selected],
        "anchor_sample_index": selected_anchor_global.astype(np.int64),
        "anchor_user_id": np.asarray(
            data["meta"].iloc[selected_anchor_global]["user_id"].astype(str).tolist(),
            dtype=np.str_,
        ),
        "clipped_fraction": clipped_fraction[selected],
        "physical_targets": synthetic_targets["physical_targets"][selected],
        "physical_weights": synthetic_targets["physical_weights"][selected],
        "response_targets": synthetic_targets["response_targets"][selected],
        "response_weights": synthetic_targets["response_weights"][selected],
        "behavior_targets": synthetic_targets["behavior_targets"][selected],
        "behavior_weights": synthetic_targets["behavior_weights"][selected],
        "usage_flag": np.asarray(["offline_candidate_only"]),
        "dataset_version": np.asarray([GAN_CANDIDATE_VERSION]),
    }
    report = {
        "thresholds": thresholds,
        "clear_anchor_count": int(len(clear_local)),
        "candidate_count": int(args.candidate_count),
        "accepted_before_limit": int(accepted.sum()),
        "saved_count": int(len(selected)),
        "acceptance_rate": float(accepted.mean()),
        "alpha_mean": float(alpha[selected].mean()) if len(selected) else 0.0,
        "behavior_mean": float(
            behavior[selected, indices["behavior"]].mean()
        ) if len(selected) else 0.0,
        "evidence_mean": float(
            physical[selected, indices["evidence"]].mean()
        ) if len(selected) else 0.0,
        "response_mean": float(
            response[selected, indices["response"]].mean()
        ) if len(selected) else 0.0,
        "real_anchor_quality": summarize_curves(p_anchor[selected], q_anchor[selected])
        if len(selected) else {},
        "generated_quality": summarize_curves(
            p_synthetic[selected], q_synthetic[selected]
        ) if len(selected) else {},
    }
    return result, report


def save_npz(path, arrays):
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def save_plot(path, arrays, seed):
    if not len(arrays["p_raw"]):
        return
    random = np.random.default_rng(seed)
    selected = random.choice(
        len(arrays["p_raw"]), size=min(6, len(arrays["p_raw"])), replace=False
    )
    width = 1400
    row_height = 250
    image = Image.new("RGB", (width, row_height * len(selected)), "white")
    draw = ImageDraw.Draw(image)
    left, right = 65, width - 30
    for row_index, sample_index in enumerate(selected):
        top = row_index * row_height + 30
        bottom = (row_index + 1) * row_height - 25
        power = arrays["p_raw"][sample_index]
        storage = arrays["q_raw"][sample_index]
        lower = min(float(storage.min()), 0.0)
        upper = max(float(power.max()), float(storage.max()), 1e-6)
        scale = max(upper - lower, 1e-6)

        def points(values):
            return [
                (
                    left + int(index * (right - left) / 95),
                    bottom - int((float(value) - lower) * (bottom - top) / scale),
                )
                for index, value in enumerate(values)
            ]

        zero_y = bottom - int((0.0 - lower) * (bottom - top) / scale)
        draw.line((left, zero_y, right, zero_y), fill=(170, 170, 170), width=1)
        draw.line(points(power), fill=(35, 95, 190), width=2)
        draw.line(points(storage), fill=(210, 55, 55), width=2)
        draw.text(
            (left, row_index * row_height + 8),
            f"candidate={sample_index}  alpha={arrays['alpha'][sample_index]:.2f}  "
            "blue=P  red=Q",
            fill=(20, 20, 20),
        )
    image.save(path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="生成物理约束的离线 GAN 弱储能候选数据"
    )
    parser.add_argument("--data_path", default="processed_result.xlsx")
    parser.add_argument("--split_manifest", default="data_split/user_split.csv")
    parser.add_argument("--user_col", default="关联用户号")
    parser.add_argument("--phase2_ckpt", default="step_2/best_model_st2.pt")
    parser.add_argument("--output_dir", default="gan_offline_candidates")
    parser.add_argument("--candidate_count", type=int, default=12000)
    parser.add_argument("--output_count", type=int, default=750)
    parser.add_argument("--gan_epochs", type=int, default=8)
    parser.add_argument("--gan_batch_size", type=int, default=256)
    parser.add_argument("--max_gan_train_samples", type=int, default=20000)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--critic_steps", type=int, default=3)
    parser.add_argument("--gradient_penalty", type=float, default=10.0)
    parser.add_argument("--generator_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=2e-4)
    parser.add_argument("--alpha_min", type=float, default=0.25)
    parser.add_argument("--alpha_max", type=float, default=0.78)
    parser.add_argument("--real_residual_weight", type=float, default=0.55)
    parser.add_argument("--trend_window", type=int, default=9)
    parser.add_argument("--max_clipped_fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        args.candidate_count = 256
        args.output_count = 32
        args.gan_epochs = 1
        args.gan_batch_size = 64
        args.max_gan_train_samples = 512
        args.critic_steps = 1
    if args.candidate_count <= 0 or args.output_count <= 0:
        parser.error("候选数和输出数必须为正数")
    if args.output_count > args.candidate_count:
        parser.error("输出数不能超过候选数")
    if not 0.0 < args.alpha_min < args.alpha_max <= 1.0:
        parser.error("alpha 范围必须满足 0 < min < max <= 1")
    if args.trend_window < 3 or args.trend_window % 2 == 0:
        parser.error("趋势窗口必须是不小于3的奇数")
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.init()
    data = load_storage_data(
        data_path=args.data_path,
        split_manifest_path=args.split_manifest,
        user_col=args.user_col,
        policy_name="old",
        seed=args.seed,
    )
    train_indices = data["split_indices"]["train"]
    p_train = data["P_raw"][train_indices].astype(np.float32)
    q_train = data["Q_raw"][train_indices].astype(np.float32)
    p_reference = np.maximum(np.percentile(p_train, 95, axis=1), 1e-6)
    base_relative = (p_train - q_train) / p_reference[:, None]
    residual = base_relative - moving_average(base_relative, args.trend_window)
    residual_scale = max(float(np.quantile(np.abs(residual), 0.99)), 1e-6)
    residual_normalized = np.clip(
        residual / residual_scale, -1.0, 1.0
    ).astype(np.float32)
    if len(residual_normalized) > args.max_gan_train_samples:
        random = np.random.default_rng(args.seed)
        selected = random.choice(
            len(residual_normalized), args.max_gan_train_samples, replace=False
        )
        residual_normalized = residual_normalized[selected]

    generator, history = train_wgan(residual_normalized, args, device)
    config, checkpoint_info = target_config_from_checkpoint(
        resolve_path(args.phase2_ckpt)
    )
    arrays, candidate_report = build_candidates(
        data, config, generator, residual_scale, args, device
    )
    dataset_path = output_dir / "gan_weak_storage_candidates.npz"
    report_path = output_dir / "gan_generation_report.json"
    model_path = output_dir / "residual_wgan_gp.pt"
    plot_path = output_dir / "gan_candidate_examples.png"
    save_npz(dataset_path, arrays)
    torch.save({
        "version": GAN_CANDIDATE_VERSION,
        "generator_state_dict": generator.state_dict(),
        "latent_dim": generator.latent_dim,
        "residual_scale": residual_scale,
        "history": history,
        "args": vars(args),
    }, model_path)
    save_plot(plot_path, arrays, args.seed)
    report = {
        "version": GAN_CANDIDATE_VERSION,
        "purpose": "生成弱储能与边界行为离线候选数据",
        "usage_status": "offline_candidate_only",
        "connected_to_formal_training": False,
        "validation_test_augmented": False,
        "method": (
            "WGAN-GP生成基础负荷残差，真实Q按alpha弱化后进行物理组合，"
            "再使用Stage-2行为目标筛选"
        ),
        "target_pipeline_version": TARGET_PIPELINE_VERSION,
        "target_config": asdict(config),
        "checkpoint_info": checkpoint_info,
        "device": str(device),
        "gan_train_samples": int(len(residual_normalized)),
        "residual_scale": residual_scale,
        "gan_history": history,
        "candidate_report": candidate_report,
        "outputs": {
            "dataset": str(dataset_path),
            "generator": str(model_path),
            "examples": str(plot_path),
        },
        "formal_training_note": (
            "该文件不会被Stage-1、Stage-2或学生训练入口自动读取。"
            "只有后续真实留出集实验稳定受益时才允许显式接入。"
        ),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "dataset": str(dataset_path),
        "saved_count": candidate_report["saved_count"],
        "acceptance_rate": candidate_report["acceptance_rate"],
        "connected_to_formal_training": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
