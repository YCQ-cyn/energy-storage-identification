# -*- coding: utf-8 -*-
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    from . import add_test_st2 as st2
    from .model import TCN_LSTM_ARCH_VERSION
    from .teacher_targets import (
        BEHAVIOR_TARGET_NAMES,
        PHYSICAL_TARGET_NAMES,
        RESPONSE_TARGET_NAMES,
        TARGET_PIPELINE_VERSION,
    )
    from .utils import load_storage_data, scale_P
except ImportError:
    import add_test_st2 as st2
    from model import TCN_LSTM_ARCH_VERSION
    from teacher_targets import (
        BEHAVIOR_TARGET_NAMES,
        PHYSICAL_TARGET_NAMES,
        RESPONSE_TARGET_NAMES,
        TARGET_PIPELINE_VERSION,
    )
    from utils import load_storage_data, scale_P


STUDENT_DATASET_VERSION = "student-distill-v2-structured-current-stage2"
STUDENT_CHANNEL_NAMES = (
    "P_norm",
    "dP_pos",
    "dP_neg",
    "price_role_seq",
    "price_role_jump",
)
STUDENT_TARGET_NAMES = (
    "storage_behavior",
    "storage_evidence",
    "response_confidence",
    "activity_strength",
    "charge_visible",
    "discharge_visible",
)
SPLIT_TO_ID = {"train": 0, "val": 1, "test": 2}
SOURCE_TO_ID = {"observed_old": 0, "policy_transfer": 1}


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


def validate_teacher_checkpoint(checkpoint):
    if checkpoint.get("target_pipeline_version") != TARGET_PIPELINE_VERSION:
        raise RuntimeError("Stage-2 checkpoint 的教师目标版本不匹配")
    if checkpoint.get("stage2_objective_version") != st2.STAGE2_OBJECTIVE_VERSION:
        raise RuntimeError("Stage-2 checkpoint 的训练目标版本不匹配")
    if checkpoint.get("model_arch_version") != TCN_LSTM_ARCH_VERSION:
        raise RuntimeError("Stage-2 checkpoint 的网络结构版本不匹配")
    if tuple(checkpoint.get("physical_target_names", ())) != tuple(PHYSICAL_TARGET_NAMES):
        raise RuntimeError("Stage-2 physical head 语义不匹配")
    if tuple(checkpoint.get("response_target_names", ())) != tuple(RESPONSE_TARGET_NAMES):
        raise RuntimeError("Stage-2 response head 语义不匹配")
    if tuple(checkpoint.get("behavior_target_names", ())) != tuple(BEHAVIOR_TARGET_NAMES):
        raise RuntimeError("Stage-2 behavior head 语义不匹配")


def load_teacher(path, device):
    checkpoint = load_checkpoint(path, device)
    validate_teacher_checkpoint(checkpoint)
    model_kwargs = checkpoint.get("model_kwargs")
    if not model_kwargs:
        raise RuntimeError("Stage-2 checkpoint 缺少 model_kwargs")
    teacher = st2.EncoderWithHead(**model_kwargs).to(device)
    teacher.load_state_dict(checkpoint["model_state_dict"], strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher, checkpoint


@torch.inference_mode()
def collect_teacher_targets(model, inputs, device, batch_size, use_amp):
    rows = []
    for start in range(0, len(inputs), batch_size):
        batch = torch.from_numpy(inputs[start:start + batch_size]).to(device)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            physical_embedding = model.encode(st2.mask_physical_context(batch))
            behavior_embedding = model.encode(st2.mask_behavior_season(batch))
            physical_norm = F.normalize(physical_embedding, dim=1)
            behavior_norm = F.normalize(behavior_embedding, dim=1)
            physical = torch.sigmoid(model.physical_head(physical_norm))
            response = torch.sigmoid(model.response_head(physical_norm))
            behavior = torch.sigmoid(model.behavior_head(behavior_norm))
            selected = torch.stack((
                behavior[:, BEHAVIOR_TARGET_NAMES.index("storage_behavior")],
                physical[:, PHYSICAL_TARGET_NAMES.index("storage_evidence")],
                response[:, RESPONSE_TARGET_NAMES.index("response_confidence")],
                physical[:, PHYSICAL_TARGET_NAMES.index("activity_strength")],
                response[:, RESPONSE_TARGET_NAMES.index("charge_visible")],
                response[:, RESPONSE_TARGET_NAMES.index("discharge_visible")],
            ), dim=1)
        rows.append(selected.float().cpu())
    return torch.cat(rows).numpy().astype(np.float32)


def make_pseudo_b(positive_inputs, power_raw, storage_raw):
    pseudo = positive_inputs.copy()
    pseudo[:, 0] = np.stack([
        scale_P((power_raw[index] - storage_raw[index]).astype(np.float32))
        for index in range(len(positive_inputs))
    ]).astype(np.float32)
    pseudo[:, 1] = 0.0
    return pseudo


def estimate_dp_clip(inputs, train_mask):
    power = inputs[train_mask, 0]
    difference = np.diff(power, axis=1, prepend=power[:, :1])
    return max(float(np.quantile(np.abs(difference), 0.99)), 1e-6)


def build_student_channels(inputs, dp_clip):
    power = inputs[:, 0]
    difference = np.zeros_like(power, dtype=np.float32)
    difference[:, 1:] = power[:, 1:] - power[:, :-1]
    positive = np.clip(difference, 0.0, dp_clip) / dp_clip
    negative = np.clip(-difference, 0.0, dp_clip) / dp_clip
    return np.stack((
        power,
        positive,
        negative,
        inputs[:, 3],
        inputs[:, 4],
    ), axis=1).astype(np.float32)


def load_transfer_cache(path, expected_length):
    with np.load(path, allow_pickle=False) as cache:
        arrays = {name: cache[name].copy() for name in cache.files}
    required = {
        "positive_inputs",
        "positive_p_raw",
        "positive_q_raw",
        "mismatch_role_seq",
        "mismatch_role_jump",
        "transfer_weight",
        "accepted",
        "policy_shift",
    }
    missing = required - set(arrays)
    if missing:
        raise KeyError(f"政策迁移缓存缺少字段: {sorted(missing)}")
    if len(arrays["positive_inputs"]) != expected_length:
        raise RuntimeError("政策迁移缓存与当前 Stage-2 数据顺序不一致")
    return arrays


def build_pair_table(data, transfer, min_transfer_shift):
    meta = data["meta"].reset_index(drop=True)
    split = meta["split"].astype(str).to_numpy()
    original_index = np.arange(len(meta), dtype=np.int64)
    shifted = np.abs(transfer["policy_shift"].astype(np.int64)) >= int(min_transfer_shift)
    transfer_index = np.flatnonzero(transfer["accepted"].astype(bool) & shifted).astype(np.int64)

    source_index = np.concatenate((original_index, transfer_index))
    source_type = np.concatenate((
        np.zeros(len(original_index), dtype=np.int64),
        np.ones(len(transfer_index), dtype=np.int64),
    ))
    pair_split = np.concatenate((split, split[transfer_index]))
    transfer_quality = np.concatenate((
        np.ones(len(original_index), dtype=np.float32),
        transfer["transfer_weight"][transfer_index].astype(np.float32),
    ))
    policy_weight = np.concatenate((
        np.zeros(len(original_index), dtype=np.float32),
        transfer["transfer_weight"][transfer_index].astype(np.float32),
    ))
    policy_shift = np.concatenate((
        np.zeros(len(original_index), dtype=np.int64),
        transfer["policy_shift"][transfer_index].astype(np.int64),
    ))
    table = pd.DataFrame({
        "pair_index": np.arange(len(source_index), dtype=np.int64),
        "source_index": source_index,
        "source_type": source_type,
        "source_type_name": np.where(source_type == 0, "observed_old", "policy_transfer"),
        "split": pair_split,
        "split_id": np.asarray([SPLIT_TO_ID[value] for value in pair_split], dtype=np.int64),
        "user_id": meta.iloc[source_index]["user_id"].astype(str).to_numpy(),
        "date": meta.iloc[source_index]["date"].astype(str).to_numpy(),
        "month": meta.iloc[source_index]["month"].to_numpy(np.int64),
        "source_row": meta.iloc[source_index]["source_row"].to_numpy(np.int64),
        "duplicate_weight": meta.iloc[source_index]["duplicate_weight"].to_numpy(np.float32),
        "transfer_quality": transfer_quality,
        "policy_weight": policy_weight,
        "policy_shift": policy_shift,
    })
    return table


def limit_pairs(table, limits, seed):
    rng = np.random.default_rng(seed)
    selected = []
    for split_name, maximum in limits.items():
        indices = table.index[table["split"] == split_name].to_numpy(np.int64)
        if maximum > 0 and len(indices) > maximum:
            source = table.loc[indices, "source_type"].to_numpy(np.int64)
            old_indices = indices[source == 0]
            transfer_indices = indices[source == 1]
            transfer_take = min(len(transfer_indices), int(round(maximum * 0.40)))
            old_take = min(len(old_indices), maximum - transfer_take)
            remaining = maximum - old_take - transfer_take
            if remaining > 0:
                transfer_take = min(len(transfer_indices), transfer_take + remaining)
            chosen = np.concatenate((
                rng.choice(old_indices, size=old_take, replace=False),
                rng.choice(transfer_indices, size=transfer_take, replace=False),
            ))
            indices = np.sort(chosen)
        selected.append(indices)
    selected = np.concatenate(selected)
    result = table.loc[selected].copy().reset_index(drop=True)
    result["pair_index"] = np.arange(len(result), dtype=np.int64)
    return result


def build_teacher_inputs(data, transfer, table):
    source_index = table["source_index"].to_numpy(np.int64)
    source_type = table["source_type"].to_numpy(np.int64)
    positive = np.empty((len(table), 5, 96), dtype=np.float32)
    power_raw = np.empty((len(table), 96), dtype=np.float32)
    storage_raw = np.empty((len(table), 96), dtype=np.float32)

    observed = source_type == SOURCE_TO_ID["observed_old"]
    relocated = ~observed
    positive[observed] = data["X"].numpy()[source_index[observed]]
    power_raw[observed] = data["P_raw"][source_index[observed]]
    storage_raw[observed] = data["Q_raw"][source_index[observed]]
    positive[relocated] = transfer["positive_inputs"][source_index[relocated]]
    power_raw[relocated] = transfer["positive_p_raw"][source_index[relocated]]
    storage_raw[relocated] = transfer["positive_q_raw"][source_index[relocated]]

    mismatch = positive.copy()
    mismatch[relocated, 3] = transfer["mismatch_role_seq"][source_index[relocated]]
    mismatch[relocated, 4] = transfer["mismatch_role_jump"][source_index[relocated]]
    pseudo = make_pseudo_b(positive, power_raw, storage_raw)
    return positive, pseudo, mismatch


def add_balance_weights(table):
    group_count = table.groupby(["split", "user_id"])["pair_index"].transform("size").to_numpy()
    raw = table["duplicate_weight"].to_numpy(np.float32) / np.maximum(group_count, 1)
    loss_weight = np.empty(len(table), dtype=np.float32)
    sampling_weight = np.empty(len(table), dtype=np.float32)
    for split_name in SPLIT_TO_ID:
        mask = table["split"].to_numpy() == split_name
        split_raw = raw[mask]
        split_raw = split_raw / max(float(split_raw.mean()), 1e-8)
        sampling_weight[mask] = np.sqrt(split_raw)
        quality = table.loc[mask, "transfer_quality"].to_numpy(np.float32)
        loss_weight[mask] = np.sqrt(split_raw * np.clip(quality, 0.05, 1.0))
    table["sampling_weight"] = sampling_weight
    table["loss_weight"] = loss_weight
    return table


def encode_user_ids(table):
    values = table["user_id"].astype(str)
    unique_users = sorted(values.unique().tolist())
    mapping = {user_id: index for index, user_id in enumerate(unique_users)}
    return values.map(mapping).to_numpy(np.int64)


def validate_arrays(arrays, table):
    pair_count = len(table)
    for name in ("positive_inputs", "pseudo_inputs", "mismatch_inputs"):
        value = arrays[name]
        if value.shape != (pair_count, 5, 96):
            raise AssertionError(f"{name} shape 异常: {value.shape}")
        if not np.isfinite(value).all():
            raise AssertionError(f"{name} 存在 NaN/Inf")
    for name in ("positive_targets", "pseudo_targets", "mismatch_targets"):
        value = arrays[name]
        if value.shape != (pair_count, len(STUDENT_TARGET_NAMES)):
            raise AssertionError(f"{name} shape 异常: {value.shape}")
        if not np.isfinite(value).all():
            raise AssertionError(f"{name} 存在 NaN/Inf")
    if set(np.unique(arrays["split_id"])) != {0, 1, 2}:
        raise AssertionError("学生数据必须同时包含 train、val、test")
    if np.any(arrays["positive_inputs"][:, 3] < -1.0001) or np.any(
        arrays["positive_inputs"][:, 3] > 1.0001
    ):
        raise AssertionError("相对电价角色范围异常")
    split_per_user = table.groupby("user_id")["split"].nunique()
    if (split_per_user > 1).any():
        raise AssertionError("发现用户跨 split 泄漏")


def summarize_targets(targets):
    rows = []
    for index, name in enumerate(STUDENT_TARGET_NAMES):
        values = targets[:, index]
        rows.append({
            "name": name,
            "mean": float(values.mean()),
            "std": float(values.std()),
            "q05": float(np.quantile(values, 0.05)),
            "q50": float(np.quantile(values, 0.50)),
            "q95": float(np.quantile(values, 0.95)),
        })
    return rows


def save_dataset(args, checkpoint, data, table, positive, pseudo, mismatch,
                 positive_targets, pseudo_targets, mismatch_targets, dp_clip, elapsed):
    output_dir = resolve_path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher_checkpoint_path = resolve_path(args.phase2_ckpt)
    teacher_checkpoint_sha256 = file_sha256(teacher_checkpoint_path)
    teacher_stage2_score = float(
        checkpoint.get("best_score", checkpoint.get("score", 0.0)) or 0.0
    )
    arrays = {
        "positive_inputs": build_student_channels(positive, dp_clip),
        "pseudo_inputs": build_student_channels(pseudo, dp_clip),
        "mismatch_inputs": build_student_channels(mismatch, dp_clip),
        "positive_targets": positive_targets,
        "pseudo_targets": pseudo_targets,
        "mismatch_targets": mismatch_targets,
        "split_id": table["split_id"].to_numpy(np.int64),
        "source_type": table["source_type"].to_numpy(np.int64),
        "source_index": table["source_index"].to_numpy(np.int64),
        "user_code": encode_user_ids(table),
        "month": table["month"].to_numpy(np.int64),
        "loss_weight": table["loss_weight"].to_numpy(np.float32),
        "sampling_weight": table["sampling_weight"].to_numpy(np.float32),
        "policy_weight": table["policy_weight"].to_numpy(np.float32),
        "policy_shift": table["policy_shift"].to_numpy(np.int64),
        "dp_clip_q99": np.asarray([dp_clip], dtype=np.float32),
        "dataset_version": np.asarray(STUDENT_DATASET_VERSION),
        "teacher_epoch": np.asarray([checkpoint.get("epoch", -1)], dtype=np.int64),
        "teacher_stage2_score": np.asarray([teacher_stage2_score], dtype=np.float32),
        "teacher_checkpoint_sha256": np.asarray(teacher_checkpoint_sha256),
    }
    validate_arrays(arrays, table)
    dataset_path = output_dir / "student_train_dataset.npz"
    temporary_path = dataset_path.with_name(dataset_path.name + ".tmp.npz")
    np.savez_compressed(temporary_path, **arrays)
    temporary_path.replace(dataset_path)
    table.to_csv(output_dir / "student_meta.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summarize_targets(positive_targets)).to_csv(
        output_dir / "student_target_summary.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "verdict": "PASS",
        "dataset_version": STUDENT_DATASET_VERSION,
        "teacher_checkpoint": str(teacher_checkpoint_path),
        "teacher_checkpoint_sha256": teacher_checkpoint_sha256,
        "teacher_epoch": int(checkpoint.get("epoch", -1)),
        "teacher_stage2_score": teacher_stage2_score,
        "channel_names": list(STUDENT_CHANNEL_NAMES),
        "target_names": list(STUDENT_TARGET_NAMES),
        "pair_count": int(len(table)),
        "split_counts": {
            split_name: int((table["split"] == split_name).sum())
            for split_name in SPLIT_TO_ID
        },
        "source_counts": {
            source_name: int((table["source_type"] == source_id).sum())
            for source_name, source_id in SOURCE_TO_ID.items()
        },
        "user_counts": {
            split_name: int(table.loc[table["split"] == split_name, "user_id"].nunique())
            for split_name in SPLIT_TO_ID
        },
        "dp_clip_q99": float(dp_clip),
        "normalization": data["normalization"],
        "policy_note": "不输入 season；仅使用当前政策的相对电价角色与角色跳变",
        "pseudo_note": "B=P-Q，仅用于软目标与成对排序；A类伪负不进入学生",
        "elapsed_seconds": float(elapsed),
    }
    (output_dir / "student_dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return dataset_path, summary


def parse_args():
    parser = argparse.ArgumentParser(description="当前 Stage-2 下游学生蒸馏数据构建")
    parser.add_argument("--phase2_ckpt", default="step_2/best_model_st2.pt")
    parser.add_argument("--data_path", default="processed_result.xlsx")
    parser.add_argument("--split_manifest", default="data_split/user_split.csv")
    parser.add_argument("--transfer_cache", default="step_2/policy_transfer_v3.npz")
    parser.add_argument("--user_col", default="关联用户号")
    parser.add_argument("--out_dir", default="step_3")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--min_transfer_shift", type=int, default=8)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.perf_counter()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"[StudentData] device={device}")
    teacher, checkpoint = load_teacher(resolve_path(args.phase2_ckpt), device)
    data = load_storage_data(
        data_path=args.data_path,
        split_manifest_path=args.split_manifest,
        user_col=args.user_col,
        policy_name="old",
        seed=args.seed,
    )
    expected_hash = checkpoint.get("data_info", {}).get("data_sha256")
    if expected_hash and expected_hash != data["data_sha256"]:
        raise RuntimeError("当前数据文件与 Stage-2 checkpoint 使用的数据不一致")
    transfer = load_transfer_cache(resolve_path(args.transfer_cache), len(data["meta"]))
    table = build_pair_table(data, transfer, args.min_transfer_shift)
    table = limit_pairs(table, {
        "train": args.max_train_samples,
        "val": args.max_val_samples,
        "test": args.max_test_samples,
    }, args.seed)
    table = add_balance_weights(table)
    positive, pseudo, mismatch = build_teacher_inputs(data, transfer, table)
    train_mask = table["split_id"].to_numpy(np.int64) == SPLIT_TO_ID["train"]
    dp_clip = estimate_dp_clip(positive, train_mask)
    use_amp = not args.no_amp
    print(f"[StudentData] pairs={len(table)}, dp_clip_q99={dp_clip:.6f}")
    positive_targets = collect_teacher_targets(
        teacher, positive, device, args.batch_size, use_amp
    )
    pseudo_targets = collect_teacher_targets(
        teacher, pseudo, device, args.batch_size, use_amp
    )
    mismatch_targets = collect_teacher_targets(
        teacher, mismatch, device, args.batch_size, use_amp
    )
    dataset_path, summary = save_dataset(
        args,
        checkpoint,
        data,
        table,
        positive,
        pseudo,
        mismatch,
        positive_targets,
        pseudo_targets,
        mismatch_targets,
        dp_clip,
        time.perf_counter() - started,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[Done] {dataset_path}")


if __name__ == "__main__":
    main()
