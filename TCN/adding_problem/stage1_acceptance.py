import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    from TCN.adding_problem.add_test_st1 import EncoderWithHead, _spectrum_stats, set_seed
    from TCN.adding_problem.teacher_targets import FixedQuantileScaler, compute_physical_components
    from TCN.adding_problem.utils import load_storage_data
except ModuleNotFoundError:
    from add_test_st1 import EncoderWithHead, _spectrum_stats, set_seed
    from teacher_targets import FixedQuantileScaler, compute_physical_components
    from utils import load_storage_data


PROBE_TARGETS = (
    "storage_to_base_ratio",
    "energy_balance",
    "charge_in_valley_ratio",
    "discharge_in_peak_ratio",
    "base_negative_ratio",
)


def build_model(model_args, device):
    return EncoderWithHead(
        input_channels=int(model_args.get("input_channels", 2)),
        nhid=int(model_args["nhid"]),
        levels=int(model_args["levels"]),
        ksize=int(model_args["ksize"]),
        dropout=float(model_args["dropout"]),
        lstm_hidden=int(model_args["lstm_hidden"]),
        lstm_layers=int(model_args["lstm_layers"]),
        embedding_dim=int(model_args["embedding_dim"]),
        proj_dim=int(model_args["proj_dim"]),
    ).to(device)


@torch.no_grad()
def encode(model, inputs, indices, device, batch_size):
    h_parts = []
    z_parts = []
    model.eval()
    for start in range(0, len(indices), batch_size):
        index_tensor = torch.from_numpy(indices[start:start + batch_size]).long()
        batch = inputs.index_select(0, index_tensor).to(device=device, dtype=torch.float32)
        h, z = model(batch)
        h_parts.append(h.cpu().numpy())
        z_parts.append(z.cpu().numpy())
    return np.concatenate(h_parts), np.concatenate(z_parts)


def fit_feature_scaler(train_features, sample_weight):
    return StandardScaler().fit(train_features, sample_weight=sample_weight)


def physical_probe(train_features, val_features, targets, train_indices, val_indices,
                   train_weight, val_weight):
    scaler = fit_feature_scaler(train_features, train_weight)
    train_x = scaler.transform(train_features)
    val_x = scaler.transform(val_features)
    results = {}
    for name in PROBE_TARGETS:
        train_y = targets[name][train_indices]
        val_y = targets[name][val_indices]
        model = Ridge(alpha=1.0).fit(train_x, train_y, sample_weight=train_weight)
        prediction = model.predict(val_x)
        results[name] = {
            "r2_user_balanced": float(r2_score(val_y, prediction, sample_weight=val_weight)),
            "mae_user_balanced": float(
                mean_absolute_error(val_y, prediction, sample_weight=val_weight)
            ),
        }
    return results


def curve_probe(train_features, val_features, train_curves, val_curves,
                train_weight, val_weight):
    scaler = fit_feature_scaler(train_features, train_weight)
    train_x = scaler.transform(train_features)
    val_x = scaler.transform(val_features)
    model = Ridge(alpha=1.0).fit(train_x, train_curves, sample_weight=train_weight)
    prediction = model.predict(val_x)
    return float(r2_score(
        val_curves,
        prediction,
        sample_weight=val_weight,
        multioutput="variance_weighted",
    ))


def mean_probe_r2(results):
    return float(np.mean([results[name]["r2_user_balanced"] for name in PROBE_TARGETS]))


def parse_args():
    parser = argparse.ArgumentParser(description="Stage-1 表征验收")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--min_h_erank", type=float, default=6.0)
    parser.add_argument("--min_z_erank", type=float, default=20.0)
    parser.add_argument("--max_val_d2_ratio", type=float, default=0.20)
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.ckpt).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else checkpoint_path.with_name(f"{checkpoint_path.stem}_acceptance.json")
    )
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_args = checkpoint["args"]

    trained_model = build_model(model_args, device)
    missing, unexpected = trained_model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    if missing or unexpected:
        raise RuntimeError(f"checkpoint 参数不兼容：missing={missing}, unexpected={unexpected}")
    set_seed(args.seed)
    random_model = build_model(model_args, device)

    data_info = checkpoint["data_info"]
    data = load_storage_data(
        data_path=data_info["data_path"],
        time_type=model_args.get("time_type"),
        seed=int(model_args["seed"]),
        split_manifest_path=data_info["split_manifest_path"],
        user_col=model_args["user_col"],
        policy_name=model_args.get("policy_name", "old"),
    )
    if data["split_strategy"] != "user-manifest":
        raise RuntimeError("Stage-1 验收未使用固定用户切分清单")
    train_indices = data["split_indices"]["train"]
    val_indices = data["split_indices"]["val"]
    inputs = data["X"][:, :2, :]

    trained_h_train, trained_z_train = encode(
        trained_model, inputs, train_indices, device, args.batch_size
    )
    trained_h_val, trained_z_val = encode(
        trained_model, inputs, val_indices, device, args.batch_size
    )
    random_h_train, random_z_train = encode(
        random_model, inputs, train_indices, device, args.batch_size
    )
    random_h_val, random_z_val = encode(
        random_model, inputs, val_indices, device, args.batch_size
    )

    meta = data["meta"]
    train_weight = meta.iloc[train_indices]["sample_weight"].to_numpy(dtype=np.float64)
    val_weight = meta.iloc[val_indices]["sample_weight"].to_numpy(dtype=np.float64)
    price_roles = data["X"][:, 3, :].numpy()
    raw_targets = compute_physical_components(data["P_raw"], data["Q_raw"], price_roles)
    target_scaler = FixedQuantileScaler.fit(raw_targets, train_indices)
    scaled_targets = target_scaler.transform(raw_targets)

    trained_probe = physical_probe(
        trained_h_train,
        trained_h_val,
        scaled_targets,
        train_indices,
        val_indices,
        train_weight,
        val_weight,
    )
    random_probe = physical_probe(
        random_h_train,
        random_h_val,
        scaled_targets,
        train_indices,
        val_indices,
        train_weight,
        val_weight,
    )

    train_curves = inputs.index_select(
        0, torch.from_numpy(train_indices)
    ).numpy().reshape(len(train_indices), -1)
    val_curves = inputs.index_select(
        0, torch.from_numpy(val_indices)
    ).numpy().reshape(len(val_indices), -1)
    trained_curve_r2 = curve_probe(
        trained_h_train, trained_h_val, train_curves, val_curves, train_weight, val_weight
    )
    random_curve_r2 = curve_probe(
        random_h_train, random_h_val, train_curves, val_curves, train_weight, val_weight
    )

    spectra = {
        "trained_h_train": _spectrum_stats(trained_h_train),
        "trained_h_val": _spectrum_stats(trained_h_val),
        "trained_z_train": _spectrum_stats(trained_z_train),
        "trained_z_val": _spectrum_stats(trained_z_val),
        "random_h_train": _spectrum_stats(random_h_train),
        "random_h_val": _spectrum_stats(random_h_val),
        "random_z_train": _spectrum_stats(random_z_train),
        "random_z_val": _spectrum_stats(random_z_val),
    }
    val_d2_ratio = float(checkpoint.get("val_metrics", {}).get("d2_ratio", float("inf")))
    finite_embeddings = all(np.isfinite(values).all() for values in (
        trained_h_train, trained_h_val, trained_z_train, trained_z_val
    ))
    core_checks = {
        "finite_embeddings": bool(finite_embeddings),
        "h_noncollapse": bool(spectra["trained_h_val"][3] >= args.min_h_erank),
        "z_noncollapse": bool(spectra["trained_z_val"][3] >= args.min_z_erank),
        "view_invariance": bool(np.isfinite(val_d2_ratio) and val_d2_ratio <= args.max_val_d2_ratio),
    }
    probe_delta = mean_probe_r2(trained_probe) - mean_probe_r2(random_probe)
    curve_delta = trained_curve_r2 - random_curve_r2
    payload = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "objective_version": checkpoint.get("objective_version", "unknown"),
        "spectra": spectra,
        "validation_d2_ratio": val_d2_ratio,
        "core_checks": core_checks,
        "diagnostics": {
            "physical_probe": {
                "trained_h": trained_probe,
                "random_h": random_probe,
                "mean_r2_delta": probe_delta,
            },
            "curve_probe": {
                "trained_h_r2": trained_curve_r2,
                "random_h_r2": random_curve_r2,
                "r2_delta": curve_delta,
            },
        },
        "accepted": all(core_checks.values()),
        "stage2_ablation_required": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"设备：{device}")
    print(
        f"H-val rank={spectra['trained_h_val']} | "
        f"Z-val rank={spectra['trained_z_val']} | val_d2_ratio={val_d2_ratio:.4f}"
    )
    print(f"物理探针平均 R2 差值（训练-随机，仅诊断）：{probe_delta:+.4f}")
    print(
        f"P/Q 曲线重建 R2（仅诊断）：训练={trained_curve_r2:+.4f} | "
        f"随机={random_curve_r2:+.4f}"
    )
    print(f"核心验收项：{core_checks}")
    print(f"Stage-1 核心验收：{'通过' if payload['accepted'] else '未通过'}")
    print("迁移收益仍需通过 Stage-2 随机初始化消融确认。")
    print(f"输出：{output_path}")


if __name__ == "__main__":
    main()
