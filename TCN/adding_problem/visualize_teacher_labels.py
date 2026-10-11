import argparse
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from TCN.adding_problem.teacher_targets import (
        BEHAVIOR_TARGET_NAMES,
        PHYSICAL_TARGET_NAMES,
        RESPONSE_TARGET_NAMES,
        TeacherTargetConfig,
        compute_stage2_targets,
    )
    from TCN.adding_problem.utils import load_storage_data
except ImportError:
    try:
        from .teacher_targets import (
            BEHAVIOR_TARGET_NAMES,
            PHYSICAL_TARGET_NAMES,
            RESPONSE_TARGET_NAMES,
            TeacherTargetConfig,
            compute_stage2_targets,
        )
        from .utils import load_storage_data
    except ImportError:
        from teacher_targets import (
            BEHAVIOR_TARGET_NAMES,
            PHYSICAL_TARGET_NAMES,
            RESPONSE_TARGET_NAMES,
            TeacherTargetConfig,
            compute_stage2_targets,
        )
        from utils import load_storage_data


PROJECT_DIR = Path(__file__).resolve().parent
TARGET_GROUPS = (
    ("physical", PHYSICAL_TARGET_NAMES, "physical_targets"),
    ("response", RESPONSE_TARGET_NAMES, "response_targets"),
    ("behavior", BEHAVIOR_TARGET_NAMES, "behavior_targets"),
)


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_DIR / path


def configure_plot():
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def contiguous_regions(values):
    values = np.asarray(values)
    starts = np.r_[0, np.flatnonzero(np.diff(values) != 0) + 1]
    ends = np.r_[starts[1:], len(values)]
    return [(int(start), int(end), float(values[start])) for start, end in zip(starts, ends)]


def shade_price_roles(axis, role_seq):
    colors = {-1.0: "#dbeafe", 0.0: "#f3f4f6", 1.0: "#fee2e2"}
    for start, end, role in contiguous_regions(role_seq):
        axis.axvspan(start / 4.0, end / 4.0, color=colors.get(role, "#f3f4f6"), alpha=0.45)


def parse_indices(value):
    if not value:
        return None
    return np.asarray([int(item.strip()) for item in value.split(",") if item.strip()], dtype=np.int64)


def select_samples(data, split, user_id, sample_indices, sample_count, config):
    meta = data["meta"]
    if sample_indices is not None:
        if np.any((sample_indices < 0) | (sample_indices >= len(meta))):
            raise ValueError("sample_indices 中存在越界索引")
        return sample_indices

    mask = np.ones(len(meta), dtype=bool)
    if split != "all":
        mask &= meta["split"].to_numpy() == split
    if user_id:
        mask &= meta["user_id"].astype(str).to_numpy() == str(user_id)
    candidates = np.flatnonzero(mask)
    if candidates.size == 0:
        raise ValueError("没有符合筛选条件的样本")
    if candidates.size <= sample_count:
        return candidates

    pool_size = min(candidates.size, max(2048, sample_count * 64))
    pool_positions = np.linspace(0, candidates.size - 1, pool_size, dtype=int)
    pool = candidates[pool_positions]
    role_seq = data["X"].index_select(
        0, torch.as_tensor(pool, dtype=torch.long)
    )[:, 3].numpy()
    targets = compute_stage2_targets(
        data["P_raw"][pool],
        data["Q_raw"][pool],
        role_seq,
        config=config,
    )
    storage_index = BEHAVIOR_TARGET_NAMES.index("storage_behavior")
    edge_index = PHYSICAL_TARGET_NAMES.index("edge_pair_match")
    storage_score = targets["behavior_targets"][:, storage_index]
    edge_score = targets["physical_targets"][:, edge_index]
    chosen = []
    for score in (storage_score, edge_score):
        order = pool[np.argsort(score)]
        positions = np.linspace(0, len(order) - 1, sample_count, dtype=int)
        chosen.extend(order[positions].tolist())
    chosen = list(dict.fromkeys(chosen))
    if len(chosen) < sample_count:
        for index in pool:
            if int(index) not in chosen:
                chosen.append(int(index))
            if len(chosen) >= sample_count:
                break
    return np.asarray(chosen[:sample_count], dtype=np.int64)


def target_values(targets, row_index):
    values = []
    for group, names, key in TARGET_GROUPS:
        for index, name in enumerate(names):
            values.append((group, name, float(targets[key][row_index, index])))
    return values


def plot_sample(output_path, sample_index, meta_row, p_raw, q_raw, role_seq, targets, pointwise):
    hours = np.arange(96) / 4.0
    q_aligned = pointwise["q_aligned_raw"]
    background = pointwise["background_raw"]
    background_smooth = pointwise["background_smooth_raw"]
    observed_response = pointwise["observed_response_raw"]
    response_center = pointwise["response_center_raw"]
    active_mask = pointwise["active_mask"]

    figure, axes = plt.subplots(
        4,
        1,
        figsize=(16, 15),
        gridspec_kw={"height_ratios": [1.25, 1.15, 0.9, 1.0]},
        constrained_layout=True,
    )
    for axis in axes[:3]:
        shade_price_roles(axis, role_seq)

    axes[0].plot(hours, p_raw, color="#111827", linewidth=1.8, label="原始 P")
    axes[0].plot(hours, background, color="#6b7280", linewidth=1.1, alpha=0.8, label="B=P-Q")
    axes[0].plot(hours, background_smooth, color="#16a34a", linewidth=1.4, label="平滑背景 B")
    axes[0].plot(hours, response_center, color="#2563eb", linewidth=1.2, linestyle="--", label="期望 P=B平滑+Q")
    axes[0].fill_between(
        hours,
        pointwise["response_band_lower_raw"],
        pointwise["response_band_upper_raw"],
        where=active_mask,
        interpolate=True,
        color="#ef4444",
        alpha=0.20,
        label="P 响应带",
    )
    axes[0].plot(
        hours,
        np.where(active_mask, pointwise["response_band_lower_raw"], np.nan),
        color="#dc2626",
        linewidth=0.8,
    )
    axes[0].plot(
        hours,
        np.where(active_mask, pointwise["response_band_upper_raw"], np.nan),
        color="#dc2626",
        linewidth=0.8,
    )
    axes[0].set_ylabel("功率原始值")
    axes[0].set_title("总负荷、去储能背景与 P 响应带")
    axes[0].legend(loc="upper right", ncol=3)

    axes[1].fill_between(
        hours,
        pointwise["response_delta_band_lower_raw"],
        pointwise["response_delta_band_upper_raw"],
        where=active_mask,
        interpolate=True,
        color="#ef4444",
        alpha=0.24,
        label="Q ± 局部噪声",
    )
    axes[1].plot(
        hours,
        np.where(active_mask, pointwise["response_delta_band_lower_raw"], np.nan),
        color="#dc2626",
        linewidth=0.8,
    )
    axes[1].plot(
        hours,
        np.where(active_mask, pointwise["response_delta_band_upper_raw"], np.nan),
        color="#dc2626",
        linewidth=0.8,
    )
    axes[1].plot(hours, q_aligned, color="#1d4ed8", linewidth=1.8, label="对齐后的 Q")
    axes[1].plot(hours, observed_response, color="#111827", linewidth=1.5, label="P-B平滑（实际响应）")
    axes[1].axhline(0.0, color="#374151", linewidth=0.8)
    axes[1].set_ylabel("储能响应原始值")
    axes[1].set_title("实际响应带：黑线落入红色带表示 Q 在 P 中可解释")
    axes[1].legend(loc="upper right", ncol=3)

    axes[2].plot(hours, pointwise["point_alignment"], linewidth=1.5, label="P/Q 点响应")
    axes[2].plot(hours, pointwise["point_run_score"], linewidth=1.2, label="Q 段长度结构")
    axes[2].plot(hours, pointwise["segment_edge_support"], linewidth=1.2, label="成对边界支持")
    axes[2].plot(hours, pointwise["point_behavior_evidence"], linewidth=1.5, label="政策行为证据")
    boundary_mask = pointwise["point_edge_score"] > 0.0
    axes[2].scatter(
        hours[boundary_mask],
        pointwise["point_edge_score"][boundary_mask],
        color="#dc2626",
        s=28,
        zorder=4,
        label="边界响应点",
    )
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_ylabel("约束得分")
    axes[2].set_xlabel("小时")
    axes[2].set_title("逐点约束证据")
    axes[2].legend(loc="upper right", ncol=3)

    values = target_values(targets, 0)
    names = [name for _, name, _ in values]
    scores = [score for _, _, score in values]
    colors = [
        "#2563eb" if group == "physical" else "#16a34a" if group == "response" else "#ea580c"
        for group, _, _ in values
    ]
    y_positions = np.arange(len(names))
    axes[3].barh(y_positions, scores, color=colors, alpha=0.85)
    axes[3].set_yticks(y_positions, labels=names)
    axes[3].set_xlim(0.0, 1.0)
    axes[3].invert_yaxis()
    axes[3].set_xlabel("教师约束目标")
    axes[3].grid(axis="x", alpha=0.25)
    for position, score in zip(y_positions, scores):
        axes[3].text(min(score + 0.015, 0.96), position, f"{score:.3f}", va="center", fontsize=9)

    title = (
        f"sample={sample_index} | user={meta_row['user_id']} | date={meta_row['date']} | "
        f"split={meta_row['split']} | month={meta_row['month']} | best_lag={int(targets['auxiliary']['best_lag'][0])}"
    )
    figure.suptitle(title, fontsize=14)
    figure.text(
        0.5,
        0.002,
        "蓝色背景=低价，灰色=平价，红色=高价；响应带是基于局部背景噪声的软诊断范围，不是硬阈值。",
        ha="center",
        fontsize=10,
    )
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def point_rows(sample_index, role_seq, p_raw, q_raw, pointwise):
    rows = []
    for point_index in range(96):
        rows.append({
            "sample_index": sample_index,
            "point": point_index,
            "time": f"{point_index // 4:02d}:{(point_index % 4) * 15:02d}",
            "price_role": float(role_seq[point_index]),
            "P_raw": float(p_raw[point_index]),
            "Q_raw": float(q_raw[point_index]),
            "Q_aligned_raw": float(pointwise["q_aligned_raw"][point_index]),
            "base_load_raw": float(pointwise["background_raw"][point_index]),
            "base_smooth_raw": float(pointwise["background_smooth_raw"][point_index]),
            "observed_response_raw": float(pointwise["observed_response_raw"][point_index]),
            "response_error_raw": float(pointwise["response_error_raw"][point_index]),
            "response_band_hit": bool(pointwise["response_band_hit"][point_index]),
            "response_delta_band_lower_raw": float(
                pointwise["response_delta_band_lower_raw"][point_index]
            ),
            "response_delta_band_upper_raw": float(
                pointwise["response_delta_band_upper_raw"][point_index]
            ),
            "expected_P_raw": float(pointwise["response_center_raw"][point_index]),
            "response_band_lower_raw": float(pointwise["response_band_lower_raw"][point_index]),
            "response_band_upper_raw": float(pointwise["response_band_upper_raw"][point_index]),
            "local_noise_raw": float(pointwise["local_noise_raw"][point_index]),
            "point_alignment": float(pointwise["point_alignment"][point_index]),
            "point_run_score": float(pointwise["point_run_score"][point_index]),
            "point_edge_score": float(pointwise["point_edge_score"][point_index]),
            "segment_edge_support": float(pointwise["segment_edge_support"][point_index]),
            "point_evidence": float(pointwise["point_evidence"][point_index]),
            "point_role_fit": float(pointwise["point_role_fit"][point_index]),
            "point_behavior_evidence": float(pointwise["point_behavior_evidence"][point_index]),
            "supported_charge_raw": float(pointwise["supported_charge_raw"][point_index]),
            "supported_discharge_raw": float(pointwise["supported_discharge_raw"][point_index]),
        })
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="可视化 Stage-2 教师约束和真实 P/Q 响应带")
    parser.add_argument("--data_path", type=str, default="processed_result.xlsx")
    parser.add_argument("--split_manifest", type=str, default="data_split/user_split.csv")
    parser.add_argument("--user_col", type=str, default="关联用户号")
    parser.add_argument("--policy_name", type=str, default="old", choices=["old", "new"])
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test", "all"])
    parser.add_argument("--user_id", type=str, default=None)
    parser.add_argument("--sample_indices", type=str, default=None)
    parser.add_argument("--sample_count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--time_type", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="teacher_constraint_review")
    parser.add_argument("--alignment_max_lag", type=int, default=2)
    parser.add_argument("--edge_window_points", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.sample_count <= 0:
        raise ValueError("sample_count 必须大于 0")
    configure_plot()
    output_dir = resolve_path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_storage_data(
        data_path=resolve_path(args.data_path),
        train_ratio=0.9,
        time_type=args.time_type,
        seed=args.seed,
        split_manifest_path=resolve_path(args.split_manifest),
        user_col=args.user_col,
        policy_name=args.policy_name,
    )
    config = TeacherTargetConfig(
        alignment_max_lag=args.alignment_max_lag,
        edge_window_points=args.edge_window_points,
    )
    selected = select_samples(
        data,
        args.split,
        args.user_id,
        parse_indices(args.sample_indices),
        args.sample_count,
        config,
    )

    summary_rows = []
    all_point_rows = []
    for sample_index in selected:
        role_seq = data["X"][sample_index, 3].numpy()
        sample_targets = compute_stage2_targets(
            data["P_raw"][sample_index:sample_index + 1],
            data["Q_raw"][sample_index:sample_index + 1],
            role_seq[None, :],
            config=config,
            include_pointwise=True,
        )
        pointwise = {key: values[0] for key, values in sample_targets["pointwise"].items()}
        meta_row = data["meta"].iloc[int(sample_index)]
        plot_sample(
            output_dir / f"sample_{int(sample_index):06d}.png",
            int(sample_index),
            meta_row,
            data["P_raw"][sample_index],
            data["Q_raw"][sample_index],
            role_seq,
            sample_targets,
            pointwise,
        )
        summary = {
            "sample_index": int(sample_index),
            "user_id": str(meta_row["user_id"]),
            "date": str(meta_row["date"]),
            "split": str(meta_row["split"]),
            "month": int(meta_row["month"]),
            "best_lag": int(sample_targets["auxiliary"]["best_lag"][0]),
        }
        for _, name, value in target_values(sample_targets, 0):
            summary[name] = value
        summary_rows.append(summary)
        all_point_rows.extend(
            point_rows(
                int(sample_index),
                role_seq,
                data["P_raw"][sample_index],
                data["Q_raw"][sample_index],
                pointwise,
            )
        )

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "sample_constraint_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(all_point_rows).to_csv(
        output_dir / "sample_point_details.csv", index=False, encoding="utf-8-sig"
    )
    print(f"[Done] 输出目录: {output_dir}")
    print(f"[Done] 样本索引: {selected.tolist()}")
    print("[Note] 响应带中心为 Q，对应到 P 上是平滑背景+Q；半宽来自局部背景噪声。")


if __name__ == "__main__":
    main()
