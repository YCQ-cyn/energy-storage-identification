# -*- coding: utf-8 -*-
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


SPLIT_NAMES = ("train", "val", "test")


def normalize_user_ids(values: pd.Series) -> pd.Series:
    normalized = values.astype(str).str.strip()
    normalized = normalized.str.replace(r"\.0$", "", regex=True)
    return normalized


def load_filtered_rows(path: str, user_col: str) -> tuple[pd.DataFrame, str, str, str]:
    df = pd.read_excel(path)
    required = [
        user_col,
        "P_has_negative",
        "Sign_Classification",
        "Storage_to_Load_Ratio",
        "Spike_Abnormal",
        "month",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"输入文件缺少必要列: {missing}")

    industry_col = "所属行业" if "所属行业" in df.columns else ""
    date_col = "日期" if "日期" in df.columns else ""
    account_col = "发电户号" if "发电户号" in df.columns else ""

    mask = (
        (df["P_has_negative"] == 0)
        & (df["Sign_Classification"] == 2)
        & (df["Storage_to_Load_Ratio"] <= 0.8)
        & (df["Storage_to_Load_Ratio"] >= 0.01)
        & (df["Spike_Abnormal"] == 0)
    )
    keep = [user_col, "month"]
    keep.extend(column for column in [industry_col, date_col, account_col] if column)
    filtered = df.loc[mask, keep].copy()
    filtered[user_col] = normalize_user_ids(filtered[user_col])
    invalid = filtered[user_col].str.lower().isin(["", "nan", "none", "null"])
    if invalid.any():
        raise ValueError(f"用户列 {user_col} 存在空值或无效值")
    if date_col:
        filtered[date_col] = pd.to_datetime(filtered[date_col], errors="coerce")
    return filtered, industry_col, date_col, account_col


def build_user_profiles(
    filtered: pd.DataFrame,
    user_col: str,
    industry_col: str,
    date_col: str,
    account_col: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    users = np.array(sorted(filtered[user_col].unique()), dtype=str)
    profile = pd.DataFrame({"user_id": users})
    grouped = filtered.groupby(user_col, sort=False)

    profile["sample_count"] = profile["user_id"].map(grouped.size()).astype(int)
    if date_col:
        profile["unique_date_count"] = profile["user_id"].map(grouped[date_col].nunique()).astype(int)
        profile["date_min"] = profile["user_id"].map(grouped[date_col].min()).dt.strftime("%Y-%m-%d")
        profile["date_max"] = profile["user_id"].map(grouped[date_col].max()).dt.strftime("%Y-%m-%d")
    else:
        profile["unique_date_count"] = profile["sample_count"]
        profile["date_min"] = ""
        profile["date_max"] = ""

    profile["month_count"] = profile["user_id"].map(grouped["month"].nunique()).astype(int)
    if account_col:
        profile["generation_account_count"] = profile["user_id"].map(
            grouped[account_col].nunique(dropna=True)
        ).astype(int)
    else:
        profile["generation_account_count"] = 0

    if industry_col:
        industry = grouped[industry_col].agg(
            lambda values: values.mode().iloc[0] if not values.mode().empty else "UNKNOWN"
        )
        profile["industry"] = profile["user_id"].map(industry).fillna("UNKNOWN").astype(str)
    else:
        profile["industry"] = "UNKNOWN"

    user_index = {user_id: index for index, user_id in enumerate(users)}
    month_matrix = np.zeros((len(users), 12), dtype=np.float64)
    for (user_id, month), count in filtered.groupby([user_col, "month"]).size().items():
        month_int = int(month)
        if 1 <= month_int <= 12:
            month_matrix[user_index[str(user_id)], month_int - 1] = float(count)

    count_bins = pd.qcut(
        profile["sample_count"].rank(method="first"),
        q=min(4, len(profile)),
        labels=False,
        duplicates="drop",
    ).astype(int)
    count_bin_matrix = pd.get_dummies(count_bins).to_numpy(dtype=np.float64)

    industry_counts = profile["industry"].value_counts()
    balanced_industries = sorted(industry_counts[industry_counts >= 5].index)
    if balanced_industries:
        industry_matrix = np.stack(
            [(profile["industry"].values == name).astype(np.float64) for name in balanced_industries],
            axis=1,
        )
    else:
        industry_matrix = np.zeros((len(profile), 0), dtype=np.float64)
    return profile, month_matrix, count_bin_matrix, industry_matrix


def split_sizes(n_users: int, train_ratio: float, val_ratio: float) -> tuple[list[int], np.ndarray]:
    test_ratio = 1.0 - train_ratio - val_ratio
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("train/val/test 比例必须都大于 0")
    train_count = int(round(n_users * train_ratio))
    val_count = int(round(n_users * val_ratio))
    test_count = n_users - train_count - val_count
    if min(train_count, val_count, test_count) < 1:
        raise ValueError("用户数量不足以生成三个非空集合")
    counts = [train_count, val_count, test_count]
    return counts, np.asarray(counts, dtype=np.float64) / float(n_users)


def ratio_error(groups: list[np.ndarray], matrix: np.ndarray, target: np.ndarray) -> float:
    if matrix.shape[1] == 0:
        return 0.0
    totals = matrix.sum(axis=0)
    valid = totals > 0
    values = np.stack([matrix[group].sum(axis=0) for group in groups])[:, valid]
    ratios = values / totals[valid]
    return float(np.mean(np.abs(ratios - target[:, None])))


def search_balanced_split(
    profile: pd.DataFrame,
    month_matrix: np.ndarray,
    count_bin_matrix: np.ndarray,
    industry_matrix: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    trials: int,
    min_eval_days: int,
) -> tuple[np.ndarray, dict]:
    counts, target = split_sizes(len(profile), train_ratio, val_ratio)
    sample_counts = profile["sample_count"].to_numpy(dtype=np.float64)
    total_samples = float(sample_counts.sum())
    rng = np.random.default_rng(seed)
    best = None
    forced_train = np.where(sample_counts < float(min_eval_days))[0]
    remaining = np.where(sample_counts >= float(min_eval_days))[0]
    train_remaining_count = counts[0] - len(forced_train)
    if train_remaining_count < 0:
        raise ValueError("低样本用户数量超过训练集用户容量，请降低 min_eval_days")
    if train_remaining_count + counts[1] + counts[2] != len(remaining):
        raise AssertionError("用户切分数量计算错误")

    for _ in range(trials):
        permutation = rng.permutation(remaining)
        train_end = train_remaining_count
        val_end = train_remaining_count + counts[1]
        groups = [
            np.concatenate([forced_train, permutation[:train_end]]),
            permutation[train_end:val_end],
            permutation[val_end:],
        ]
        day_ratios = np.asarray([sample_counts[group].sum() for group in groups]) / total_samples
        day_error = float(np.mean(np.abs(day_ratios - target)))
        month_error = ratio_error(groups, month_matrix, target)
        count_bin_error = ratio_error(groups, count_bin_matrix, target)
        industry_error = ratio_error(groups, industry_matrix, target)
        score = 5.0 * day_error + 2.0 * month_error + 1.5 * count_bin_error + 0.5 * industry_error
        if best is None or score < best[0]:
            best = (score, groups, day_ratios, day_error, month_error, count_bin_error, industry_error)

    score, groups, day_ratios, day_error, month_error, count_bin_error, industry_error = best
    assignment = np.empty(len(profile), dtype=object)
    for split_name, group in zip(SPLIT_NAMES, groups):
        assignment[group] = split_name
    diagnostics = {
        "objective": float(score),
        "user_counts": {name: int(len(group)) for name, group in zip(SPLIT_NAMES, groups)},
        "day_counts": {
            name: int(sample_counts[group].sum()) for name, group in zip(SPLIT_NAMES, groups)
        },
        "day_ratios": {
            name: float(value) for name, value in zip(SPLIT_NAMES, day_ratios)
        },
        "mean_day_ratio_error": day_error,
        "mean_month_ratio_error": month_error,
        "mean_sample_bin_ratio_error": count_bin_error,
        "mean_industry_ratio_error": industry_error,
        "min_eval_days": int(min_eval_days),
        "forced_train_users": int(len(forced_train)),
    }
    return assignment, diagnostics


def build_summary(
    profile: pd.DataFrame,
    filtered: pd.DataFrame,
    user_col: str,
    date_col: str,
    diagnostics: dict,
    args: argparse.Namespace,
) -> dict:
    split_stats = {}
    for split_name in SPLIT_NAMES:
        split_profile = profile[profile["split"] == split_name]
        split_users = set(split_profile["user_id"])
        split_rows = filtered[filtered[user_col].isin(split_users)]
        split_stats[split_name] = {
            "users": int(len(split_profile)),
            "samples": int(len(split_rows)),
            "sample_count_min": int(split_profile["sample_count"].min()),
            "sample_count_median": float(split_profile["sample_count"].median()),
            "sample_count_max": int(split_profile["sample_count"].max()),
            "month_samples": {
                str(int(month)): int(count)
                for month, count in split_rows["month"].value_counts().sort_index().items()
            },
        }

    duplicate_user_dates = 0
    duplicate_rows = 0
    if date_col:
        duplicated = filtered.duplicated([user_col, date_col], keep=False)
        duplicate_rows = int(duplicated.sum())
        duplicate_user_dates = int(filtered.loc[duplicated, [user_col, date_col]].drop_duplicates().shape[0])

    return {
        "version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(Path(args.input).resolve()),
        "user_column": args.user_col,
        "seed": args.seed,
        "trials": args.trials,
        "min_eval_days": args.min_eval_days,
        "requested_ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": 1.0 - args.train_ratio - args.val_ratio,
        },
        "filter": {
            "P_has_negative": 0,
            "Sign_Classification": 2,
            "Storage_to_Load_Ratio": [0.01, 0.8],
            "Spike_Abnormal": 0,
        },
        "total_users": int(profile["user_id"].nunique()),
        "total_samples": int(len(filtered)),
        "duplicate_rows_by_user_date": duplicate_rows,
        "duplicate_user_dates": duplicate_user_dates,
        "diagnostics": diagnostics,
        "splits": split_stats,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成全项目统一的用户级 train/val/test 切分")
    parser.add_argument("--input", default="processed_result.xlsx")
    parser.add_argument("--user_col", default="关联用户号")
    parser.add_argument("--output", default="data_split/user_split.csv")
    parser.add_argument("--summary", default="data_split/user_split_summary.json")
    parser.add_argument("--train_ratio", type=float, default=0.75)
    parser.add_argument("--val_ratio", type=float, default=0.125)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--trials", type=int, default=60000)
    parser.add_argument("--min_eval_days", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    filtered, industry_col, date_col, account_col = load_filtered_rows(args.input, args.user_col)
    profile, month_matrix, count_bin_matrix, industry_matrix = build_user_profiles(
        filtered, args.user_col, industry_col, date_col, account_col
    )
    assignment, diagnostics = search_balanced_split(
        profile,
        month_matrix,
        count_bin_matrix,
        industry_matrix,
        args.train_ratio,
        args.val_ratio,
        args.seed,
        args.trials,
        args.min_eval_days,
    )
    profile["split"] = assignment
    profile["split_seed"] = args.seed
    profile["split_version"] = 1
    profile = profile.sort_values(["split", "user_id"]).reset_index(drop=True)

    if profile.groupby("user_id")["split"].nunique().max() != 1:
        raise AssertionError("同一用户被分配到了多个集合")
    if set(profile["split"].unique()) != set(SPLIT_NAMES):
        raise AssertionError("train/val/test 集合不完整")

    output_path = Path(args.output)
    summary_path = Path(args.summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(output_path, index=False, encoding="utf-8-sig")

    summary = build_summary(profile, filtered, args.user_col, date_col, diagnostics, args)
    summary["manifest_sha256"] = hashlib.sha256(output_path.read_bytes()).hexdigest()
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"切分清单: {output_path}")
    print(f"切分报告: {summary_path}")
    print(json.dumps(summary["splits"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
