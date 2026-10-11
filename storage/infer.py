# -*- coding: utf-8 -*-
import argparse
import json
import math
import re
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from .distill import (
        STUDENT_CHANNEL_NAMES,
        STUDENT_DATASET_VERSION,
        STUDENT_TARGET_NAMES,
        build_student_channels,
    )
    from .student import (
        STUDENT_OBJECTIVE_VERSION,
        StudentModel,
        load_checkpoint,
    )
    from .model import MODEL_ARCH_VERSION
    from .data import build_price_features, scale_P
except ImportError:
    from distill import (
        STUDENT_CHANNEL_NAMES,
        STUDENT_DATASET_VERSION,
        STUDENT_TARGET_NAMES,
        build_student_channels,
    )
    from student import (
        STUDENT_OBJECTIVE_VERSION,
        StudentModel,
        load_checkpoint,
    )
    from model import MODEL_ARCH_VERSION
    from data import build_price_features, scale_P


TIME_COLUMNS = tuple(
    f"{minute // 60}:{minute % 60:02d}" for minute in range(0, 24 * 60, 15)
)
DATE_COLUMN_ALIASES = (
    "数据日期", "日期", "用电日期", "采集日期", "date", "data_date", "day",
)
USER_COLUMN_ALIASES = (
    "户号", "关联用户号", "用户号", "用户编号", "客户编号", "customer_id",
    "user_id", "user",
)
SUMMARY_COLUMNS = (
    "户号",
    "储能情况",
    "总天数",
    "有效天数",
    "强储能占比",
    "中储能占比",
    "无或储能占比不高占比",
)
DETAIL_SCORE_NAMES = {
    "storage_behavior": "储能评分",
    "storage_evidence": "储能证据评分",
    "response_confidence": "响应可信度评分",
    "activity_strength": "活跃强度评分",
    "charge_visible": "充电可见度评分",
    "discharge_visible": "放电可见度评分",
}


def resolve_path(path):
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).resolve().parent / candidate


def normalize_time_header(column):
    if isinstance(column, time):
        if column.second != 0:
            return None
        return f"{column.hour}:{column.minute:02d}"
    if isinstance(column, (int, float, np.integer, np.floating)):
        value = float(column)
        if 0.0 <= value < 1.0:
            minute_of_day = int(round(value * 24 * 60))
            if minute_of_day < 24 * 60 and minute_of_day % 15 == 0:
                return f"{minute_of_day // 60}:{minute_of_day % 60:02d}"
    text = str(column).strip().replace("：", ":")
    text = text.replace("时", ":").replace("点", ":").replace("分", "")
    parts = text.split(":")
    if len(parts) not in (2, 3) or not all(part.isdigit() for part in parts):
        return None
    hour, minute = int(parts[0]), int(parts[1])
    second = int(parts[2]) if len(parts) == 3 else 0
    if not 0 <= hour <= 23 or minute not in (0, 15, 30, 45) or second != 0:
        return None
    return f"{hour}:{minute:02d}"


def detect_power_columns(frame):
    column_map = {}
    for column in frame.columns:
        normalized = normalize_time_header(column)
        if normalized not in TIME_COLUMNS:
            continue
        if normalized in column_map:
            raise ValueError(f"输入文件存在重复时刻列：{normalized}")
        column_map[normalized] = column
    if len(column_map) == len(TIME_COLUMNS):
        return [column_map[column] for column in TIME_COLUMNS]
    p_columns = []
    normalized_names = {
        re.sub(r"\s+", "", str(column)).upper(): column for column in frame.columns
    }
    for index in range(1, 97):
        key = f"P{index}"
        if key not in normalized_names:
            break
        p_columns.append(normalized_names[key])
    if len(p_columns) == 96:
        return p_columns
    missing = [column for column in TIME_COLUMNS if column not in column_map]
    raise ValueError(
        "无法识别96点负荷列。建议使用从 0:00 到 23:45 的96个时刻列，"
        "也兼容 P1-P96；"
        f"当前缺少时刻列：{', '.join(missing[:8])}"
    )


def load_input_frame(path, max_samples):
    options = {"nrows": max_samples} if max_samples > 0 else {}
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        return pd.read_excel(path, **options)
    if suffix == ".csv":
        return pd.read_csv(path, **options)
    raise ValueError("输入文件仅支持 .xlsx 或 .csv；旧版 .xls 请先另存为 .xlsx")


def normalize_column_name(value):
    return re.sub(r"[\s_\-]+", "", str(value).strip()).lower()


def resolve_input_column(frame, requested, aliases, display_name):
    normalized = {
        normalize_column_name(column): column for column in frame.columns
    }
    candidates = [] if requested in (None, "", "auto") else [requested]
    candidates.extend(aliases)
    for candidate in candidates:
        key = normalize_column_name(candidate)
        if key in normalized:
            return normalized[key]
    raise ValueError(
        f"无法识别{display_name}列，支持的常见列名包括：{', '.join(aliases[:5])}"
    )


def parse_dates(values):
    text = values.astype("string").str.strip()
    normalized = (
        text.str.replace("年", "-", regex=False)
        .str.replace("月", "-", regex=False)
        .str.replace("日", "", regex=False)
        .str.replace(".", "-", regex=False)
    )
    try:
        dates = pd.to_datetime(normalized, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        dates = pd.to_datetime(normalized, errors="coerce")
    compact_mask = normalized.str.fullmatch(r"\d{8}", na=False) & dates.isna()
    if compact_mask.any():
        dates.loc[compact_mask] = pd.to_datetime(
            normalized.loc[compact_mask], format="%Y%m%d", errors="coerce"
        )
    numeric = pd.to_numeric(values, errors="coerce")
    serial_mask = numeric.between(1, 100000) & dates.isna()
    if serial_mask.any():
        dates.loc[serial_mask] = pd.to_datetime(
            numeric.loc[serial_mask], unit="D", origin="1899-12-30", errors="coerce"
        )
    return dates


def build_validity(frame, date_column, user_column, power_columns):
    users = frame[user_column].astype("string").str.strip()
    dates = parse_dates(frame[date_column])
    power = frame[power_columns].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=np.float32
    )
    valid_user = users.notna().to_numpy() & users.ne("").fillna(False).to_numpy()
    valid_date = dates.notna().to_numpy()
    valid_power = np.isfinite(power).all(axis=1)
    valid = valid_user & valid_date & valid_power
    reasons = []
    for index in range(len(frame)):
        row_reasons = []
        if not valid_user[index]:
            row_reasons.append("户号缺失")
        if not valid_date[index]:
            row_reasons.append("数据日期缺失或无法解析")
        if not valid_power[index]:
            row_reasons.append("96点用户有功功率存在缺失、非数值或无穷值")
        reasons.append("；".join(row_reasons))
    return users, dates, power, valid, np.asarray(reasons, dtype=object)


def build_teacher_style_inputs(power_raw, months, policy_name):
    rows = []
    for index in range(len(power_raw)):
        power = np.asarray(scale_P(power_raw[index]), dtype=np.float32)
        policy = build_price_features(int(months[index]), policy_name=policy_name)
        rows.append(np.stack((
            power,
            np.zeros_like(power),
            np.zeros_like(power),
            policy["price_role_seq"].astype(np.float32),
            policy["price_role_jump"].astype(np.float32),
        ), axis=0))
    return np.asarray(rows, dtype=np.float32)


def load_student(path, device):
    checkpoint = load_checkpoint(path, device)
    if checkpoint.get("student_objective_version") != STUDENT_OBJECTIVE_VERSION:
        raise RuntimeError("学生 checkpoint 训练目标版本不匹配")
    if checkpoint.get("student_dataset_version") != STUDENT_DATASET_VERSION:
        raise RuntimeError("学生 checkpoint 数据版本不匹配")
    if checkpoint.get("model_arch_version") != MODEL_ARCH_VERSION:
        raise RuntimeError("学生 checkpoint 网络结构版本不匹配")
    if tuple(checkpoint.get("channel_names", ())) != tuple(STUDENT_CHANNEL_NAMES):
        raise RuntimeError("学生 checkpoint 输入通道语义不匹配")
    if tuple(checkpoint.get("target_names", ())) != tuple(STUDENT_TARGET_NAMES):
        raise RuntimeError("学生 checkpoint 输出目标语义不匹配")
    model = StudentModel(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


@torch.inference_mode()
def predict(model, inputs, device, batch_size, use_amp):
    rows = []
    for start in range(0, len(inputs), batch_size):
        batch = torch.from_numpy(inputs[start:start + batch_size]).to(device)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            scores = model(batch)["scores"]
        rows.append(scores.float().cpu())
    if not rows:
        return np.empty((0, len(STUDENT_TARGET_NAMES)), dtype=np.float32)
    return torch.cat(rows).numpy()


def classify_scores(scores, strong_threshold, medium_threshold):
    scores = np.asarray(scores, dtype=np.float64)
    return np.where(
        scores > strong_threshold,
        "强储能",
        np.where(scores >= medium_threshold, "中储能", "无或储能占比不高"),
    )


def top_fraction_mean(scores, ratio):
    count = max(1, int(math.ceil(len(scores) * ratio)))
    return float(np.mean(np.partition(scores, len(scores) - count)[-count:]))


def aggregate_users(detail, user_column, strong_threshold, medium_threshold, top_ratio):
    rows = []
    user_rows = detail[detail[user_column].notna() & detail[user_column].ne("")]
    for user_id, group in user_rows.groupby(user_column, sort=False):
        valid_group = group[group["是否有效"] == "是"]
        scores = valid_group["储能评分"].to_numpy(np.float64)
        row = {
            "户号": user_id,
            "总天数": int(len(group)),
            "有效天数": int(len(valid_group)),
        }
        if len(scores) == 0:
            row.update({
                "储能情况": "无有效数据",
                "强储能占比": 0.0,
                "中储能占比": 0.0,
                "无或储能占比不高占比": 0.0,
            })
        else:
            day_levels = classify_scores(scores, strong_threshold, medium_threshold)
            user_score = top_fraction_mean(scores, top_ratio)
            row.update({
                "储能情况": classify_scores(
                    np.asarray([user_score]), strong_threshold, medium_threshold
                )[0],
                "强储能占比": float(np.mean(day_levels == "强储能")),
                "中储能占比": float(np.mean(day_levels == "中储能")),
                "无或储能占比不高占比": float(
                    np.mean(day_levels == "无或储能占比不高")
                ),
            })
        rows.append(row)
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def build_detail_frame(frame, users, dates, valid, reasons, prediction,
                       date_column, user_column, strong_threshold, medium_threshold):
    detail = frame.copy()
    detail[user_column] = users
    detail.insert(0, "跳过原因", reasons)
    detail.insert(0, "是否有效", np.where(valid, "是", "否"))
    detail.insert(0, "月份", dates.dt.month.astype("Int64"))
    detail_scores = np.full(
        (len(detail), len(STUDENT_TARGET_NAMES)), np.nan, dtype=np.float32
    )
    detail_scores[valid] = prediction * 100.0
    insert_at = 3
    for target_index, target_name in enumerate(STUDENT_TARGET_NAMES):
        column_name = DETAIL_SCORE_NAMES[target_name]
        detail.insert(insert_at, column_name, detail_scores[:, target_index])
        insert_at += 1
    day_level = np.full(len(detail), "未计算", dtype=object)
    day_level[valid] = classify_scores(
        detail_scores[valid, 0], strong_threshold, medium_threshold
    )
    detail.insert(4, "日储能情况", day_level)
    leading = [
        date_column,
        user_column,
        "月份",
        "是否有效",
        "跳过原因",
        "储能评分",
        "日储能情况",
        "储能证据评分",
        "响应可信度评分",
        "活跃强度评分",
        "充电可见度评分",
        "放电可见度评分",
    ]
    remaining = [column for column in detail.columns if column not in leading]
    return detail[leading + remaining]


def compact_summary(summary):
    return summary[["户号", "储能情况", "总天数", "有效天数"]].copy()


def write_results(output_path, summary, detail, compact_output=False):
    if compact_output:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            compact_summary(summary).to_excel(
                writer, sheet_name="排查结果", index=False
            )
            sheet = writer.book["排查结果"]
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            sheet.column_dimensions["A"].width = 24
            sheet.column_dimensions["B"].width = 24
            sheet.column_dimensions["C"].width = 12
            sheet.column_dimensions["D"].width = 12
        return
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="用户汇总", index=False)
        detail.to_excel(writer, sheet_name="日明细", index=False)
        for sheet_name in ("用户汇总", "日明细"):
            sheet = writer.book[sheet_name]
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
        summary_sheet = writer.book["用户汇总"]
        for column in range(5, 8):
            for row in range(2, summary_sheet.max_row + 1):
                summary_sheet.cell(row=row, column=column).number_format = "0.00%"
        detail_sheet = writer.book["日明细"]
        score_headers = set(DETAIL_SCORE_NAMES.values())
        for cell in detail_sheet[1]:
            if cell.value in score_headers:
                for row in range(2, detail_sheet.max_row + 1):
                    detail_sheet.cell(row=row, column=cell.column).number_format = "0.00"


def parse_args():
    parser = argparse.ArgumentParser(description="仅使用用户有功功率的学生模型储能证据推理")
    parser.add_argument("--ckpt", default="train_student/best_model_student.pt")
    parser.add_argument("--input_path", default="表格中的非储能.xlsx")
    parser.add_argument("--output_path", default="表格中的非储能 infer.xlsx")
    parser.add_argument("--policy_name", default="old", choices=["old", "new"])
    parser.add_argument("--user_col", default="户号")
    parser.add_argument("--date_col", default="数据日期")
    parser.add_argument("--strong_threshold", type=float, default=80.0, help="强储能分数")
    parser.add_argument("--medium_threshold", type=float, default=60.0, help="中储能分数")
    parser.add_argument("--user_top_ratio", type=float, default=0.20, help="前百分之多少的数据计算平均")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--compact_output", action="store_true")
    parser.add_argument("--preview_json", default="")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 <= args.medium_threshold < args.strong_threshold <= 100.0:
        raise ValueError("分档阈值必须满足 0 <= medium < strong <= 100")
    if not 0.0 < args.user_top_ratio <= 1.0:
        raise ValueError("user_top_ratio 必须位于 (0, 1]")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    checkpoint_path = resolve_path(args.ckpt)
    model, checkpoint = load_student(checkpoint_path, device)
    input_path = resolve_path(args.input_path)
    frame = load_input_frame(input_path, args.max_samples)
    date_column = resolve_input_column(
        frame, args.date_col, DATE_COLUMN_ALIASES, "数据日期"
    )
    user_column = resolve_input_column(
        frame, args.user_col, USER_COLUMN_ALIASES, "户号"
    )
    power_columns = detect_power_columns(frame)
    users, dates, power, valid, reasons = build_validity(
        frame, date_column, user_column, power_columns
    )
    if valid.any():
        valid_power = power[valid]
        months = dates[valid].dt.month.astype(np.int64).to_numpy()
        teacher_style = build_teacher_style_inputs(valid_power, months, args.policy_name)
        student_inputs = build_student_channels(
            teacher_style, float(checkpoint["dp_clip_q99"])
        )
        prediction = predict(
            model,
            student_inputs,
            device,
            args.batch_size,
            use_amp=not args.no_amp,
        )
    else:
        prediction = np.empty(
            (0, len(STUDENT_TARGET_NAMES)), dtype=np.float32
        )
    detail = build_detail_frame(
        frame,
        users,
        dates,
        valid,
        reasons,
        prediction,
        date_column,
        user_column,
        args.strong_threshold,
        args.medium_threshold,
    )
    summary = aggregate_users(
        detail,
        user_column,
        args.strong_threshold,
        args.medium_threshold,
        args.user_top_ratio,
    )
    output_path = resolve_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_results(output_path, summary, detail, compact_output=args.compact_output)
    if args.preview_json:
        preview_path = resolve_path(args.preview_json)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview_path.write_text(
            json.dumps(
                compact_summary(summary).to_dict("records"),
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    print(
        f"[Infer] total={len(frame)}, valid={int(valid.sum())}, "
        f"skipped={int((~valid).sum())}, users={len(summary)}, device={device}"
    )
    print(
        f"[Rule] strong>{args.strong_threshold:g}, "
        f"medium>={args.medium_threshold:g} and <= {args.strong_threshold:g}, "
        f"user_top_ratio={args.user_top_ratio:.0%}"
    )
    print(f"[Done] {output_path}")


if __name__ == "__main__":
    main()
