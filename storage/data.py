import hashlib
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


DATA_PIPELINE_VERSION = "clean-pq-v3-shared-p-scale"
P_SCALE_QUANTILE = 95.0
Q_GLOBAL_QUANTILE = 95.0
P_CLIP_MAX = 2.0
Q_CLIP_ABS = 3.0

def normalize_user_ids(values):
    normalized = values.astype(str).str.strip()
    return normalized.str.replace(r"\.0$", "", regex=True)


def load_user_split_manifest(path):
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到用户切分清单: {manifest_path}")
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig", dtype={"user_id": str})
    required = {"user_id", "split"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"用户切分清单缺少列: {sorted(missing)}")
    manifest = manifest.copy()
    manifest["user_id"] = normalize_user_ids(manifest["user_id"])
    manifest["split"] = manifest["split"].astype(str).str.strip().str.lower()
    if manifest["user_id"].duplicated().any():
        duplicates = manifest.loc[manifest["user_id"].duplicated(), "user_id"].head().tolist()
        raise ValueError(f"用户切分清单存在重复用户: {duplicates}")
    allowed = {"train", "val", "test"}
    unknown = set(manifest["split"].astype(str)) - allowed
    if unknown:
        raise ValueError(f"用户切分清单存在未知 split: {sorted(unknown)}")
    if set(manifest["split"].astype(str)) != allowed:
        raise ValueError("用户切分清单必须同时包含 train、val、test")
    return manifest


def attach_user_split(df, split_manifest_path, user_col="关联用户号"):
    if user_col not in df.columns:
        raise ValueError(f"数据中缺少用户列: {user_col}")
    manifest = load_user_split_manifest(split_manifest_path)
    split_map = manifest.set_index("user_id")["split"]
    result = df.copy()
    result["_user_id"] = normalize_user_ids(result[user_col])
    result["_split"] = result["_user_id"].map(split_map)
    if result["_split"].isna().any():
        missing_users = result.loc[result["_split"].isna(), "_user_id"].drop_duplicates().head().tolist()
        raise ValueError(f"以下用户未出现在切分清单中: {missing_users}")
    return result

def _as_finite_float_array(x, name):
    values = np.asarray(x, dtype=np.float32)
    if values.size == 0:
        raise ValueError(f"{name} 不能为空")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} 包含 NaN 或无穷值")
    return values


def _robust_reference(x, q=P_SCALE_QUANTILE, eps=1e-6, absolute=False):
    values = np.abs(x) if absolute else x
    ref = float(np.percentile(values, q))
    if ref < eps:
        ref = float(np.max(values))
    return ref if ref >= eps else 1.0


def scale_P(x, q=P_SCALE_QUANTILE, eps=1e-6, clip_max=P_CLIP_MAX, return_ref=False):
    values = _as_finite_float_array(x, "P")
    ref = _robust_reference(values, q=q, eps=eps)
    scaled = np.clip(values / ref, 0.0, clip_max).astype(np.float32)
    if return_ref:
        return scaled, ref
    return scaled


def scale_Q(x, q=Q_GLOBAL_QUANTILE, eps=1e-6, p_ref=None,
            q_global_scale=None, clip_abs=None):
    values = _as_finite_float_array(x, "Q")
    if np.allclose(values, 0.0):
        return np.zeros_like(values, dtype=np.float32)

    if p_ref is None:
        # 兼容旧调用；主数据链路统一使用 scale_pq。
        ref = _robust_reference(values, q=q, eps=eps, absolute=True)
        clip_abs = 2.0 if clip_abs is None else float(clip_abs)
    else:
        q_global_scale = 1.0 if q_global_scale is None else float(q_global_scale)
        ref = float(p_ref) * q_global_scale
        clip_abs = Q_CLIP_ABS if clip_abs is None else float(clip_abs)
        if ref < eps:
            raise ValueError("用户有功功率参考值与储能充放电功率全局尺度的乘积必须为正数")

    return np.clip(values / ref, -clip_abs, clip_abs).astype(np.float32)


def scale_pq(P, Q, q_global_scale, p_quantile=P_SCALE_QUANTILE,
             p_clip_max=P_CLIP_MAX, q_clip_abs=Q_CLIP_ABS, eps=1e-6):
    P_scaled, p_ref = scale_P(
        P, q=p_quantile, eps=eps, clip_max=p_clip_max, return_ref=True
    )
    Q_scaled = scale_Q(
        Q,
        eps=eps,
        p_ref=p_ref,
        q_global_scale=q_global_scale,
        clip_abs=q_clip_abs,
    )
    return P_scaled, Q_scaled, p_ref


def fit_q_global_scale(P, Q, indices=None, p_quantile=P_SCALE_QUANTILE,
                       q_quantile=Q_GLOBAL_QUANTILE, eps=1e-6):
    P_values = _as_finite_float_array(P, "P")
    Q_values = _as_finite_float_array(Q, "Q")
    if P_values.shape != Q_values.shape or P_values.ndim != 2:
        raise ValueError("用户有功功率、储能充放电功率必须是形状一致的二维曲线矩阵")

    if indices is not None:
        selected = np.asarray(indices, dtype=np.int64)
        if selected.size == 0:
            raise ValueError("训练集为空，无法拟合储能充放电功率全局尺度")
        P_values = P_values[selected]
        Q_values = Q_values[selected]

    p_refs = np.percentile(P_values, p_quantile, axis=1)
    p_max = np.max(P_values, axis=1)
    p_refs = np.where(p_refs >= eps, p_refs, p_max)
    p_refs = np.where(p_refs >= eps, p_refs, 1.0).astype(np.float32)
    q_relative = np.abs(Q_values) / p_refs[:, None]
    q_global_scale = float(np.percentile(q_relative, q_quantile))
    if q_global_scale < eps:
        q_global_scale = float(np.max(q_relative))
    return q_global_scale if q_global_scale >= eps else 1.0

# 平滑
def smooth(x, window_size=4):
    if window_size <= 1:
        return x
    cumsum = np.cumsum(np.insert(x, 0, 0))
    smoothed = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    pad_left = np.ones(window_size // 2) * smoothed[0]
    pad_right = np.ones(window_size - window_size // 2) * smoothed[-1]
    smoothed_full = np.concatenate([pad_left, smoothed, pad_right])
    if len(smoothed_full) > len(x):
        smoothed_full = smoothed_full[:len(x)]
    elif len(smoothed_full) < len(x):
        smoothed_full = np.pad(smoothed_full, (0, len(x) - len(smoothed_full)), mode='edge')
    return smoothed_full


# 噪声
def add_noise(x, std=0.01):
    noise = np.random.normal(0, std, x.shape)
    return x + noise


def augment_sync_intensity(P, Q, scale_min=0.9, scale_max=1.1, smooth_prob=0.7,
                           smooth_window=4, noise_std=0.01, rng=None):
    P = np.asarray(P, dtype=np.float32).copy()
    Q = np.asarray(Q, dtype=np.float32).copy()
    rng = np.random if rng is None else rng

    if scale_max > scale_min:
        scale = rng.uniform(scale_min, scale_max)
    else:
        scale = 1.0
    P = P * scale
    Q = Q * scale

    if smooth_prob > 0 and rng.random() < smooth_prob and smooth_window > 1:
        P = smooth(P, window_size=smooth_window)
        Q = smooth(Q, window_size=smooth_window)

    if noise_std > 0:
        P = P + rng.normal(0.0, noise_std, P.shape)
        Q = Q + rng.normal(0.0, noise_std, Q.shape)

    return P.astype(np.float32), Q.astype(np.float32)


def augment_sync_intensity_torch(inputs, scale_min=0.9, scale_max=1.1,
                                 smooth_prob=0.7, smooth_window=4,
                                 noise_std=0.01):
    if inputs.ndim != 3 or inputs.size(1) < 2:
        raise ValueError("批量增强输入应为 B×C×T，且至少包含 用户有功功率与储能充放电功率 两个通道")

    output = inputs.clone()
    pq = output[:, :2, :]
    batch_size = pq.size(0)
    if scale_max > scale_min:
        scale = torch.empty(
            batch_size, 1, 1, device=pq.device, dtype=pq.dtype
        ).uniform_(float(scale_min), float(scale_max))
        pq = pq * scale

    window = int(smooth_window)
    if smooth_prob > 0.0 and window > 1 and pq.size(-1) >= window:
        moving = F.avg_pool1d(pq, kernel_size=window, stride=1)
        left_size = window // 2
        right_size = pq.size(-1) - left_size - moving.size(-1)
        pieces = []
        if left_size > 0:
            pieces.append(moving[:, :, :1].expand(-1, -1, left_size))
        pieces.append(moving)
        if right_size > 0:
            pieces.append(moving[:, :, -1:].expand(-1, -1, right_size))
        smoothed = torch.cat(pieces, dim=-1)
        smooth_mask = torch.rand(
            batch_size, 1, 1, device=pq.device
        ) < float(smooth_prob)
        pq = torch.where(smooth_mask, smoothed, pq)

    if noise_std > 0.0:
        pq = pq + torch.randn_like(pq) * float(noise_std)

    output[:, :2, :] = pq
    return output


# 固定电价的 96 点掩码
def _idx(h, m=0):
    return h * 4 + (m // 15)


POLICY_REGISTRY = {
    # 当前训练的旧政策
    "old": {
        "season_meta": {
            "summer": {
                "months": [6, 7, 8],
                "gui_name": "夏季",
                "background_code": 1.0,  # background_code 模型粗负荷背景编码，不是政策季节本身
            },
            "other": {
                "months": [1, 2, 3, 4, 5, 9, 10, 11, 12],
                "gui_name": "非夏季",
                "background_code": 0.0,
            },
        },
        "segments": {
            "summer": [
                {"role": "valley", "start": (0, 0), "end": (8, 0)},
                {"role": "peak", "start": (8, 0), "end": (11, 0)},
                {"role": "flat", "start": (11, 0), "end": (18, 0)},
                {"role": "peak", "start": (18, 0), "end": (22, 0)},
                {"role": "flat", "start": (22, 0), "end": (24, 0)},
            ],
            "other": [
                {"role": "valley", "start": (0, 0), "end": (8, 0)},
                {"role": "peak", "start": (8, 0), "end": (11, 0)},
                {"role": "flat", "start": (11, 0), "end": (17, 0)},
                {"role": "peak", "start": (17, 0), "end": (22, 0)},
                {"role": "flat", "start": (22, 0), "end": (24, 0)},
            ],
        }
    },

    "new": {
        "season_meta": {
            "summer_winter": {
                "months": [1, 2, 6, 7, 8, 12],
                "gui_name": "夏冬季",
                "background_code": 0.0,
            },
            "spring_autumn": {
                "months": [3, 4, 5, 9, 10, 11],
                "gui_name": "春秋季",
                "background_code": 0.0,
            },
        },
        "segments": {
            "summer_winter": [
                {"role": "valley", "start": (0, 0), "end": (6, 0)},
                {"role": "flat", "start": (6, 0), "end": (11, 0)},
                {"role": "valley", "start": (11, 0), "end": (13, 0)},
                {"role": "flat", "start": (13, 0), "end": (14, 0)},
                {"role": "peak", "start": (14, 0), "end": (22, 0)},
                {"role": "flat", "start": (22, 0), "end": (24, 0)},
            ],
            "spring_autumn": [
                {"role": "flat", "start": (22, 0), "end": (2, 0)},
                {"role": "valley", "start": (2, 0), "end": (6, 0)},
                {"role": "flat", "start": (6, 0), "end": (10, 0)},
                {"role": "valley", "start": (10, 0), "end": (14, 0)},
                {"role": "flat", "start": (14, 0), "end": (15, 0)},
                {"role": "peak", "start": (15, 0), "end": (22, 0)},
            ],
        },
        "major_holiday_overrides": [
            {"role": "deep_valley", "start": (10, 0), "end": (14, 0)},
        ],
        "critical_peak_overrides": {
            "7": [
                {"role": "critical_peak", "start": (14, 0), "end": (15, 0)},
                {"role": "critical_peak", "start": (19, 30), "end": (21, 30)},
            ],
            "8": [
                {"role": "critical_peak", "start": (14, 0), "end": (15, 0)},
                {"role": "critical_peak", "start": (19, 30), "end": (21, 30)},
            ],
            "12": [
                {"role": "critical_peak", "start": (18, 0), "end": (20, 0)},
            ],
            "1": [
                {"role": "critical_peak", "start": (18, 0), "end": (20, 0)},
            ],
        },
    }
}


def resolve_policy_season(policy_name, month):
    month = int(month)
    if policy_name not in POLICY_REGISTRY:
        raise ValueError(f"未知 policy_name: {policy_name}")

    season_meta = POLICY_REGISTRY[policy_name]["season_meta"]

    for season_name, meta in season_meta.items():
        if month in meta["months"]:
            return season_name

    raise ValueError(f"month={month} 在 policy={policy_name} 中未找到对应的season")


def get_background_season_flag(policy_name, season_name, month=None):
    if policy_name not in POLICY_REGISTRY:
        raise ValueError(f"未知 policy_name: {policy_name}")

    season_meta = POLICY_REGISTRY[policy_name]["season_meta"]
    if season_name not in season_meta:
        raise ValueError(f"policy={policy_name} 下不存在 season={season_name}")

    if month is not None:
        background_code = 1.0 if int(month) in [6, 7, 8] else 0.0
    else:
        background_code = float(season_meta[season_name]["background_code"])
    return np.ones(96, dtype=np.float32) * background_code


def get_policy_cfg(policy_name, month=None, season_name=None, day_type='workday', user_type='normal'):
    if policy_name not in POLICY_REGISTRY:
        raise ValueError(f"未知 policy_name: {policy_name}")

    registry = POLICY_REGISTRY[policy_name]

    if season_name is None:
        if month is None:
            raise ValueError("get_policy_cfg 需要 month 或 season_name 至少提供一个")
        season_name = resolve_policy_season(policy_name, month)

    if season_name not in registry["season_meta"]:
        raise ValueError(f"policy={policy_name} 下不存在 season={season_name}")

    if season_name not in registry["segments"]:
        raise ValueError(f"policy={policy_name} 下不存在 season={season_name} 的时段配置")

    segments = [dict(segment) for segment in registry["segments"][season_name]]
    if len(segments) == 0:
        raise ValueError(f"policy={policy_name}, season={season_name} 的 segments 为空")

    large_industrial = user_type in {"large_industrial", "industrial_315_plus"}
    if policy_name == "new" and large_industrial and month is not None:
        segments.extend(
            dict(segment)
            for segment in registry.get("critical_peak_overrides", {}).get(str(int(month)), [])
        )
    if policy_name == "new" and large_industrial and day_type in {"major_holiday", "holiday"}:
        segments.extend(dict(segment) for segment in registry.get("major_holiday_overrides", []))

    return {
        "policy_name": policy_name,
        "month": None if month is None else int(month),
        "season_name": season_name,
        "season_meta": registry["season_meta"][season_name],
        "day_type": day_type,
        "user_type": user_type,
        "segments": segments
    }


def build_price_role_seq(policy_cfg):
    role_map = {
        "deep_valley": -1.0,
        "valley": -1.0,
        "flat": 0.0,
        "peak": 1.0,
        "critical_peak": 1.0,
    }

    seq = np.zeros(96, dtype=np.float32)
    for seg in policy_cfg["segments"]:
        role = seg["role"]
        s = _idx(*seg["start"])
        e = _idx(*seg["end"])
        if s < e:
            seq[s:e] = role_map[role]
        elif s > e:
            seq[s:] = role_map[role]
            seq[:e] = role_map[role]
    return seq


def build_price_weight_seq(policy_cfg):
    weight_map = {
        "deep_valley": 2.0,
        "valley": 1.5,
        "flat": 0.5,
        "peak": 1.5,
        "critical_peak": 2.0,
    }

    seq = np.ones(96, dtype=np.float32) * 0.2
    for seg in policy_cfg["segments"]:
        role = seg["role"]
        s = _idx(*seg["start"])
        e = _idx(*seg["end"])
        if s < e:
            seq[s:e] = weight_map[role]
        elif s > e:
            seq[s:] = weight_map[role]
            seq[:e] = weight_map[role]
    return seq


def build_price_jump_seq(price_role_seq):
    jump = np.diff(price_role_seq, prepend=price_role_seq[0]).astype(np.float32)
    jump[0] = 0.0
    return jump


def build_price_features(month=None, season_name=None, policy_name='old', day_type='workday', user_type='normal'):
    policy_cfg = get_policy_cfg(
        policy_name=policy_name,
        month=month,
        season_name=season_name,
        day_type=day_type,
        user_type=user_type
    )

    season_flag = get_background_season_flag(policy_name, policy_cfg["season_name"], month=month)
    price_role_seq = build_price_role_seq(policy_cfg)
    price_weight_seq = build_price_weight_seq(policy_cfg)
    price_role_jump = build_price_jump_seq(price_role_seq)

    return {
        "season_flag": season_flag.astype(np.float32),
        "price_role_seq": price_role_seq.astype(np.float32),
        "price_role_jump": price_role_jump.astype(np.float32),
        "price_weight_seq": price_weight_seq.astype(np.float32),
        "policy_cfg": policy_cfg,
    }


# 计算负荷的跳变
def compute_dP_channels(P):
    dP = np.diff(P, prepend=P[0])[:96]
    dP_pos = np.clip(dP, 0, None)
    dP_neg = np.clip(-dP, 0, None)
    dP_pos = dP_pos / (np.max(np.abs(dP_pos)) + 1e-6)
    dP_neg = dP_neg / (np.max(np.abs(dP_neg)) + 1e-6)
    return dP_pos, dP_neg


# 计算储能的跳变
def compute_dQ_channels(Q):
    dQ = np.diff(Q, prepend=Q[0])[:96]
    dQ_pos = np.clip(dQ, 0, None)
    dQ_neg = np.clip(-dQ, 0, None)
    dQ_pos = dQ_pos / (np.max(np.abs(dQ_pos)) + 1e-6)
    dQ_neg = dQ_neg / (np.max(np.abs(dQ_neg)) + 1e-6)
    return dQ_pos, dQ_neg


# ========= 伪负样本生成工具 =========
def _clip96(x):
    x = np.asarray(x, dtype=np.float32)
    if x.shape[-1] > 96:
        return x[:96]
    if x.shape[-1] < 96:
        return np.pad(x, (0, 96 - x.shape[-1]), mode='edge')
    return x


def _segment_indices(n_seg=5):
    # 将 96 点粗分成 n_seg 段
    cuts = sorted(set([0, 96] + [random.randint(8, 88) for _ in range(max(0, n_seg - 1))]))
    return cuts


def tf_q_zero_like(Q, price_role_seq, amp_low=0.5, amp_high=0.5, amp_mid=0.3):
    Q = np.asarray(Q, dtype=np.float32).copy()
    raw_std = float(np.std(Q))
    if raw_std < 1e-4:
        return Q

    q_std = float(max(raw_std, 1e-3))

    mask_low = (price_role_seq == -1)
    mask_mid = (price_role_seq == 0)
    mask_high = (price_role_seq == 1)

    if mask_low.any():
        Q[mask_low] = np.random.normal(loc=0.0, scale=amp_low * q_std * 0.2, size=mask_low.sum())
    if mask_high.any():
        Q[mask_high] = np.random.normal(loc=0.0, scale=amp_high * q_std * 0.2, size=mask_high.sum())
    if mask_mid.any():
        Q[mask_mid] = np.random.normal(loc=0.0, scale=amp_mid * q_std * 0.2, size=mask_mid.sum())

    return Q


def tf_q_reverse_price(Q, price_role_seq, amp_low=0.4, amp_high=0.4):
    Q = np.asarray(Q, dtype=np.float32).copy()
    raw_std = float(np.std(Q))
    if raw_std < 1e-4:
        return Q

    q_std = float(max(raw_std, 1e-3))

    mask_low = (price_role_seq == -1)
    mask_high = (price_role_seq == 1)

    base_low = -amp_low * q_std
    base_high = +amp_high * q_std
    noise_scale = 0.3 * q_std

    if mask_low.any():
        Q[mask_low] = base_low + np.random.normal(loc=0.0, scale=noise_scale, size=mask_low.sum())
    if mask_high.any():
        Q[mask_high] = base_high + np.random.normal(loc=0.0, scale=noise_scale, size=mask_high.sum())

    return Q


def tf_smooth_noise(Q, w=4, std=0.03):
    return add_noise(smooth(Q, window_size=w), std=std)


def tf_time_warp(Q, max_warp=0.12):
    Q = np.asarray(Q, dtype=np.float32)
    cuts = _segment_indices(n_seg=random.choice([4, 5, 6]))
    pieces = [Q[cuts[i]:cuts[i+1]] for i in range(len(cuts)-1)]
    new = []
    for p in pieces:
        if len(p) < 4:
            new.append(p)
            continue
        r = 1.0 + random.uniform(-max_warp, max_warp)
        L = max(2, int(round(len(p)*r)))
        tmp = np.interp(np.linspace(0, len(p)-1, L), np.arange(len(p)), p)
        new.append(tmp)
    Q2 = np.concatenate(new, axis=0)
    return _clip96(Q2)


def tf_shuffle_blocks(Q, n_seg=5):
    Q = np.asarray(Q, dtype=np.float32)
    cuts = _segment_indices(n_seg=n_seg)
    pieces = [Q[cuts[i]:cuts[i+1]].copy() for i in range(len(cuts)-1)]
    random.shuffle(pieces)
    return _clip96(np.concatenate(pieces, axis=0))


def tf_q_sign_flip_chunks(Q, n_chunks=1, chunk_len=8):
    Q = np.asarray(Q, dtype=np.float32).copy()
    for _ in range(n_chunks):
        a = random.randint(0, max(0, 96 - chunk_len))
        Q[a:a+chunk_len] *= -1.0
    return Q


def make_pseudo_negative_sample_typeA(x, pool_same_group=None):
    P, Q, season_flag, price_role_seq, price_role_jump = [x[i].copy() for i in range(5)]
    mode = random.choice(["zero", "interval", "reverse"])

    if mode == "zero":
        Q_neg = tf_q_zero_like(Q, price_role_seq, amp_low=0.3, amp_high=0.3, amp_mid=0.2)
        Q_neg = tf_smooth_noise(Q_neg, w=3, std=0.01)

    elif mode == "reverse":
        Q_neg = tf_q_reverse_price(Q, price_role_seq, amp_low=0.5, amp_high=0.5)
        Q_neg = tf_smooth_noise(Q_neg, w=3, std=0.01)

    else:
        Q_neg = np.asarray(Q, dtype=np.float32).copy()
        interval_ops = [
            lambda q: tf_time_warp(q, max_warp=0.12),
            lambda q: tf_shuffle_blocks(q, n_seg=random.choice([4, 5, 6])),
            lambda q: tf_q_sign_flip_chunks(q, n_chunks=1, chunk_len=random.choice([6, 8, 10])),
        ]
        random.shuffle(interval_ops)
        n_ops = random.choice([1, 2])
        for fn in interval_ops[:n_ops]:
            Q_neg = fn(Q_neg)

    Q_neg = _clip96(Q_neg).astype(np.float32)
    P_neg = P.copy()

    x_neg = np.stack(
        [P_neg, Q_neg, season_flag, price_role_seq, price_role_jump], axis=0
    ).astype(np.float32)
    return x_neg


def make_pseudo_negative_sample_typeB(x, pool_same_group=None, P_raw=None, Q_raw=None):
    P, Q, season_flag, price_role_seq, price_role_jump = [x[i].copy() for i in range(5)]
    if P_raw is None or Q_raw is None:
        raise ValueError("B 类伪负样本必须基于原始用户有功功率、储能充放电功率构造")

    P_neg = scale_P(np.asarray(P_raw, dtype=np.float32) - np.asarray(Q_raw, dtype=np.float32))
    P_neg = smooth(P_neg, window_size=4)
    P_neg = add_noise(P_neg, std=0.01)
    P_neg = np.clip(P_neg, 0.0, P_CLIP_MAX).astype(np.float32)

    Q_neg = np.zeros_like(Q, dtype=np.float32)

    x_neg = np.stack(
        [P_neg, Q_neg, season_flag, price_role_seq, price_role_jump], axis=0
    ).astype(np.float32)
    return x_neg


def build_pseudo_pool(seed=1111, time_type=None, policy_name="old",
                      data_path="processed_result.xlsx",
                      split_manifest_path="data_split/user_split.csv",
                      user_col="关联用户号", train_ratio=0.9):
    np.random.seed(seed)
    random.seed(seed)

    source_path = _resolve_input_path(data_path)
    df = pd.read_excel(source_path).copy()
    df["_source_row"] = df.index.astype(np.int64)
    if "P-Q_has_negative" not in df.columns:
        raise ValueError("输入数据缺少 P-Q_has_negative 列")

    df = filter_storage_rows(df, time_type=time_type)

    if split_manifest_path:
        manifest_path = _resolve_input_path(split_manifest_path)
        df = attach_user_split(df, manifest_path, user_col=user_col)
        df = df[df["_split"] == "train"].copy()
    else:
        df["_user_id"] = normalize_user_ids(df[user_col])
        split_meta = pd.DataFrame({"user_id": df["_user_id"].to_numpy()})
        split_values = _assign_user_level_split(
            split_meta, train_ratio, np.random.RandomState(seed)
        )
        df = df[split_values == "train"].copy()

    p_columns = [f"P{i}" for i in range(1, 97)]
    q_columns = [f"Q{i}" for i in range(1, 97)]
    P_train = _to_curve_matrix(df, p_columns, "P")
    Q_train = _to_curve_matrix(df, q_columns, "Q")
    q_global_scale = fit_q_global_scale(P_train, Q_train)

    cond_pseudo = (
        df["Storage_to_Load_Ratio"].between(0.1, 0.6)
        & (df["P-Q_has_negative"] == 0)
    )
    df = df.loc[cond_pseudo].copy()
    if len(df) == 0:
        raise ValueError("伪负样本池构造失败：满足条件的基底样本为 0 条。")

    P_data = _to_curve_matrix(df, p_columns, "P")
    Q_data = _to_curve_matrix(df, q_columns, "Q")

    X_base = []

    for i in range(len(P_data)):
        P, Q, _ = scale_pq(P_data[i], Q_data[i], q_global_scale=q_global_scale)

        if "month" not in df.columns:
            raise ValueError("processed_result.xlsx 中缺少 month 列，请补充 month 后再训练")

        month = int(df.iloc[i]["month"])
        price_feat = build_price_features(month=month, policy_name=policy_name)

        season_flag = price_feat["season_flag"]
        price_role_seq = price_feat["price_role_seq"]
        price_role_jump = price_feat["price_role_jump"]

        x = np.stack([P, Q, season_flag, price_role_seq, price_role_jump], axis=0)

        x[0] = add_noise(smooth(x[0]))
        x[1] = add_noise(smooth(x[1]))

        X_base.append(x)

    pool_all = X_base if len(X_base) >= 2 else None

    X_neg = []
    pseudo_types = []

    for index, x_i in enumerate(X_base):
        x_neg_A = make_pseudo_negative_sample_typeA(x_i, pool_same_group=pool_all)
        X_neg.append(x_neg_A)
        pseudo_types.append(0)

        x_neg_B = make_pseudo_negative_sample_typeB(
            x_i,
            pool_same_group=pool_all,
            P_raw=P_data[index],
            Q_raw=Q_data[index],
        )
        X_neg.append(x_neg_B)
        pseudo_types.append(1)

    if len(X_neg) == 0:
        raise ValueError("伪负样本池构造失败：X_neg 为空。")

    pseudo_pool = torch.tensor(np.array(X_neg, dtype=np.float32), dtype=torch.float32)
    print(f"[PseudoPool] base={len(X_base)} -> pseudo={len(X_neg)} (每条基底固定2条)")
    print(f"[PseudoPool] 伪负池形状: {pseudo_pool.shape}")

    pseudo_types = np.array(pseudo_types, dtype=np.int64)
    np.save("pseudo_pool_types.npy", pseudo_types)
    n_A = int((pseudo_types == 0).sum())
    n_B = int((pseudo_types == 1).sum())
    print(f"[PseudoPool] A 类伪负: {n_A}, B 类伪负: {n_B}, 类型标记已保存到 pseudo_pool_types.npy")

    return pseudo_pool


def _resolve_input_path(path):
    input_path = Path(path)
    if input_path.is_file():
        return input_path.resolve()
    local_path = Path(__file__).resolve().parent / input_path
    if local_path.is_file():
        return local_path.resolve()
    raise FileNotFoundError(f"找不到输入文件: {path}")


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def filter_storage_rows(df, time_type=None):
    required = {
        "P_has_negative", "Sign_Classification", "Storage_to_Load_Ratio",
        "Spike_Abnormal", "month",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"输入数据缺少必要列: {sorted(missing)}")

    mask = (
        (df["P_has_negative"] == 0)
        & (df["Sign_Classification"] == 2)
        & df["Storage_to_Load_Ratio"].between(0.01, 0.8)
        & (df["Spike_Abnormal"] == 0)
    )
    result = df.loc[mask].copy()
    if time_type is not None:
        if "Time_Type" not in result.columns:
            raise ValueError("输入数据缺少 Time_Type 列")
        result = result[result["Time_Type"] == time_type].copy()
    if result.empty:
        raise ValueError("筛选后没有可用样本")
    return result


def _normalize_split_weights(meta, values):
    values = np.asarray(values, dtype=np.float32).copy()
    for split_name in meta["split"].unique():
        mask = meta["split"].to_numpy() == split_name
        mean_value = float(values[mask].mean()) if mask.any() else 0.0
        if mean_value > 0:
            values[mask] /= mean_value
    return values


def _to_curve_matrix(df, columns, name):
    numeric = df[columns].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float32)
    invalid = np.argwhere(~np.isfinite(values))
    if invalid.size:
        row_index, column_index = invalid[0]
        source_row = int(df.iloc[row_index]["_source_row"])
        raise ValueError(
            f"{name} 曲线包含无效值: source_row={source_row}, column={columns[column_index]}"
        )
    return values


def _assign_user_level_split(meta, train_ratio, rng):
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio 必须位于 0 和 1 之间")
    users = meta["user_id"].drop_duplicates().to_numpy()
    if len(users) < 2:
        raise ValueError("未提供切分清单时，至少需要两个用户才能切分训练集和验证集")

    shuffled_users = rng.permutation(users)
    train_user_count = int(len(users) * train_ratio)
    train_user_count = min(max(train_user_count, 1), len(users) - 1)
    train_users = set(shuffled_users[:train_user_count].tolist())
    return np.where(meta["user_id"].isin(train_users), "train", "val")


def load_storage_data(data_path="processed_result.xlsx", train_ratio=0.9, time_type=None,
                      seed=1111, split_manifest_path=None, user_col="关联用户号",
                      policy_name="old", duplicate_policy="weight"):
    if duplicate_policy not in {"keep", "weight"}:
        raise ValueError("duplicate_policy 仅支持 keep 或 weight")

    source_path = _resolve_input_path(data_path)
    manifest_path = _resolve_input_path(split_manifest_path) if split_manifest_path else None
    raw_df = pd.read_excel(source_path).copy()
    raw_df["_source_row"] = raw_df.index.astype(np.int64)
    df = filter_storage_rows(raw_df, time_type=time_type)

    if user_col not in df.columns:
        raise ValueError(f"输入数据缺少用户列: {user_col}")
    if manifest_path:
        df = attach_user_split(df, manifest_path, user_col=user_col)
    else:
        df["_user_id"] = normalize_user_ids(df[user_col])

    p_columns = [f"P{i}" for i in range(1, 97)]
    q_columns = [f"Q{i}" for i in range(1, 97)]
    missing_curve_columns = [column for column in p_columns + q_columns if column not in df.columns]
    if missing_curve_columns:
        raise ValueError(f"输入数据缺少曲线列: {missing_curve_columns[:8]}")

    df = df.reset_index(drop=True)
    p_raw = _to_curve_matrix(df, p_columns, "P")
    q_raw = _to_curve_matrix(df, q_columns, "Q")
    if np.any(p_raw < -1e-6):
        row_index, point_index = np.argwhere(p_raw < -1e-6)[0]
        raise ValueError(
            f"筛选后的用户有功功率仍包含负值: source_row={int(df.iloc[row_index]['_source_row'])}, "
            f"column={p_columns[point_index]}"
        )

    meta_rows = []
    price_features_list = []

    date_column = "日期" if "日期" in df.columns else None
    industry_column = "所属行业" if "所属行业" in df.columns else None
    for index in range(len(df)):
        month = int(df.at[index, "month"])
        price_features = build_price_features(month=month, policy_name=policy_name)
        price_features_list.append(price_features)
        meta_rows.append({
            "sample_id": index,
            "source_row": int(df.at[index, "_source_row"]),
            "user_id": str(df.at[index, "_user_id"]),
            "date": str(df.at[index, date_column]) if date_column else "",
            "industry": str(df.at[index, industry_column]) if industry_column else "",
            "month": month,
            "policy_name": policy_name,
            "price_weight_seq": price_features["price_weight_seq"].astype(np.float32),
        })

    meta = pd.DataFrame(meta_rows)
    rng = np.random.RandomState(seed)

    if manifest_path:
        meta["split"] = df["_split"].astype(str).to_numpy()
        split_strategy = "user-manifest"
    else:
        meta["split"] = _assign_user_level_split(meta, train_ratio, rng)
        split_strategy = "user-random"

    train_indices_for_scaler = np.flatnonzero(
        meta["split"].to_numpy() == "train"
    ).astype(np.int64)
    q_global_scale = fit_q_global_scale(
        p_raw,
        q_raw,
        indices=train_indices_for_scaler,
    )

    x_list = []
    p_scale_refs = np.empty(len(df), dtype=np.float32)
    q_to_p_p95 = np.empty(len(df), dtype=np.float32)
    for index, price_features in enumerate(price_features_list):
        P_scaled, Q_scaled, p_ref = scale_pq(
            p_raw[index], q_raw[index], q_global_scale=q_global_scale
        )
        p_scale_refs[index] = p_ref
        q_to_p_p95[index] = np.percentile(np.abs(q_raw[index]) / p_ref, Q_GLOBAL_QUANTILE)
        x_list.append(np.stack([
            P_scaled,
            Q_scaled,
            price_features["season_flag"],
            price_features["price_role_seq"],
            price_features["price_role_jump"],
        ], axis=0))

    x = np.asarray(x_list, dtype=np.float32)
    labels = np.ones(len(x), dtype=np.float32)
    meta["p_scale_ref"] = p_scale_refs
    meta["q_to_p_p95"] = q_to_p_p95

    if date_column:
        duplicate_count = meta.groupby(["user_id", "date"])["sample_id"].transform("size").to_numpy()
    else:
        duplicate_count = np.ones(len(meta), dtype=np.int64)
    meta["duplicate_count"] = duplicate_count.astype(np.int64)
    duplicate_weight = np.ones(len(meta), dtype=np.float32)
    if duplicate_policy == "weight":
        duplicate_weight = 1.0 / np.maximum(duplicate_count, 1)

    user_count = meta.groupby(["split", "user_id"])["sample_id"].transform("size").to_numpy()
    user_weight = _normalize_split_weights(meta, 1.0 / np.maximum(user_count, 1))
    sample_weight = _normalize_split_weights(meta, user_weight * duplicate_weight)
    meta["user_weight"] = user_weight
    meta["duplicate_weight"] = duplicate_weight
    meta["sample_weight"] = sample_weight

    split_indices = {
        split_name: np.flatnonzero(meta["split"].to_numpy() == split_name).astype(np.int64)
        for split_name in ("train", "val", "test")
    }
    return {
        "X": torch.from_numpy(x),
        "Y": torch.from_numpy(labels),
        "P_raw": p_raw,
        "Q_raw": q_raw,
        "meta": meta,
        "split_indices": split_indices,
        "data_path": str(source_path),
        "data_sha256": file_sha256(source_path),
        "split_manifest_path": str(manifest_path) if manifest_path else "",
        "split_manifest_sha256": file_sha256(manifest_path) if manifest_path else "",
        "split_strategy": split_strategy,
        "normalization": {
            "p_quantile": P_SCALE_QUANTILE,
            "p_clip_max": P_CLIP_MAX,
            "q_global_quantile": Q_GLOBAL_QUANTILE,
            "q_global_scale": q_global_scale,
            "q_clip_abs": Q_CLIP_ABS,
        },
        "pipeline_version": DATA_PIPELINE_VERSION,
    }


def data_generator(train_ratio=0.9, time_type=None, seed=1111, return_meta=False,
                   split_manifest_path=None, user_col="关联用户号", policy_name="old",
                   data_path="processed_result.xlsx"):
    data = load_storage_data(
        data_path=data_path,
        train_ratio=train_ratio,
        time_type=time_type,
        seed=seed,
        split_manifest_path=split_manifest_path,
        user_col=user_col,
        policy_name=policy_name,
    )
    train_indices = data["split_indices"]["train"]
    val_indices = data["split_indices"]["val"]
    test_indices = data["split_indices"]["test"]

    train_index_tensor = torch.from_numpy(train_indices)
    val_index_tensor = torch.from_numpy(val_indices)
    x_train = data["X"].index_select(0, train_index_tensor)
    y_train = data["Y"].index_select(0, train_index_tensor)
    x_val = data["X"].index_select(0, val_index_tensor)
    y_val = data["Y"].index_select(0, val_index_tensor)

    print(
        f"数据切分: train={len(train_indices)}, val={len(val_indices)}, "
        f"reserved_test={len(test_indices)} | pipeline={data['pipeline_version']}"
    )
    print(
        f"归一化: P95→1, Q共享P尺度, "
        f"train_q95={data['normalization']['q_global_scale']:.6f}"
    )
    print(f"输入维度: {tuple(x_train.shape)}")
    if len(x_train) > 0:
        print(f"P范围: [{x_train[:, 0].min():.4f}, {x_train[:, 0].max():.4f}]")
        print(f"Q范围: [{x_train[:, 1].min():.4f}, {x_train[:, 1].max():.4f}]")

    if not return_meta:
        return x_train, y_train, x_val, y_val

    meta = data["meta"]
    meta_train = meta.iloc[train_indices].to_dict("records")
    meta_val = meta.iloc[val_indices].to_dict("records")
    return x_train, y_train, x_val, y_val, meta_train, meta_val


# 预留GUI接口
def get_policy_season_options(policy_name):
    if policy_name not in POLICY_REGISTRY:
        raise ValueError(f"未知 policy_name: {policy_name}")

    season_meta = POLICY_REGISTRY[policy_name]["season_meta"]
    out = []
    for season_name, meta in season_meta.items():
        out.append({
            "season_name": season_name,
            "gui_name": meta["gui_name"],
            "months": meta["months"],
        })
    return out


def get_policy_season_detail(policy_name, season_name):
    cfg = get_policy_cfg(policy_name=policy_name, season_name=season_name)
    return {
        "policy_name": cfg["policy_name"],
        "season_name": cfg["season_name"],
        "gui_name": cfg["season_meta"]["gui_name"],
        "months": cfg["season_meta"]["months"],
        "segments": cfg["segments"],
    }
