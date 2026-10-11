# Data Format

[简体中文](DATA_FORMAT.md) | **English**

## Inference input

Excel workbooks (`.xlsx`) and comma-separated text files (`.csv`) are supported. Each row represents one customer's active-power curve for one day.

```text
数据日期  户号  0:00  0:15  0:30  ...  23:30  23:45
```

| Content | Requirement |
| --- | --- |
| Data date | Excel date values, `YYYY-MM-DD`, `YYYY/MM/DD`, `YYYYMMDD`, and common Chinese date strings are supported. |
| Customer number | Store as text to preserve long identifiers and leading zeros. |
| Active power | 96 measurements per day, sampled every 15 minutes, using consistent units. |
| Valid day | The date, customer number, and all load measurements must be valid; records with missing measurements are skipped during inference. |

Existing date aliases include “数据日期”, “日期”, “用电日期”, and “采集日期”. Customer-number aliases include “户号”, “关联用户号”, “用户号”, and “用户编号”. Refer to `storage/infer.py` for the exact supported aliases and time-header parsing.

## Training input

Training uses a preprocessed Excel workbook. The existing curve-column names are fixed interface fields; the table below describes their full physical meaning.

| Actual fields | Full meaning |
| --- | --- |
| `关联用户号` | Customer identifier used for customer-level splitting by default. |
| `month` | Month of the record, used to select policy-specific time periods. |
| `P1` through `P96` | The 96 customer active-power measurements. |
| `Q1` through `Q96` | The 96 synchronized energy-storage charging and discharging power measurements. |
| `P_has_negative` | Negative active-power flag. |
| `Sign_Classification` | Sign-classification field from the existing preprocessing process. |
| `Storage_to_Load_Ratio` | Storage-to-load ratio field. |
| `Spike_Abnormal` | Abnormal-spike flag. |
| `日期` | Optional original date retained in sample metadata. |
| `Time_Type` | Required when filtering by time type. |

In this project, energy-storage charging and discharging power is the storage device's power component, not electrical reactive power. It is aligned with customer active power by day and sampling time to construct baseline load and storage-behavior supervision.

The current training filter retains records with a negative active-power flag of 0, a sign-classification value of 2, a storage-to-load ratio between 0.01 and 0.8, and an abnormal-spike flag of 0. Before using another data source, verify the origin and meaning of these fields.

## Data storage

Keep business datasets and customer-split manifests outside the repository and pass their absolute paths to the commands.

```text
D:/storage-private-data/
├─ processed_result.xlsx
└─ data_split/
   └─ user_split.csv
```

The `.gitignore` rules exclude data files, runtime caches, and diagnostic output while explicitly retaining the released model checkpoints.
