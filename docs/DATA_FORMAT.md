# 数据格式 / Data Format

## 推理输入 / Inference input

支持 Excel 工作簿（`.xlsx`）和逗号分隔文本文件（`.csv`）。每行表示一个用户一天的有功功率曲线。

Excel workbooks (`.xlsx`) and comma-separated text files (`.csv`) are supported. Each row represents one customer's active-power curve for one day.

```text
数据日期  户号  0:00  0:15  0:30  ...  23:30  23:45
```

| 内容 / Content | 要求 / Requirement |
| --- | --- |
| 数据日期 / Data date | 支持 Excel 日期、`YYYY-MM-DD`、`YYYY/MM/DD`、`YYYYMMDD` 和常见中文日期。 / Excel date values, the listed date formats, and common Chinese date strings are supported. |
| 户号 / Customer number | 推荐作为文本保存，避免长数字精度损失或前导零丢失。 / Store as text to preserve long identifiers and leading zeros. |
| 有功功率 / Active power | 单日 96 点，采样间隔 15 分钟，使用一致的计量单位。 / 96 measurements per day, sampled every 15 minutes, using consistent units. |
| 有效日 / Valid day | 日期、户号和全部负荷点均需有效，任一点缺失的记录跳过推理。 / The date, customer number, and all load measurements must be valid; records with missing measurements are skipped during inference. |

日期列支持“数据日期”“日期”“用电日期”“采集日期”等现有别名；户号列支持“户号”“关联用户号”“用户号”“用户编号”等现有别名。实际别名和时间列解析以 `storage/infer.py` 为准。

Existing date aliases include “数据日期”, “日期”, “用电日期”, and “采集日期”. Customer-number aliases include “户号”, “关联用户号”, “用户号”, and “用户编号”. Refer to `storage/infer.py` for the exact supported aliases and time-header parsing.

## 训练输入 / Training input

训练接口使用预处理后的 Excel 工作簿。现有曲线列名是固定接口字段，下表用完整物理含义说明。

Training uses a preprocessed Excel workbook. The existing curve-column names are fixed interface fields; the table below describes their full physical meaning.

| 实际字段 / Actual fields | 完整含义 / Full meaning |
| --- | --- |
| `关联用户号` | 用户标识，默认用于用户级切分。 / Customer identifier used for customer-level splitting by default. |
| `month` | 数据所属月份，用于选择当前政策的时段。 / Month of the record, used to select policy-specific time periods. |
| `P1` 至 `P96` | 用户有功功率的 96 个采样点。 / The 96 customer active-power measurements. |
| `Q1` 至 `Q96` | 储能充放电功率的 96 个同步采样点。 / The 96 synchronized energy-storage charging and discharging power measurements. |
| `P_has_negative` | 有功功率负值标记。 / Negative active-power flag. |
| `Sign_Classification` | 既有预处理流程的符号分类字段。 / Sign-classification field from the existing preprocessing process. |
| `Storage_to_Load_Ratio` | 储能与负荷占比字段。 / Storage-to-load ratio field. |
| `Spike_Abnormal` | 异常尖峰标记。 / Abnormal-spike flag. |
| `日期` | 可选的原始日期，用于保留样本元信息。 / Optional original date retained in sample metadata. |
| `Time_Type` | 使用时间类型筛选参数时需要。 / Required when filtering by time type. |

本项目中的储能充放电功率是储能设备的功率分量，不是电气意义上的无功功率。它与用户有功功率按同一天、同一采样时刻对齐，用于构造基础负荷和储能行为监督。

In this project, energy-storage charging and discharging power is the storage device's power component, not electrical reactive power. It is aligned with customer active power by day and sampling time to construct baseline load and storage-behavior supervision.

现有训练筛选保留：有功功率负值标记为 0、符号分类为 2、储能与负荷占比在 0.01 至 0.8 之间且异常尖峰标记为 0 的记录。首次使用不同来源数据前，应核对这些字段的来源和含义。

The current training filter retains records with a negative active-power flag of 0, a sign-classification value of 2, a storage-to-load ratio between 0.01 and 0.8, and an abnormal-spike flag of 0. Before using another data source, verify the origin and meaning of these fields.

## 数据存放 / Data storage

建议将业务数据与用户切分清单保存在仓库外部，通过绝对路径传入。

Keep business datasets and customer-split manifests outside the repository and pass their absolute paths to the commands.

```text
D:/storage-private-data/
├─ processed_result.xlsx
└─ data_split/
   └─ user_split.csv
```

`.gitignore` 会排除数据文件、运行缓存和诊断输出；发布模型权重通过明确的例外规则保留。

The `.gitignore` rules exclude data files, runtime caches, and diagnostic output while explicitly retaining the released model checkpoints.
