# 用户侧储能识别

**简体中文** | [English](README.en.md)

从用户日负荷曲线中识别储能充放电行为，支持相对电价政策迁移、教师学生蒸馏以及本地可视化排查。

## 项目目标

训练阶段使用用户有功功率、储能充放电功率与相对电价时段信息；部署阶段仅使用用户有功功率和当前政策的峰、平、谷时段，输出每日储能行为证据分数及用户级排查结果。

目标是识别利益驱动下的低电价时段充电、高电价时段放电及其负荷响应。分数表示储能行为证据强度，未经过严格非储能用户真值校准，不能直接解释为设备存在概率。

## 核心方法

| 技术内容 | 说明 |
| --- | --- |
| 用户级数据切分 | 同一户号的全部日期仅属于一个数据集合，降低用户泄漏。 |
| 自监督表征 | 利用有功功率与储能充放电功率的同步增强学习时序联动。 |
| 多目标教师 | 学习物理证据、功率响应可信度及相对电价下的套利行为。 |
| 政策迁移与反事实 | 用政策角色迁移、时序错配和储能分量衰减约束行为判断。 |
| 教师学生蒸馏 | 将完整信息教师的连续目标传给仅依赖有功功率的学生。 |
| 本地排查界面 | 文件上传、表内滚动、户号包含搜索、黄色匹配高亮及结果下载。 |

## 代码结构

```text
energy-storage-identification/
├─ storage/
│  ├─ temporal.py
│  ├─ model.py
│  ├─ data.py
│  ├─ split.py
│  ├─ stage1.py
│  ├─ teacher.py
│  ├─ targets.py
│  ├─ policy.py
│  ├─ distill.py
│  ├─ student.py
│  ├─ infer.py
│  ├─ acceptance.py
│  ├─ diag_stage1.py
│  ├─ diag_teacher.py
│  ├─ visualize.py
│  ├─ generate.py
│  ├─ step_1/best_model_st1.pt
│  ├─ step_2/best_model_st2.pt
│  ├─ train_student/best_model_student.pt
│  └─ web/
├─ docs/
├─ requirements.txt
└─ start_web.bat
```

| 文件 | 作用 |
| --- | --- |
| `storage/temporal.py`、`storage/model.py` | 因果时间卷积网络与长短期记忆网络。 |
| `storage/data.py`、`storage/split.py` | 数据读取、归一化、政策特征和用户级切分。 |
| `storage/stage1.py` | 第一阶段自监督表征训练。 |
| `storage/teacher.py`、`storage/targets.py`、`storage/policy.py` | 教师训练、行为目标构建和政策迁移。 |
| `storage/distill.py`、`storage/student.py` | 蒸馏数据构建与学生训练。 |
| `storage/infer.py` | 日级推理、用户级汇总和表格输出。 |
| `storage/acceptance.py`、`storage/diag_stage1.py`、`storage/diag_teacher.py` | 表征验收及教师诊断。 |
| `storage/visualize.py` | 教师目标与功率响应可视化。 |
| `storage/generate.py` | 可选离线生成对抗网络候选生成，不自动接入训练。 |
| `storage/web/` | Java 本地服务和网页界面。 |

## 环境配置

Python 3.10 或更新版本，PyTorch 2.1 或更新版本。网页服务需要 Java 开发工具包 17 或更新版本。训练可选使用 NVIDIA 图形处理器及相应的计算运行环境。

在仓库根目录执行以下命令。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 快速推理

仓库包含三个训练完成的模型权重，不包含原始业务数据、处理后的数据、用户清单或训练缓存。

Windows 用户可双击根目录的 `start_web.bat`，在本机浏览器中上传表格并下载结果。

```text
http://127.0.0.1:8765
```

也可直接执行推理命令。

```powershell
python -m storage.infer `
  --ckpt train_student/best_model_student.pt `
  --input_path D:/private-data/input.xlsx `
  --output_path D:/private-data/storage_result.xlsx `
  --policy_name new
```

推理输入推荐首行为 `数据日期`、`户号`，随后为 `0:00` 至 `23:45` 的 96 个有功功率点，间隔 15 分钟。

## 输出与分级

简表输出四列：户号、储能情况、总天数、有效天数。有效记录要求日期、户号以及全部负荷点均完整有效。

当前实现的默认分级边界是：分数大于 80 为强储能；分数大于等于 60 且小于等于 80 为中储能；其余为无或储能占比不高。用户级分数为有效日中最高 20% 日分数的均值，样本数量向上取整且至少取一天。

这些边界和汇总方式属于业务筛查规则。详细输出还保留每日结果和各等级有效日占比。

## 文档

- [数据格式](docs/DATA_FORMAT.md)
- [训练流程](docs/TRAINING_PIPELINE.md)
- [模型说明](docs/MODEL_CARD.md)
- [本地网页服务](storage/web/README.md)

光伏、生产启停、自备电源和计量异常可能影响识别效果。新政策或新区域应结合人工复核进行验证；反事实样本的区分指标不能替代真实非储能用户的识别准确率。
