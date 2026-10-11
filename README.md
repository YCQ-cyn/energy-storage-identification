# 面向政策迁移的用户侧储能行为识别

本项目从 15 分钟级用户负荷曲线中识别储能行为。训练阶段利用有功功率 `P`、储能功率 `Q` 与相对电价角色构建教师监督，部署阶段仅使用用户有功功率 `P` 和电价时段角色输出储能证据分数及用户级排查结果。

项目不是普通二分类：训练数据中的用户均可能具有不同强弱的储能行为，模型学习的是谷充峰放、功率响应、持续运行与套利一致性等连续行为证据，再按业务阈值形成用户级分级。

## 核心特点

- 用户级切分：同一户号只属于训练、验证或测试集合之一，避免按天随机切分造成用户泄漏。
- 政策迁移：使用峰、平、谷相对角色，不绑定固定月份、具体电价金额或固定充放电时刻。
- 特权信息蒸馏：教师训练可见 `P/Q`，学生部署仅依赖 `P` 与相对电价角色。
- 反事实约束：通过政策迁移、时序错配和储能分量衰减，约束模型识别行为而非记忆旧政策。
- 用户级输出：按每日连续分数汇总为强储能、中储能、无或储能占比不高。
- 本地前端：支持 Excel/CSV 上传、结果预览、户号片段搜索和 Excel 下载。

## 仓库结构

```text
.
├─ TCN/
│  ├─ tcn.py                         # 因果 TCN 主干
│  └─ adding_problem/
│     ├─ build_user_split.py         # 用户级 train/val/test 切分
│     ├─ add_test_st1.py             # Stage-1 自监督表征训练
│     ├─ add_test_st2.py             # Stage-2 多约束教师训练
│     ├─ add_test_st3.py             # 构建学生蒸馏数据
│     ├─ add_test_student.py         # P-only 学生训练
│     ├─ add_test_infer.py           # 日级推理与用户级汇总
│     ├─ teacher_targets.py           # 教师行为目标构造
│     ├─ policy_transfer.py           # 相对电价角色迁移
│     ├─ utils.py                     # 数据读取、归一化与特征构造
│     ├─ diag_st1.py                  # Stage-1 综合诊断
│     ├─ stage1_acceptance.py         # Stage-1 验收
│     ├─ diag_st2.py                  # Stage-2 综合诊断
│     ├─ visualize_teacher_labels.py  # 教师约束可视化
│     ├─ generate_gan_candidates.py   # 可选离线弱储能候选生成
│     └─ storage_web/                 # Java 本地服务与静态前端
├─ docs/
├─ requirements.txt
└─ start_web.bat
```

## 环境要求

- Python 3.10 或更高版本
- PyTorch 2.1 或更高版本
- Windows 前端一键启动需要 JDK 17 或更高版本
- CUDA 可选；训练推荐使用 NVIDIA GPU

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 快速使用

仓库附带三个训练完成的模型权重，不包含任何原始或处理后的业务数据。

### Web 页面

Windows 双击根目录的 `start_web.bat`。页面默认打开：

```text
http://127.0.0.1:8765
```

### 命令行推理

在仓库根目录执行：

```powershell
python -m TCN.adding_problem.add_test_infer `
  --ckpt train_student/best_model_student.pt `
  --input_path D:/private-data/input.xlsx `
  --output_path D:/private-data/storage_result.xlsx `
  --policy_name new
```

输入表格格式见 [数据格式](docs/DATA_FORMAT.md)，完整训练顺序见 [训练流程](docs/TRAINING_PIPELINE.md)。

## 输出口径

用户级简表包含：

- 户号
- 储能情况
- 总天数
- 有效天数

默认以每日储能分数进行用户级汇总：`>=80` 为强储能日，`>=60` 且 `<80` 为中储能日，其余为无或储能占比不高。阈值属于业务筛查口径，不等同于具有严格负样本定义的分类准确率。

## 数据与隐私

本仓库不包含 Excel、CSV、NPZ、用户切分清单、用户标识、训练缓存、诊断图或运行日志。请将数据保存在仓库外部，并通过命令行参数传入绝对路径。

## 使用边界

- 输出是储能行为证据和辅助排查分级，不应直接替代现场核查。
- 光伏、自备电源、生产班次变化和计量异常可能影响纯 `P` 推理结果。
- 当新政策改变峰平谷时段时，应传入对应政策角色；模型不依赖具体电价金额。
- 没有严格非储能真值时，不应把反事实伪负样本上的区分能力表述为真实场景分类准确率。

更多说明见 [模型说明](docs/MODEL_CARD.md)。
