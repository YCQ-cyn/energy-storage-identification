# 训练流程 / Training Workflow

全部命令从仓库根目录执行。业务数据与用户切分清单保存在仓库外部。相对的模型路径默认相对于 `storage` 目录解析；下面明确指定的第一阶段输出目录则相对于运行目录解析。

Run all commands from the repository root. Keep business data and customer-split manifests outside the repository. Relative model paths are normally resolved against the `storage` directory; the explicitly specified first-stage output directory below is resolved against the working directory.

## 1. 用户级切分 / Customer-level splitting

```powershell
python -m storage.split `
  --input D:/storage-private-data/processed_result.xlsx `
  --output D:/storage-private-data/data_split/user_split.csv `
  --summary D:/storage-private-data/data_split/user_split_summary.json
```

同一用户的全部日期只属于训练、验证或测试集合中的一个。 / All dates for one customer belong to exactly one of the training, validation, or test splits.

## 2. 自监督表征训练 / Self-supervised representation training

```powershell
python -m storage.stage1 `
  --data_path D:/storage-private-data/processed_result.xlsx `
  --split_manifest D:/storage-private-data/data_split/user_split.csv `
  --save_dir storage/step_1 `
  --cuda
```

使用有功功率与储能充放电功率的同步增强，以及方差、不变性和协方差正则化目标，学习稳定的时序表征。没有图形处理器时省略 `--cuda`。

Learn stable temporal representations using synchronized augmentation of active power and storage charging and discharging power, together with variance, invariance, and covariance regularization. Omit `--cuda` when no graphics processing unit is available.

## 3. 表征验收 / Representation acceptance

```powershell
python -m storage.acceptance `
  --ckpt storage/step_1/best_model_st1.pt `
  --output storage/step_1/best_model_st1_acceptance.json
```

## 4. 教师训练 / Teacher training

```powershell
python -m storage.teacher `
  --data_path D:/storage-private-data/processed_result.xlsx `
  --split_manifest D:/storage-private-data/data_split/user_split.csv `
  --phase1_ckpt step_1/best_model_st1.pt `
  --save_dir step_2 `
  --cuda
```

基于原始有功功率与储能充放电功率构造物理证据、响应可信度与套利行为目标。训练包含政策迁移、错配、层级一致性和储能分量衰减约束。默认先冻结主干两轮，再解冻并在三轮内逐步提升主干学习率。

Construct physical-evidence, response-confidence, and arbitrage-behavior targets from raw active power and storage charging and discharging power. Training includes policy transfer, mismatch, hierarchical consistency, and storage-component attenuation constraints. By default, the backbone is frozen for two epochs, then unfrozen with its learning rate ramped over three epochs.

真实训练记录按旧政策读取；政策变化样本由重构流程生成。 / Real training records are read under the old policy; policy-change samples are produced by the reconstruction process.

## 5. 蒸馏数据构建 / Distillation dataset construction

```powershell
python -m storage.distill `
  --phase2_ckpt step_2/best_model_st2.pt `
  --data_path D:/storage-private-data/processed_result.xlsx `
  --split_manifest D:/storage-private-data/data_split/user_split.csv `
  --out_dir step_3
```

## 6. 学生训练 / Student training

```powershell
python -m storage.student `
  --student_npz step_3/student_train_dataset.npz `
  --phase2_ckpt step_2/best_model_st2.pt `
  --save_dir train_student
```

学生使用归一化有功功率、正向变化量、负向变化量、相对电价角色及角色跳变，部署时无需储能设备功率数据。

The student uses normalized active power, positive changes, negative changes, relative tariff roles, and tariff-role transitions. Deployment does not require storage-device power measurements.

## 7. 模型诊断 / Model diagnostics

```powershell
python -m storage.diag_stage1 `
  --ckpt storage/step_1/best_model_st1.pt `
  --data_path D:/storage-private-data/processed_result.xlsx `
  --split_manifest D:/storage-private-data/data_split/user_split.csv `
  --cuda

python -m storage.diag_teacher `
  --phase2_ckpt step_2/best_model_st2.pt `
  --data_path D:/storage-private-data/processed_result.xlsx `
  --split_manifest D:/storage-private-data/data_split/user_split.csv `
  --cuda
```

验证集用于调参及模型选择，测试集仅用于最终报告。教师反事实样本的区分能力仅用于约束和诊断。

Use validation data for tuning and model selection, and reserve test data for final reporting. Discrimination on teacher counterfactual samples is used only for constraints and diagnostics.

## 8. 推理 / Inference

```powershell
python -m storage.infer `
  --input_path D:/private-data/input.xlsx `
  --output_path D:/private-data/storage_result.xlsx `
  --policy_name new
```

使用 `--compact_output` 仅导出户号、储能情况、总天数和有效天数；不加该参数则输出用户汇总和每日明细。

Use `--compact_output` to export only customer number, storage assessment, total days, and valid days. Without this flag, the output contains customer summaries and daily details.
