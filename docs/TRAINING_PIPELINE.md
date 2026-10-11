# 训练流程

**简体中文** | [English](TRAINING_PIPELINE.en.md)

全部命令从仓库根目录执行。业务数据与用户切分清单保存在仓库外部。相对的模型路径默认相对于 `storage` 目录解析；下面明确指定的第一阶段输出目录则相对于运行目录解析。

## 1. 用户级切分

```powershell
python -m storage.split `
  --input D:/storage-private-data/processed_result.xlsx `
  --output D:/storage-private-data/data_split/user_split.csv `
  --summary D:/storage-private-data/data_split/user_split_summary.json
```

同一用户的全部日期只属于训练、验证或测试集合中的一个。

## 2. 自监督表征训练

```powershell
python -m storage.stage1 `
  --data_path D:/storage-private-data/processed_result.xlsx `
  --split_manifest D:/storage-private-data/data_split/user_split.csv `
  --save_dir storage/step_1 `
  --cuda
```

使用有功功率与储能充放电功率的同步增强，以及方差、不变性和协方差正则化目标，学习稳定的时序表征。没有图形处理器时省略 `--cuda`。

## 3. 表征验收

```powershell
python -m storage.acceptance `
  --ckpt storage/step_1/best_model_st1.pt `
  --output storage/step_1/best_model_st1_acceptance.json
```

## 4. 教师训练

```powershell
python -m storage.teacher `
  --data_path D:/storage-private-data/processed_result.xlsx `
  --split_manifest D:/storage-private-data/data_split/user_split.csv `
  --phase1_ckpt step_1/best_model_st1.pt `
  --save_dir step_2 `
  --cuda
```

基于原始有功功率与储能充放电功率构造物理证据、响应可信度与套利行为目标。训练包含政策迁移、错配、层级一致性和储能分量衰减约束。默认先冻结主干两轮，再解冻并在三轮内逐步提升主干学习率。

真实训练记录按旧政策读取；政策变化样本由重构流程生成。

## 5. 蒸馏数据构建

```powershell
python -m storage.distill `
  --phase2_ckpt step_2/best_model_st2.pt `
  --data_path D:/storage-private-data/processed_result.xlsx `
  --split_manifest D:/storage-private-data/data_split/user_split.csv `
  --out_dir step_3
```

## 6. 学生训练

```powershell
python -m storage.student `
  --student_npz step_3/student_train_dataset.npz `
  --phase2_ckpt step_2/best_model_st2.pt `
  --save_dir train_student
```

学生使用归一化有功功率、正向变化量、负向变化量、相对电价角色及角色跳变，部署时无需储能设备功率数据。

## 7. 模型诊断

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

## 8. 推理

```powershell
python -m storage.infer `
  --input_path D:/private-data/input.xlsx `
  --output_path D:/private-data/storage_result.xlsx `
  --policy_name new
```

使用 `--compact_output` 仅导出户号、储能情况、总天数和有效天数；不加该参数则输出用户汇总和每日明细。
