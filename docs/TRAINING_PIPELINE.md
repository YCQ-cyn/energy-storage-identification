# 训练流程

以下命令均在仓库根目录执行。示例将私有数据与切分清单放在仓库外部。

## 1. 用户级切分

```powershell
python -m TCN.adding_problem.build_user_split `
  --input D:/storage-private-data/processed_result.xlsx `
  --output D:/storage-private-data/data_split/user_split.csv `
  --summary D:/storage-private-data/data_split/user_split_summary.json
```

切分以户号为单位，同一用户的所有日期必须位于同一集合。

## 2. Stage-1 自监督表征

```powershell
python -m TCN.adding_problem.add_test_st1 `
  --data_path D:/storage-private-data/processed_result.xlsx `
  --split_manifest D:/storage-private-data/data_split/user_split.csv `
  --save_dir TCN/adding_problem/step_1 `
  --cuda
```

Stage-1 使用同步增强和 VICReg 类目标学习 `P/Q` 联动表征，不使用普通二分类标签。

## 3. Stage-1 验收

```powershell
python -m TCN.adding_problem.stage1_acceptance `
  --ckpt TCN/adding_problem/step_1/best_model_st1.pt `
  --output TCN/adding_problem/step_1/best_model_st1_acceptance.json
```

## 4. Stage-2 教师训练

```powershell
python -m TCN.adding_problem.add_test_st2 `
  --data_path D:/storage-private-data/processed_result.xlsx `
  --split_manifest D:/storage-private-data/data_split/user_split.csv `
  --phase1_ckpt TCN/adding_problem/step_1/best_model_st1.pt `
  --save_dir TCN/adding_problem/step_2 `
  --cuda
```

教师使用原始 `P/Q` 构造物理、响应和套利行为目标，并通过政策迁移、错配及储能分量衰减约束减少旧政策捷径。

## 5. 构建蒸馏数据

```powershell
python -m TCN.adding_problem.add_test_st3 `
  --phase2_ckpt step_2/best_model_st2.pt `
  --data_path D:/storage-private-data/processed_result.xlsx `
  --split_manifest D:/storage-private-data/data_split/user_split.csv `
  --out_dir step_3
```

## 6. P-only 学生训练

```powershell
python -m TCN.adding_problem.add_test_student `
  --student_npz step_3/student_train_dataset.npz `
  --phase2_ckpt step_2/best_model_st2.pt `
  --save_dir train_student
```

学生输入为 `P_norm`、正负变化量、相对电价角色和角色跳变，不读取部署时不可获得的 `Q`。

## 7. 诊断

```powershell
python -m TCN.adding_problem.diag_st1 --ckpt TCN/adding_problem/step_1/best_model_st1.pt --cuda
python -m TCN.adding_problem.diag_st2 --phase2_ckpt TCN/adding_problem/step_2/best_model_st2.pt --cuda
```

验证集用于调参和模型选择，测试集只用于最终报告。伪负样本及政策反事实只用于行为约束与可分性诊断，不能替代真实非储能真值。
