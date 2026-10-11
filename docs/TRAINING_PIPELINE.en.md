# Training Workflow

[简体中文](TRAINING_PIPELINE.md) | **English**

Run all commands from the repository root. Keep business data and customer-split manifests outside the repository. Relative model paths are normally resolved against the `storage` directory; the explicitly specified first-stage output directory below is resolved against the working directory.

## Customer-level splitting

```powershell
python -m storage.split `
  --input D:/storage-private-data/processed_result.xlsx `
  --output D:/storage-private-data/data_split/user_split.csv `
  --summary D:/storage-private-data/data_split/user_split_summary.json
```

All dates for one customer belong to exactly one of the training, validation, or test splits.

## Self-supervised representation training

```powershell
python -m storage.stage1 `
  --data_path D:/storage-private-data/processed_result.xlsx `
  --split_manifest D:/storage-private-data/data_split/user_split.csv `
  --save_dir storage/step_1 `
  --cuda
```

Learn stable temporal representations using synchronized augmentation of active power and storage charging and discharging power, together with variance, invariance, and covariance regularization. Omit `--cuda` when no graphics processing unit is available.

## Representation acceptance

```powershell
python -m storage.acceptance `
  --ckpt storage/step_1/best_model_st1.pt `
  --output storage/step_1/best_model_st1_acceptance.json
```

## Teacher training

```powershell
python -m storage.teacher `
  --data_path D:/storage-private-data/processed_result.xlsx `
  --split_manifest D:/storage-private-data/data_split/user_split.csv `
  --phase1_ckpt step_1/best_model_st1.pt `
  --save_dir step_2 `
  --cuda
```

Construct physical-evidence, response-confidence, and arbitrage-behavior targets from raw active power and storage charging and discharging power. Training includes policy transfer, mismatch, hierarchical consistency, and storage-component attenuation constraints. By default, the backbone is frozen for two epochs, then unfrozen with its learning rate ramped over three epochs.

Real training records are read under the old policy; policy-change samples are produced by the reconstruction process.

## Distillation dataset construction

```powershell
python -m storage.distill `
  --phase2_ckpt step_2/best_model_st2.pt `
  --data_path D:/storage-private-data/processed_result.xlsx `
  --split_manifest D:/storage-private-data/data_split/user_split.csv `
  --out_dir step_3
```

## Student training

```powershell
python -m storage.student `
  --student_npz step_3/student_train_dataset.npz `
  --phase2_ckpt step_2/best_model_st2.pt `
  --save_dir train_student
```

The student uses normalized active power, positive changes, negative changes, relative tariff roles, and tariff-role transitions. Deployment does not require storage-device power measurements.

## Model diagnostics

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

Use validation data for tuning and model selection, and reserve test data for final reporting. Discrimination on teacher counterfactual samples is used only for constraints and diagnostics.

## Inference

```powershell
python -m storage.infer `
  --input_path D:/private-data/input.xlsx `
  --output_path D:/private-data/storage_result.xlsx `
  --policy_name new
```

Use `--compact_output` to export only customer number, storage assessment, total days, and valid days. Without this flag, the output contains customer summaries and daily details.
