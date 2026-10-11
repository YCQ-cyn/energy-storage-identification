# User-Side Energy Storage Identification

[简体中文](README.md) | **English**

Identify energy storage charging and discharging behavior from daily customer load curves, with relative electricity tariff adaptation, teacher–student distillation, and a local screening interface.

## Project objective

Training uses customer active power, energy storage charging and discharging power, and relative electricity tariff periods. Deployment uses only customer active power and the current peak, standard, and off-peak periods to produce daily behavior evidence scores and customer-level screening results.

The objective is to identify economically motivated charging during lower-priced periods, discharging during higher-priced periods, and the associated load response. Scores represent the strength of behavioral evidence; they are not calibrated probabilities of equipment presence against verified non-storage customers.

## Core approach

| Component | Description |
| --- | --- |
| Customer-level splitting | All dates for one customer belong to a single split, reducing customer leakage. |
| Self-supervised representation | Learn temporal relationships through synchronized augmentation of active power and storage charging and discharging power. |
| Multi-target teacher | Learn physical evidence, power-response confidence, and tariff-relative arbitrage behavior. |
| Policy transfer and counterfactuals | Use tariff-role transfer, temporal mismatch, and attenuation of the storage component to constrain behavior assessment. |
| Teacher–student distillation | Transfer continuous targets from a teacher with privileged information to a student that requires only active power measurements. |
| Local screening interface | File upload, in-table scrolling, substring search for customer numbers, yellow match highlighting, and result download. |

## Code structure

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

| File | Purpose |
| --- | --- |
| `storage/temporal.py`, `storage/model.py` | Causal temporal convolutional network and long short-term memory network. |
| `storage/data.py`, `storage/split.py` | Data loading, normalization, tariff features, and customer-level splitting. |
| `storage/stage1.py` | First-stage self-supervised representation training. |
| `storage/teacher.py`, `storage/targets.py`, `storage/policy.py` | Teacher training, behavior-target construction, and policy transfer. |
| `storage/distill.py`, `storage/student.py` | Distillation dataset construction and student training. |
| `storage/infer.py` | Daily inference, customer-level aggregation, and spreadsheet output. |
| `storage/acceptance.py`, `storage/diag_stage1.py`, `storage/diag_teacher.py` | Representation acceptance checks and teacher diagnostics. |
| `storage/visualize.py` | Teacher-target and power-response visualization. |
| `storage/generate.py` | Optional offline candidate generation with a generative adversarial network; candidates are not automatically used for training. |
| `storage/web/` | Java local service and browser interface. |

## Environment setup

Use Python 3.10 or newer and PyTorch 2.1 or newer. The web service requires Java Development Kit 17 or newer. Training can optionally use an NVIDIA graphics processing unit with a compatible compute runtime.

Run these commands from the repository root.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Quick inference

The repository includes three trained model checkpoints. It does not contain raw business data, processed datasets, customer lists, or training caches.

On Windows, double-click `start_web.bat` in the repository root to upload a spreadsheet and download results through the local browser interface.

```text
http://127.0.0.1:8765
```

Inference is also available from the command line.

```powershell
python -m storage.infer `
  --ckpt train_student/best_model_student.pt `
  --input_path D:/private-data/input.xlsx `
  --output_path D:/private-data/storage_result.xlsx `
  --policy_name new
```

The recommended inference headers are `数据日期` (data date), `户号` (customer number), followed by 96 active-power measurements from `0:00` to `23:45` at 15-minute intervals.

## Output and classification

The compact output contains four columns: customer number, storage assessment, total days, and valid days. A valid record requires a valid date, customer number, and all load measurements.

The current default boundaries are: above 80 for strong storage behavior; from 60 through 80, inclusive, for moderate storage behavior; otherwise, no storage evidence or a low storage share. The customer-level score is the mean of the highest-scoring 20% of valid days, with the selected count rounded upward and at least one day selected.

These boundaries and the aggregation rule are business screening settings. Detailed output also retains daily results and the proportion of valid days in each assessment category.

## Documentation

- [Data format](docs/DATA_FORMAT.en.md)
- [Training workflow](docs/TRAINING_PIPELINE.en.md)
- [Model card](docs/MODEL_CARD.en.md)
- [Local web service](storage/web/README.en.md)

Photovoltaic generation, production starts and stops, onsite generation, and metering anomalies can affect identification. Validate new policies and regions with manual review; counterfactual discrimination metrics do not replace accuracy measured on verified non-storage customers.
