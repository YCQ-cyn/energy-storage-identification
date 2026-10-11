# User-Side Energy Storage Identification

[简体中文](README.md) | **English**

An auxiliary screening project for electricity marketing and operations teams. It analyzes the load response associated with energy storage charging and discharging using daily customer load curves and the applicable time-of-use tariff policy, producing customer-level results to prioritize manual review.

## Business context and objective

Customer-side energy storage primarily reduces electricity costs by charging during lower-priced periods and discharging during higher-priced periods. Its operation is superimposed on the customer's existing load. Production starts and stops, load shifting, and photovoltaic generation also change that load, so an isolated increase or decrease is insufficient evidence.

The project identifies economically motivated charging and discharging behavior by combining load changes, sustained operation, and relative tariff preferences. It supports three practical questions: which customers show stronger storage evidence, which customers require further review, and how many valid days support each assessment.

Screening usually has access to total customer load without independent storage-device measurements. Training therefore uses customer active power and storage charging and discharging power to establish supervision, while deployment requires only customer active power and the applicable peak, standard, and off-peak periods.

## Business workflow

1. Upload daily load records: each row contains a date, a customer number, and 96 active-power measurements at 15-minute intervals.
2. Select the applicable tariff policy: peak, standard, and off-peak periods are generated using the record month; exact electricity prices are unnecessary.
3. Identify valid records: records with an invalid date, customer number, or load measurement are skipped during inference, while total-day and valid-day counts are retained.
4. Produce customer-level screening results: preview the results, search customer numbers, and download the workbook for manual review and subsequent field screening.

The local service provides file upload, scrolling within the results table, customer-number substring search, highlighted matches, and result download. Files are processed locally.

## Identification method

### 1. Learn how storage operation appears in total load

The first stage learns temporal representations through self-supervised training on synchronized customer active power and storage charging and discharging power. Synchronized augmentation preserves their timing and relationship; variance, invariance, and covariance regularization help maintain informative representations.

The network combines a causal temporal convolutional network with a long short-term memory network to capture local power changes and sustained operation within a day, providing representations for the teacher and student.

### 2. Teach physical response and tariff-driven behavior together

The second-stage teacher uses independent storage-device power available during training. Behavioral supervision is constructed from raw power values and used alongside normalized network inputs, avoiding the interpretation of independently scaled amplitudes as original physical quantities.

| Evidence layer | What the teacher learns | Business interpretation |
| --- | --- | --- |
| Physical relationships | Alignment of customer active power with storage charging and discharging power, paired boundaries, and coherent operating segments | Whether device operation produces an explainable response in total customer load |
| Response visibility | Charging visibility, discharging visibility, and response confidence | Whether storage response is clear or obscured by other load changes |
| Tariff behavior | Preference for lower-priced charging, higher-priced discharging, and the relative tariff levels of both actions | Whether operation is consistent with cost savings under time-of-use tariffs |

The teacher combines these signals into a continuous storage-behavior score. Peak and off-peak relationships provide behavioral evidence; operation during standard-price periods can still contribute through physical relationships and visible responses. Such operation is not automatically treated as absence of storage.

### 3. Reduce dependence on fixed clock times through policy transfer

Tariff changes move peak and off-peak periods, while the economic motive to charge cheaply and discharge at higher prices remains. Training relocates storage operating segments according to relative tariff roles and reconstructs customer active power accordingly, keeping device actions consistent with the new policy context.

The teacher's physical and response branches mask tariff context; its behavior branch retains relative tariff roles and masks the season channel. Inference generates time-period features under the selected current policy to reduce dependence on the old policy's fixed clock times.

The implementation currently contains old-policy and new-policy configurations. Additional policies require time-period configuration and validation of transfer performance.

### 4. Check evidence with paired counterfactuals

Training constructs paired counterfactuals through timing or sign mismatches, attenuation of the customer active-power response, and removal of the storage component. The model learns that corresponding evidence should decrease when physical relationships are disrupted or storage responses weaken.

These samples support continuous-target learning and pairwise ranking. They provide training references when verified non-storage ground truth is unavailable, but they are not verified non-storage labels.

### 5. Transfer teacher knowledge to a student requiring only total load

The student receives normalized customer active power, positive changes, negative changes, relative tariff roles, and tariff-role transitions. Multi-target distillation and paired constraints transfer the teacher's storage behavior, physical evidence, response confidence, activity strength, and charging and discharging visibility.

Deployment does not require storage-device measurements or consecutive multi-day records. Identification is performed for individual days before aggregation into customer-level results.

Training, validation, and test data are split by customer, with all dates for one customer assigned to a single set. Validation examines representation collapse, teacher-target fitting, counterfactual ranking, and policy-transfer stability; student evaluation also checks retention of teacher knowledge. Counterfactual discrimination supports method validation, while real-world identification performance requires verified field outcomes.

## Screening results and interpretation

The preview and compact workbook contain four columns: **customer number, storage assessment, total days, and valid days**. Detailed output also includes daily results and the proportion of valid days in each category.

| Score range | Screening category |
| --- | --- |
| Above 80 | Strong storage behavior |
| From 60 through 80, inclusive | Moderate storage behavior |
| Below 60 | No storage evidence or a low storage share |

The customer-level score is the mean of the highest-scoring 20% of valid days, selecting a count rounded upward and at least one day. This rule emphasizes evidence on active days; assessments based on few valid days should be reviewed with additional data.

Scores measure behavioral evidence and have not been calibrated as probabilities against verified non-storage customers. Strong and moderate categories support screening priorities. Lower scores may reflect weak storage, inactivity, or masking by other loads and do not establish absence of equipment. Photovoltaic generation, production starts and stops, onsite generation, and metering anomalies require manual review.

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

## Documentation

- [Data format](docs/DATA_FORMAT.en.md)
- [Training workflow](docs/TRAINING_PIPELINE.en.md)
- [Model card](docs/MODEL_CARD.en.md)
- [Local web service](storage/web/README.en.md)
