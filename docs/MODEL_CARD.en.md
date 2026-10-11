# Model Card

[简体中文](MODEL_CARD.md) | **English**

## Objective and output

Identify energy-storage charging and discharging behavior in daily customer load curves. Core evidence includes economically motivated charging during lower-priced periods, discharging during higher-priced periods, and the visible response in customer active power. Storage operation can also occur during standard-price periods.

Outputs are continuous behavior-evidence scores and screening categories. The current scores are not calibrated against verified non-storage customer ground truth and should not be treated as equipment-presence probabilities or installation findings.

## Model pipeline

1. First stage: learn synchronized temporal representations from customer active power and storage charging and discharging power.
2. Second stage: train a teacher with privileged information to predict physical evidence, response confidence, and arbitrage behavior.
3. Distillation preparation: construct continuous teacher targets, policy-transfer samples, and paired counterfactual samples.
4. Student training: use normalized active power, positive changes, negative changes, relative tariff roles, and tariff-role transitions.
5. Inference and aggregation: produce daily scores and aggregate valid days into customer-level assessments.

The architecture combines a causal temporal convolutional network with a long short-term memory network. Input semantics and objective versions are checked when loading model checkpoints.

## Policy generalization

The model uses relative tariff roles and does not require exact electricity prices. Time-period sequences are generated from the record month and selected policy. The teacher's behavior branch retains tariff roles, while relevant branches mask season or policy context to reduce shortcuts.

The current implementation contains old-policy and new-policy configurations. Additional policies must first be defined in `storage/data.py`. Behavior-relocation training can reduce reliance on fixed clock times, but does not guarantee automatic generalization to every unseen policy.

## Aggregation rule

Daily scores above 80 indicate strong storage behavior; scores from 60 through 80, inclusive, indicate moderate storage behavior; remaining scores indicate no storage evidence or a low storage share. Customer-level scoring uses the mean of the highest-scoring 20% of valid days, selecting a count rounded upward and at least one day.

Daily category proportions and the final customer-level category are different statistics. The mean over the highest-scoring days emphasizes active-storage evidence and is not the fraction of all days with storage operation.

## Limitations and use

- Without verified non-storage ground truth, real-world binary-classification accuracy cannot be established.
- Photovoltaic generation, production starts and stops, load shifting, and metering anomalies can affect active-power curves.
- Assessments for customers with few valid days are less stable.
- Business thresholds should be calibrated to the region, policy, and field-review outcomes.
- Mismatched pseudo-negative samples, attenuated-storage samples, and generative-adversarial-network candidates are not verified non-storage labels.

Use the model for customer prioritization, auxiliary screening, and manual review. After a policy change, inspect behavior-score distributions, valid-day counts, and representative power-response examples.
