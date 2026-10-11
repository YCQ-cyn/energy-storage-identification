# 模型说明 / Model Card

## 目标与输出 / Objective and output

识别用户日负荷中的储能充放电行为，核心证据是经济利益驱动下的低电价时段充电、高电价时段放电及其在用户有功功率中的可见响应。平价时段也可能有储能动作。

Identify energy-storage charging and discharging behavior in daily customer load curves. Core evidence includes economically motivated charging during lower-priced periods, discharging during higher-priced periods, and the visible response in customer active power. Storage operation can also occur during standard-price periods.

输出为连续行为证据分数和辅助排查等级。当前分数没有经过严格非储能用户真值校准，不能作为设备存在概率或现场安装结论。

Outputs are continuous behavior-evidence scores and screening categories. The current scores are not calibrated against verified non-storage customer ground truth and should not be treated as equipment-presence probabilities or installation findings.

## 模型链路 / Model pipeline

1. 第一阶段：利用用户有功功率和储能充放电功率学习同步时序表征。 / First stage: learn synchronized temporal representations from customer active power and storage charging and discharging power.
2. 第二阶段：训练完整信息教师，目标覆盖物理证据、响应可信度和套利行为。 / Second stage: train a teacher with privileged information to predict physical evidence, response confidence, and arbitrage behavior.
3. 蒸馏准备：构建教师连续目标、政策迁移样本和反事实成对样本。 / Distillation preparation: construct continuous teacher targets, policy-transfer samples, and paired counterfactual samples.
4. 学生训练：输入归一化有功功率、正向变化量、负向变化量、相对电价角色和角色跳变。 / Student training: use normalized active power, positive changes, negative changes, relative tariff roles, and tariff-role transitions.
5. 推理汇总：产生每日分数，并依据有效日进行用户级汇总。 / Inference and aggregation: produce daily scores and aggregate valid days into customer-level assessments.

网络采用因果时间卷积网络与长短期记忆网络组合。模型输入语义和目标版本会在加载权重时校验。

The architecture combines a causal temporal convolutional network with a long short-term memory network. Input semantics and objective versions are checked when loading model checkpoints.

## 政策泛化 / Policy generalization

模型使用相对电价角色，不要求输入具体电价金额。根据数据月份和选定政策生成时段序列，教师行为分支保留电价角色，相关分支屏蔽季节或政策上下文以减少捷径。

The model uses relative tariff roles and does not require exact electricity prices. Time-period sequences are generated from the record month and selected policy. The teacher's behavior branch retains tariff roles, while relevant branches mask season or policy context to reduce shortcuts.

当前代码内置旧政策和新政策配置。新的政策需要先在 `storage/data.py` 中配置相应时段；重定位训练可降低固定时钟依赖，但不保证任意未见政策都自动适用。

The current implementation contains old-policy and new-policy configurations. Additional policies must first be defined in `storage/data.py`. Behavior-relocation training can reduce reliance on fixed clock times, but does not guarantee automatic generalization to every unseen policy.

## 汇总规则 / Aggregation rule

每日分数大于 80 为强储能，大于等于 60 且小于等于 80 为中储能，其余为无或储能占比不高。用户级评分采用有效日中最高 20% 分数的平均值，选取天数向上取整且至少为一天。

Daily scores above 80 indicate strong storage behavior; scores from 60 through 80, inclusive, indicate moderate storage behavior; remaining scores indicate no storage evidence or a low storage share. Customer-level scoring uses the mean of the highest-scoring 20% of valid days, selecting a count rounded upward and at least one day.

每日等级占比与最终用户级等级是不同的统计量；最高分日均值强调活跃储能证据，不等于储能全天数占比。

Daily category proportions and the final customer-level category are different statistics. The mean over the highest-scoring days emphasizes active-storage evidence and is not the fraction of all days with storage operation.

## 局限与使用 / Limitations and use

- 缺少严格非储能真值时，不能直接报告真实二分类准确率。 / Without verified non-storage ground truth, real-world binary-classification accuracy cannot be established.
- 光伏、生产启停、负荷转移和计量异常会混入有功功率曲线。 / Photovoltaic generation, production starts and stops, load shifting, and metering anomalies can affect active-power curves.
- 有效天数较少的用户，汇总结论稳定性较弱。 / Assessments for customers with few valid days are less stable.
- 业务阈值应结合区域、政策和现场复核结果校准。 / Business thresholds should be calibrated to the region, policy, and field-review outcomes.
- 错配伪负、储能衰减样本和生成对抗网络候选不能作为真实非储能标签。 / Mismatched pseudo-negative samples, attenuated-storage samples, and generative-adversarial-network candidates are not verified non-storage labels.

建议用于候选用户排序、辅助排查和人工复核。新政策上线后，应检查行为分数分布、有效天数与代表性样本的功率响应。

Use the model for customer prioritization, auxiliary screening, and manual review. After a policy change, inspect behavior-score distributions, valid-day counts, and representative power-response examples.
