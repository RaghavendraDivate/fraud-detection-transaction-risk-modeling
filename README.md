````markdown
# Fraud Detection & Transaction Risk Modeling Pipeline

An end-to-end machine learning pipeline for detecting potentially fraudulent
transactions, ranking transaction risk, and prioritizing limited fraud
investigation capacity.

This project uses the IEEE-CIS Fraud Detection dataset and focuses on
leakage-aware feature engineering, imbalanced classification, temporal
validation, model comparison, risk scoring, explainability, and
business-oriented evaluation.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Project Objectives](#project-objectives)
- [Project Pipeline](#project-pipeline)
- [Data Understanding & EDA](#data-understanding--eda)
- [Hypothesis Testing](#hypothesis-testing)
- [Feature Engineering](#feature-engineering)
- [Leakage Prevention](#leakage-prevention)
- [Temporal Validation](#temporal-validation)
- [Preprocessing](#preprocessing)
- [Model Development](#model-development)
- [Model Comparison](#model-comparison)
- [Champion Model](#champion-model)
- [Risk Scoring](#risk-scoring)
- [Investigation Capacity Analysis](#investigation-capacity-analysis)
- [Cost-Sensitive Threshold Analysis](#cost-sensitive-threshold-analysis)
- [Explainability](#explainability)
- [Key Findings](#key-findings)
- [Business Recommendation](#business-recommendation)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Reproducibility](#reproducibility)
- [Dataset Availability](#dataset-availability)

---

# Project Overview

Fraud detection is an imbalanced classification problem in which fraudulent
transactions represent only a small proportion of the overall transaction
population.

A practical fraud detection system should therefore do more than maximize
classification accuracy.

It should identify high-risk transactions, rank them effectively, and allow
limited investigation resources to be directed toward transactions with the
highest observed fraud concentration.

This project develops an end-to-end transaction-risk modeling pipeline:

```text
Raw IEEE-CIS Data
        |
        v
Data Understanding
        |
        v
Exploratory Data Analysis
        |
        v
Hypothesis Testing
        |
        v
Feature Engineering
        |
        v
Leakage Prevention
        |
        v
Temporal Validation
        |
        v
Preprocessing
        |
        v
Model Development
        |
        v
Model Comparison
        |
        v
Champion Model Selection
        |
        v
Risk Scoring
        |
        v
Explainability
        |
        v
Business Impact Analysis
        |
        v
Final Recommendation
````

The final system is evaluated primarily as a **transaction-risk ranking
system** rather than as a simple binary classifier.

---

# Business Problem

Financial institutions process large volumes of transactions, making it
impractical to manually investigate every transaction.

The objective is to use machine learning to prioritize transactions for
investigation while balancing:

* Fraud detection
* False-positive investigations
* Investigation capacity
* Customer friction
* Fraud-loss severity

### Business Question

> Can a machine learning model effectively rank transactions so that a small
> investigation queue contains a substantially higher concentration of
> fraudulent transactions than the overall transaction population?

---

# Dataset

This project uses the **IEEE-CIS Fraud Detection** dataset.

The dataset contains transaction-level information and additional identity
information for a subset of transactions.

## Training Data

| Dataset          |    Rows | Columns |
| ---------------- | ------: | ------: |
| Transaction data | 590,540 |     394 |
| Identity data    | 144,233 |      41 |

The transaction and identity datasets are joined using `TransactionID`.

After joining:

* Transactions: 590,540
* Combined columns: 434
* Transactions with identity information: 144,233
* Transactions without identity information: 446,307

Identity information is therefore available for approximately 24.4% of
transactions.

## Target Variable

The target variable is `isFraud`.

| Value | Meaning                |
| ----- | ---------------------- |
| `0`   | Legitimate transaction |
| `1`   | Fraudulent transaction |

### Target Distribution

| Class      | Transactions | Percentage |
| ---------- | -----------: | ---------: |
| Legitimate |      569,877 |     96.50% |
| Fraudulent |       20,663 |      3.50% |

The strong class imbalance makes accuracy a poor primary evaluation metric.

---

# Dataset Considerations

The IEEE-CIS dataset contains a large number of anonymized variables,
particularly the `V*` feature family.

The project does not assign unsupported business meanings to anonymized
variables.

Instead, these variables are treated as statistical predictors and evaluated
using predictive performance and explainability techniques.

`TransactionID` is used for joining and traceability but is excluded from model
inputs because it identifies a transaction rather than describing its
characteristics.

`TransactionDT` represents relative transaction time rather than a standard
calendar timestamp.

---

# Project Objectives

The project aims to:

1. Understand the structure and quality of the transaction and identity data
2. Investigate fraud patterns through exploratory data analysis
3. Apply hypothesis testing to selected variables
4. Engineer leakage-aware transaction features
5. Handle substantial missingness
6. Handle high-cardinality categorical variables
7. Preserve chronological ordering during model validation
8. Compare multiple classification approaches
9. Select a practical champion model
10. Generate transaction-level risk scores
11. Evaluate fraud concentration under different investigation capacities
12. Analyze model behavior using multiple explainability approaches
13. Translate model results into business-oriented recommendations

---

# Project Pipeline

The project is organized into six major analytical stages.

### 01. Data Understanding

* Dataset profiling
* Data types
* Missingness
* Feature families
* Target distribution
* Identifier analysis
* Temporal structure

### 02. Exploratory Data Analysis

* Fraud distribution
* Transaction amount
* Temporal patterns
* Identity availability
* Device characteristics
* Categorical variables
* Missingness patterns

### 03. Statistical Analysis

* Chi-square tests
* Mann–Whitney U test
* Cramér's V
* Rank-biserial correlation

### 04. Feature Engineering

* Time features
* Log transaction amount
* Identity availability
* Missingness features
* Frequency encoding
* Rare-category grouping
* Feature quality filtering

### 05. Machine Learning

* Logistic Regression
* Random Forest
* XGBoost
* LightGBM
* Temporal validation
* Imbalanced classification
* Model comparison

### 06. Risk Evaluation

* Risk-score distributions
* Risk bands
* Investigation-capacity analysis
* Cost-sensitive threshold analysis
* Explainability
* Business impact
* Final recommendation

---

# Data Understanding & EDA

## Transaction Amount

`TransactionAmt` is highly right-skewed.

Key statistics:

| Statistic |      Value |
| --------- | ---------: |
| Mean      |     135.03 |
| Median    |      68.77 |
| Minimum   |      0.251 |
| Maximum   | 31,937.391 |

A log-transformed version of transaction amount was created:

```text
log_TransactionAmt = log(1 + TransactionAmt)
```

Both the original and transformed representations were retained.

Fraud rates across transaction-amount ranges were not monotonic, so the
project did not use a simple amount-based fraud threshold.

---

## Identity Availability

Identity information is present for approximately 24.4% of transactions.

Observed fraud rates:

| Identity Information | Fraud Rate |
| -------------------- | ---------: |
| Not available        |      2.09% |
| Available            |      7.85% |

Transactions with identity information showed a substantially higher observed
fraud rate.

This was treated as an **association**, not a causal relationship.

A `has_identity` feature was therefore created.

---

## Temporal Analysis

`TransactionDT` covers approximately 182 days of relative transaction time.

Derived temporal variables include:

* Relative day
* Relative week
* Relative hour
* Relative day index

Fraud prevalence varies across the observed time period, supporting the use
of chronological validation.

The derived day index is treated as a relative index and is not interpreted
as a real-world weekday.

---

## Missingness

The dataset contains substantial missingness.

Several features have extremely high missing-value proportions.

Rather than automatically deleting highly sparse variables, the project
investigated missingness as a potential predictive signal.

Aggregate missingness features were therefore created:

* `missing_count`
* `missing_ratio`
* `identity_missing_count`
* `has_identity`

---

# Hypothesis Testing

Hypothesis testing was used as an exploratory analytical tool rather than as
the sole mechanism for feature selection.

The following relationships were evaluated:

| Variable                        | Test           |
| ------------------------------- | -------------- |
| `id_35` vs `isFraud`            | Chi-square     |
| `card6` vs `isFraud`            | Chi-square     |
| `ProductCD` vs `isFraud`        | Chi-square     |
| `TransactionAmt` vs `isFraud`   | Mann–Whitney U |
| Relative time week vs `isFraud` | Chi-square     |

## Results

| Variable           |      Test Statistic |            Effect Size | Decision          |
| ------------------ | ------------------: | ---------------------: | ----------------- |
| `id_35`            |        χ² = 2888.47 |    Cramér's V = 0.1431 | Reject H₀         |
| `card6`            |        χ² = 5957.03 |    Cramér's V = 0.1006 | Reject H₀         |
| `ProductCD`        |       χ² = 16742.17 |    Cramér's V = 0.1684 | Reject H₀         |
| `TransactionAmt`   | U = 5,858,540,820.5 | Rank-biserial ≈ -0.005 | Fail to reject H₀ |
| Relative time week |         χ² = 918.16 |    Cramér's V = 0.0394 | Reject H₀         |

The categorical variables showed statistically significant associations with
fraud.

Transaction amount did not show a statistically significant difference in
its overall distribution at the selected significance level.

Relative time showed statistically significant variation, but its effect size
was small.

Statistical significance was not used as a substitute for predictive
validation.

---

# Feature Engineering

Feature engineering was designed around the following prediction scenario:

> A new transaction arrives, and the system estimates its fraud risk using
> information available at or before the transaction.

## Engineered Features

The final feature pipeline includes:

### Temporal Features

* Relative transaction day
* Relative transaction week
* Relative transaction hour
* Relative day index

### Transaction Features

* Original transaction amount
* Log-transformed transaction amount

### Identity Features

* Identity availability
* Identity missing-value count

### Missingness Features

* Overall missing-value count
* Overall missing-value ratio

### Categorical Features

* One-hot encoded low-cardinality variables
* Rare-grouped `id_30`
* Rare-grouped `id_31`
* Frequency-encoded `DeviceInfo`
* Frequency-encoded `id_33`

---

# Feature Quality

The initial feature inventory contained 441 candidate model features.

After removing clear redundancy and near-constant features, the final model
input contained:

```text
433 model inputs
404 numerical inputs
29 categorical inputs
```

A perfect redundancy was identified between `D4` and `D12`.

`D12` was removed while retaining `D4`.

Several near-constant variables were also removed:

```text
V1
V14
V41
V65
V88
V107
V305
```

These decisions were made before model fitting.

---

# Leakage Prevention

Leakage prevention is a core requirement of the project.

The model should only use information that would be available at the
transaction prediction point.

## Controls Applied

* `isFraud` excluded from model inputs
* `TransactionID` excluded from model inputs
* EDA-only helper variables excluded
* `D12` removed due to perfect redundancy
* Near-constant features removed
* Frequency encoding fitted on training data only
* Rare-category rules fitted on training data only
* Numerical imputation fitted on training data only
* Categorical encoding fitted on training data only
* Validation data never used to fit preprocessing parameters

No full-dataset target encoding was used.

Historical fraud-rate features were also avoided because they require careful
time-aware construction to prevent future information from entering the
training process.

---

# Temporal Validation

Fraud patterns can change over time.

A chronological 80/20 split was therefore used instead of a random split.

## Training Period

```text
472,432 transactions
```

## Validation Period

```text
118,108 transactions
```

The model was trained on earlier transactions and evaluated on later
transactions.

| Dataset    | Fraud Rate |
| ---------- | ---------: |
| Training   |      3.51% |
| Validation |      3.44% |

The similar fraud prevalence provides a stable basis for model evaluation
while preserving chronological ordering.

---

# Preprocessing

The final preprocessing pipeline contained:

```text
433 raw model inputs
        ↓
404 numerical inputs
29 categorical inputs
        ↓
Median imputation
        +
Categorical missing-value handling
        +
Frequency encoding
        +
Rare-category grouping
        +
One-hot encoding
        +
Scaling
        ↓
768 transformed features
```

Sparse CSR matrices were used to manage memory efficiently.

Numerical imputation and scaling were fitted on the training period only.

Categorical encoding was also fitted using training data only, with unseen
validation categories handled safely.

---

# Model Development

Four classification models were evaluated.

### Logistic Regression

Used as the linear baseline.

Configuration included:

* Balanced class weighting
* `saga` solver
* Regularization
* Maximum 100 iterations

The model reached the iteration limit without full convergence, but remained
usable as a baseline.

### Random Forest

Used as a nonlinear tree-based challenger.

### XGBoost

Used as the primary gradient-boosting model.

The model used:

* Histogram-based tree construction
* Class weighting
* Subsampling
* Feature subsampling
* Controlled tree depth
* Learning-rate regularization

### LightGBM

Used as a second gradient-boosting challenger.

The model used:

* Histogram-based learning
* Class weighting
* Controlled tree depth
* Feature and row subsampling

---

# Evaluation Metrics

Because fraud represents only approximately 3.5% of transactions, accuracy was
not treated as the primary evaluation metric.

The main metrics were:

* **PR-AUC** — important for imbalanced classification
* **ROC-AUC** — overall ranking discrimination
* **Precision** — proportion of flagged transactions that are fraudulent
* **Recall** — proportion of observed fraud captured
* **F1-score** — balance between precision and recall
* **Precision@K** — fraud concentration within a fixed investigation capacity
* **Recall@K** — fraud captured within a fixed investigation capacity

---

# Model Comparison

| Model               | PR-AUC | ROC-AUC | Precision @ 0.5 | Recall @ 0.5 | F1 @ 0.5 |
| ------------------- | -----: | ------: | --------------: | -----------: | -------: |
| Logistic Regression | 0.1808 |  0.8336 |           9.37% |       77.78% |   16.73% |
| Random Forest       | 0.4528 |  0.8709 |          19.21% |       65.67% |   29.72% |
| XGBoost             | 0.5099 |  0.9041 |          21.49% |       74.53% |   33.36% |
| LightGBM            | 0.5106 |  0.9063 |          19.49% |       76.33% |   31.05% |

LightGBM achieved the highest global PR-AUC and ROC-AUC.

However, the difference between LightGBM and XGBoost was small.

Operational ranking performance was therefore also considered.

---

# Champion Model

## XGBoost

XGBoost was selected as the champion model.

The selection considered both:

1. Global discrimination performance
2. Operational performance under investigation-capacity constraints

XGBoost achieved slightly better Top 1%, Top 5%, and Top 10% ranking
performance than LightGBM.

The two gradient-boosting models were therefore considered close in overall
performance, with XGBoost providing the stronger operational ranking result
for the selected use case.

---

# Risk Scoring

The champion model produces a continuous risk score for every transaction.

These scores are primarily interpreted as **ranking signals**, rather than
calibrated probabilities, because class weighting was used during model
training.

## Risk Score Distribution

| Statistic       | Legitimate |  Fraud |
| --------------- | ---------: | -----: |
| Mean            |     0.2162 | 0.6986 |
| Median          |     0.1548 | 0.7698 |
| 75th percentile |     0.2812 | 0.9674 |
| 90th percentile |     0.4932 | 0.9941 |
| 95th percentile |     0.6419 | 0.9970 |

Fraudulent transactions are strongly shifted toward higher risk scores.

This indicates meaningful separation between legitimate and fraudulent
transactions.

---

# Risk Band Analysis

The risk scores were divided into bands to examine fraud concentration.

| Risk Band | Transactions | Fraudulent | Fraud Rate |
| --------- | -----------: | ---------: | ---------: |
| 0.0–0.1   |       36,385 |         92 |      0.25% |
| 0.1–0.2   |       33,960 |        196 |      0.58% |
| 0.2–0.3   |       18,141 |        250 |      1.38% |
| 0.3–0.4   |        9,425 |        246 |      2.61% |
| 0.4–0.5   |        6,099 |        251 |      4.12% |
| 0.5–0.6   |        4,313 |        312 |      7.23% |
| 0.6–0.7   |        3,439 |        377 |     10.96% |
| 0.7–0.8   |        2,536 |        428 |     16.88% |
| 0.8–0.9   |        1,803 |        456 |     25.29% |
| 0.9–1.0   |        2,007 |      1,456 |     72.55% |

The 0.9–1.0 risk band contains a very high concentration of fraudulent
transactions compared with the overall validation fraud rate of 3.44%.

---

# Investigation Capacity Analysis

A practical fraud system may have a limited number of transactions that can
be investigated manually.

Therefore, transactions were ranked by XGBoost risk score and evaluated at
different investigation capacities.

| Investigation Capacity | Transactions Reviewed | Fraud Found | Precision | Fraud Capture |
| ---------------------- | --------------------: | ----------: | --------: | ------------: |
| Top 1%                 |                 1,182 |       1,031 |    87.23% |        25.37% |
| Top 2%                 |                 2,363 |       1,565 |    66.23% |        38.51% |
| Top 5%                 |                 5,906 |       2,274 |    38.50% |        55.95% |
| Top 10%                |                11,811 |       2,865 |    24.26% |        70.50% |
| Top 20%                |                23,622 |       3,385 |    14.33% |        83.29% |

The overall validation fraud rate was approximately 3.44%.

### Key operational result

At the **Top 1%** investigation capacity:

* 1,182 transactions are reviewed
* 1,031 are fraudulent
* Precision is approximately 87%
* Approximately 25% of observed fraud is captured

At the **Top 5%** capacity:

* 5,906 transactions are reviewed
* 2,274 are fraudulent
* Precision is approximately 39%
* Approximately 56% of observed fraud is captured

At the **Top 10%** capacity:

* 11,811 transactions are reviewed
* 2,865 are fraudulent
* Precision is approximately 24%
* Approximately 71% of observed fraud is captured

This demonstrates the trade-off between investigation volume and fraud
capture.

---

# Cost-Sensitive Threshold Analysis

A classification threshold should not automatically be fixed at 0.5.

Threshold selection depends on the relative business cost of false positives
and false negatives.

Illustrative relative-cost scenarios were therefore evaluated.

These are **not actual financial costs**.

| Scenario               | FP Cost | FN Cost | Lowest Relative-Cost Threshold |
| ---------------------- | ------: | ------: | -----------------------------: |
| Balanced               |       1 |       1 |                           0.95 |
| Fraud-sensitive        |       1 |       5 |                           0.75 |
| Highly fraud-sensitive |       1 |      10 |                           0.60 |

The analysis demonstrates that increasing the relative cost of missed fraud
moves the preferred operating point toward a lower threshold.

However, a production threshold would also require:

* Investigation capacity
* Fraud-loss estimates
* Investigation cost
* Customer-friction estimates
* Regulatory constraints
* Business risk tolerance

Therefore, the thresholds above are analytical examples rather than
production recommendations.

---

# Explainability

Multiple complementary explainability techniques were used.

## Global Feature Importance

The strongest global features included:

* `V264`
* `V258`
* `V218`
* `V91`
* `V70`
* `V295`
* `V294`
* `C4`
* `C8`
* `V201`

Missingness-related features and `ProductCD` also appeared among important
model inputs.

Because many variables are anonymized, no unsupported business meaning was
assigned to these features.

---

## Permutation Importance

Permutation importance was evaluated using PR-AUC.

Selected results:

| Feature | Mean PR-AUC Drop |
| ------- | ---------------: |
| `V258`  |         0.031544 |
| `C4`    |         0.012022 |
| `C8`    |         0.009698 |
| `V201`  |         0.007939 |
| `V257`  |         0.007629 |
| `V294`  |         0.005251 |
| `V264`  |         0.002723 |

`V258` showed particularly strong importance in the permutation analysis.

Small negative permutation-importance values were not automatically treated
as evidence that a feature should be removed, because permutation importance
can vary due to sampling and feature interactions.

---

## XGBoost Contribution Analysis

XGBoost contribution values were analyzed using a validation sample.

Important features by mean absolute contribution included:

* `C14`
* `V258`
* `TransactionAmt`
* `C13`
* `card6_debit`
* `C1`
* `C11`
* `V70`
* `V294`
* `card1`

The three approaches provide different perspectives:

```text
Global Feature Importance
        |
        v
Which features the model uses heavily

Permutation Importance
        |
        v
How model performance changes when a feature is disrupted

Contribution Analysis
        |
        v
How strongly features contribute to predictions
```

Using multiple perspectives provides stronger evidence than relying on a
single feature-importance measure.

---

# Key Findings

## 1. Fraud is strongly concentrated in high-risk transactions

Fraudulent transactions receive substantially higher model risk scores than
legitimate transactions.

The highest risk band, 0.9–1.0, has a fraud rate of approximately 72.55%.

---

## 2. The model is particularly useful as a ranking system

The strongest operational result comes from ranking transactions according to
risk rather than relying on a single arbitrary classification threshold.

The highest-risk 1% achieved approximately 87% precision.

---

## 3. Investigation capacity determines the operating point

There is no universally optimal investigation threshold.

A smaller investigation queue provides higher precision, while a larger queue
captures more total fraud at the cost of investigating more legitimate
transactions.

---

## 4. Missingness contains predictive information

The dataset contains extensive missingness.

Missingness-related features also appeared in the model's important predictors,
supporting the decision to preserve missingness information rather than
automatically deleting sparse variables.

---

## 5. Anonymized variables can still provide predictive signal

Several anonymized `V*` variables were among the strongest predictors.

Their predictive value can be established statistically, but their underlying
business meanings cannot be reliably inferred from the anonymized dataset.

---

## 6. Global model performance and operational performance can differ

LightGBM achieved slightly higher global PR-AUC and ROC-AUC.

XGBoost performed slightly better at the Top 1%, Top 5%, and Top 10%
investigation capacities.

This resulted in XGBoost being selected as the practical champion model.

---

# Business Recommendation

The champion XGBoost model should conceptually be used as a
**transaction-risk ranking system**.

A potential operational framework is:

### High-Risk Queue

Prioritize the highest-risk transactions for immediate investigation.

The Top 1% produced approximately 87% precision in the validation period.

### Extended Investigation Queue

Increase the investigation queue toward approximately the Top 5% when
additional resources are available.

This captured approximately 56% of observed fraud with approximately 39%
precision.

### Broader Monitoring

A larger risk range, such as the Top 10–20%, may be appropriate for broader
monitoring when investigation capacity allows.

These percentages are analytical results from the validation period and should
not be treated as production policies.

The final operating point should be determined using real fraud-loss costs,
investigation capacity, customer-impact considerations, and business
requirements.

---

# Limitations

## Validation Set Reuse

The chronological validation period was used for:

* Model comparison
* Investigation-capacity analysis
* Threshold analysis
* Business impact evaluation

It should therefore not be considered a completely untouched final test set.

A separate future holdout would be required for an unbiased final performance
estimate before production deployment.

---

## Anonymized Features

Many features have anonymized names.

This limits direct business interpretation of individual predictors.

---

## Illustrative Cost Analysis

Actual fraud-loss values, investigation costs, chargebacks, and customer
friction costs were not available.

Relative costs were therefore used only to demonstrate the threshold-selection
framework.

---

## Model Calibration

Class weighting was used during model training.

The resulting model scores are therefore treated primarily as ranking signals
rather than calibrated probabilities.

---

## Dataset Generalization

The IEEE-CIS dataset represents a particular historical fraud-detection
environment.

Performance may differ when the model is applied to:

* Different institutions
* Different customer populations
* Different payment systems
* Different fraud patterns
* Different time periods

---

# Future Improvements

Potential improvements include:

* Evaluate on a completely untouched future holdout
* Apply probability calibration
* Use more robust temporal cross-validation
* Incorporate institution-specific fraud-loss estimates
* Optimize investigation capacity using real operational constraints
* Develop historical behavioral features using strictly past information
* Monitor model drift
* Monitor feature stability
* Monitor threshold performance over time
* Monitor false-positive customer impact
* Develop real-time scoring infrastructure
* Deploy the model as a production fraud-risk service

---

# Project Structure

```text
fraud-detection-transaction-risk-modeling/
│
├── data/
│   ├── raw/
│   │   └── IEEE-CIS dataset files
│   │
│   └── processed/
│       ├── train_features.csv
│       ├── validation_features.csv
│       ├── champion_risk_scores.csv
│       ├── xgboost_feature_importance.csv
│       ├── xgboost_permutation_importance.csv
│       └── xgboost_shap_importance.csv
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_hypothesis_testing.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_modelling.ipynb
│   └── 06_risk_evaluation.ipynb
│
├── reports/
│   └── figures/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Technologies Used

### Programming

* Python

### Data Analysis

* Pandas
* NumPy
* Matplotlib

### Statistical Analysis

* SciPy
* Chi-square testing
* Mann–Whitney U testing
* Cramér's V
* Rank-biserial correlation

### Machine Learning

* Scikit-learn
* Logistic Regression
* Random Forest
* XGBoost
* LightGBM

### Explainability

* XGBoost feature importance
* Permutation importance
* XGBoost contribution values

### Development

* Jupyter Notebook
* VS Code
* Git
* GitHub

---

# Reproducibility

The project is organized into sequential notebooks representing the major
stages of the analytical workflow.

```text
01 → Data Understanding
02 → Exploratory Data Analysis
03 → Hypothesis Testing
04 → Feature Engineering
05 → Model Development
06 → Risk Evaluation
```

The overall workflow is:

```text
Raw IEEE-CIS Data
        ↓
Data Understanding
        ↓
EDA
        ↓
Hypothesis Testing
        ↓
Feature Engineering
        ↓
Temporal Train/Validation Split
        ↓
Training-Fitted Preprocessing
        ↓
Model Development
        ↓
Model Comparison
        ↓
Champion XGBoost
        ↓
Risk Scores
        ↓
Explainability
        ↓
Business Impact Analysis
```

The feature-engineering stage generates the processed training and validation
datasets used by the modeling stage.

The modeling stage generates champion risk scores and explainability
artifacts used by the risk-evaluation stage.

---

# Dataset Availability

This project uses the IEEE-CIS Fraud Detection dataset.

The raw dataset is not included in this repository because of its large size
and applicable dataset usage and distribution restrictions.

The dataset should be obtained through the appropriate official competition
source.

After obtaining the dataset, place the required files inside:

```text
data/raw/
```

Expected files:

```text
train_transaction.csv
train_identity.csv
test_transaction.csv
test_identity.csv
```

The raw dataset should not be committed to GitHub.

---

# Disclaimer

This project is an educational and portfolio implementation of a fraud-risk
modeling workflow.

The reported results are based on the IEEE-CIS Fraud Detection dataset and
should not be interpreted as production performance for any financial
institution.

The operating thresholds and relative cost scenarios presented in this
project are analytical examples.

A production fraud-detection system would require institution-specific data,
cost structures, validation procedures, monitoring, governance, security, and
operational constraints.

