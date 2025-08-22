## Violent Behavior Predictor

Open-source implementation for hierarchical modeling and late fusion to predict violent behavior among inpatients.

> Data notice: This repository includes a pre-generated synthetic dataset `merged_patient_data.csv` with fully fake records for demo/testing. It contains no real patient data or PII/PHI. Do not use it for clinical decisions.

### Features
- Preprocessing (column encoding), ordinal mapping for education, duration parsing
- Univariate statistics in the training set (Mann-Whitney U / Chi-square / Fisher)
- Separate logistic regression for static and behavioral features (statsmodels)
- Probability-level weighted late fusion and optional score calibration
- Metrics: AUC, Precision, Recall, F1, Confusion Matrix; ROC (can be plotted by users if needed)
- Extensible sklearn Pipeline + GridSearch (you can extend this package directly)

### Quickstart
1. Install dependencies (see `requirements.txt`)
2. Prepare data:
   - Option A: generate synthetic data: `python generate_synthetic_dataset.py`
   - Option B: place your own `merged_patient_data.csv` in project root with target `high_risk_group`
3. Run CLI (minimal example):
```bash
python cli.py --input merged_patient_data.csv --out hierarchical_model_results --ml_suite
```

### Installation (conda example)
```bash
# 1) Create environment (Python 3.10 recommended)
conda create -n vbp_env python=3.10 -y
conda activate vbp_env

# 2) Install requirements
pip install -r requirements.txt

# (Optional) Install xgboost if you want the XGBoost baseline
pip install xgboost
```

### Structure
- `violent_behavior_predictor/`
  - `constants.py`: feature lists, labels and mappings
  - `preprocessing.py`: data encoding and subset selection
  - `stats_analysis.py`: univariate statistics
  - `logit_model.py`: statsmodels logistic regression wrapper
  - `evaluation.py`: metric computation and probability calibration
  - `fusion.py`: probability fusion and alpha search
  - `ml_workflow.py`: sklearn Pipelines and grid search (optional extensions)
- `cli.py`: command-line entrypoint
  

### Reproducibility and notes
- Split uses `train_test_split(..., stratify=y)` unless `patient_id` is present, then group-wise split.
- We only drop NA on used columns; you can switch to imputation in `preprocessing.py` if preferred.

- Fusion weight can be provided via `--weight_static`, or selected via built-in grid search.

### Dataset schema
- Required target column:
  - `high_risk_group` (0/1)
- Optional grouping column:
  - `patient_id` (ensures patient-level split if present)
- Static features:
  - Continuous: `age`, `disease_duration_years`
  - Binary (expected string values shown in parentheses):
    - `gender` (male/female), `employment` (employed/unemployed), `marital_status` (married/unmarried),
      `personality` (extroverted/introverted), `substance_abuse` (yes/no),
      `history_of_violence_or_suicide` (yes/no), `high_risk_command_hallucinations` (yes/no),
      `persecutory_delusions` (yes/no), `thought_activity` (abnormal/normal), `sensation_and_perception` (abnormal/normal),
      `intelligence` (abnormal/normal), `attention` (abnormal/normal), `memory` (abnormal/normal),
      `hopelessness_or_depression` (yes/no), `mania` (yes/no)
  - Ordinal: `education_level` with values mapped as
    - `no_schooling_or_elementary` → 0, `junior_high` → 1, `high_school_or_vocational` → 2, `college_or_higher` → 3
- Dynamic behavioral features (39 items, ordinal Integer 0–3):
  - `Rule Compliance`, `Personal Affairs Management`, `Bed Making`, `Cleaning`, `Clothing Adjustment`,
    `Physical Discomfort Description`, `Work Therapy Participation`, `Attitude Towards Others`, `Interest in Surroundings`,
    `Conversation with Others`, `Family Concern`, `Discussing Personal Interests`, `Laughs at Jokes`, `Recreational Activities`,
    `Exercise Participation`, `Neat Appearance`, `Face Washing`, `Teeth Brushing`, `Personal Hygiene`, `Foot Washing`,
    `Hand Washing Before Meals`, `Eating`, `Hair Grooming`, `Anger Expression`, `Rapid Speech`, `Agitation`,
    `Cooperation with Staff`, `Talking to Self`, `Inappropriate Laughing`, `Auditory Hallucinations`, `Immobility`,
    `Lying Down`, `Psychomotor Retardation`, `Insomnia`, `Crying`, `Self-reported Depression`, `Negative Self-Evaluation`,
    `Illness Awareness`, `Discharge Request`

### Outputs
Running the CLI writes artifacts to `--out` (default `hierarchical_model_results/`):
- `results_summary.json`: consolidated metrics and best fusion weights (AUC, sensitivity, specificity, PPV, NPV, accuracy, F1, confusion_matrix)
- `static_logit_coefficients.csv` and `static_logit_summary.txt`: multivariable statsmodels Logit details for static features
- `behavior_logit_coefficients.csv` and `behavior_logit_summary.txt`: the same for behavioral features
- `ml_metrics_static.csv`, `ml_metrics_behavior.csv` (when `--ml_suite` is on): baseline model metrics

### Programmatic usage (minimal)
```python
import pandas as pd
from violent_behavior_predictor.preprocessing import encode_dataframe, select_feature_frames, zscore_normalize_continuous

df = pd.read_csv("merged_patient_data.csv")
df = encode_dataframe(df)
X_static, X_behavior, X_all, y = select_feature_frames(df)
X_train, X_test, _ = zscore_normalize_continuous(X_all.iloc[:200], X_all.iloc[200:])
# continue with your own modeling...
```

### License
MIT License (see `LICENSE`).


