The full exploratory analysis and model development are in employee_EDA.ipynb.
The same pipeline is implemented in production form in src/train.py, src/evaluate.py, and src/predict.py.
## **Project Overview:**
# End-to-end ML flow

1. **Data ingestion**
   - Load `employee_data.csv` into a pandas DataFrame.
   - Columns include `employee_id`, `age`, `gender`, `marital_status`, `salary`, `employment_type`, `region`, `has_dependents`, `tenure_years`, and `enrolled` (target).

2. **Data quality checks**
   - Checked dataset shape and column types.
   - Verified there are no missing values in the core features or target.
   - Looked at basic statistics and distributions (age, salary, tenure, enrolled rate).

3. **Outlier and distribution analysis**
   - Explored distributions of `salary`, `age`, and `tenure_years` using histograms and boxplots.
   - Identified high-salary and long-tenure employees as natural but not erroneous outliers.
   - Chose not to drop outliers and instead rely on a tree-based model (XGBoost) that is robust to them.

4. **Feature engineering**
   From the raw columns we created:
   - `dependents_flag`: 1 if `has_dependents == "Yes"`, else 0.
   - `is_stable_employee`: 1 if `tenure_years >= 3`, else 0.
   - `is_contract`: 1 if `employment_type == "Contract"`, else 0.
   - `is_early_tenure`: 1 if `tenure_years < 1`, else 0.
   - `is_long_tenure`: 1 if `tenure_years > 5`, else 0.
   - `salary_per_tenure`: `salary / max(tenure_years, 0.1)` to capture earning intensity over tenure.

5. **Train–test split**
   - Separated `X` and `y` (`y = enrolled`).
   - Dropped `employee_id` from features.
   - Used an 80/20 train–test split with stratification on `enrolled` to preserve the class balance.

6. **Encoding & preprocessing**
   - Automatically detected numeric vs categorical columns.
   - Used `OneHotEncoder(handle_unknown="ignore")` for categorical features (`gender`, `marital_status`, `employment_type`, `region`, `has_dependents`).
   - Kept numeric features (`age`, `salary`, `tenure_years` and engineered numeric features) as-is (no scaling, since XGBoost is tree-based).

7. **Model training**
   - Base model: `XGBClassifier` (gradient boosted trees).
   - Addressed class imbalance using `scale_pos_weight = (negatives / positives)` computed from the training set.
   - Performed hyperparameter tuning with `RandomizedSearchCV` over:
     - `n_estimators`, `max_depth`, `learning_rate`
     - `subsample`, `colsample_bytree`
     - `gamma`, `reg_alpha`, `reg_lambda`
   - Selected the best model based on F1 score on cross-validation.

8. **Best model configuration**
   - Final tuned hyperparameters used in `train.py`:
     - `n_estimators = 100`
     - `max_depth = 6`
     - `learning_rate = 0.01`
     - `subsample = 1.0`
     - `colsample_bytree = 0.7`
     - `gamma = 0.1`
     - `reg_alpha = 0.1`
     - `reg_lambda = 10`
     - `scale_pos_weight = (negatives / positives)` from training data
   - Evaluation metric for training: `aucpr` and F1.

9. **Model evaluation**
   - Evaluated on the held-out test set.
   - Computed: accuracy, precision, recall, F1, ROC-AUC, and PR-AUC.
   - Inspected confusion matrix and class-wise behavior.

10. **Model persistence & serving**
    - Saved the trained model and preprocessing objects with `joblib` to `models/best_xgb_model.pkl`.
    - `src/predict.py` loads this bundle and:
      - Applies the same feature engineering and encoding.
      - Outputs `enroll_probability` (predicted probability for class 1) and `enroll_prediction` (0/1) to `outputs/preds.csv`.

### Data observations

- The dataset contains 10,000 employees with 10 columns, including demographic (`age`, `gender`, `marital_status`, `region`) and employment-related features (`salary`, `employment_type`, `has_dependents`, `tenure_years`) plus an `enrolled` target.
- The target `enrolled` is binary and moderately imbalanced, with around 62% enrolled and 38% not enrolled. This suggests the positive class is more common but the negative class is still meaningful.
- There were no missing values in the provided dataset, so no imputation was necessary.
- Numerical features such as `salary` and `tenure_years` show reasonable ranges but with some high-value outliers (e.g., higher salaries and longer tenures). These look like realistic business cases rather than data errors.
- Categorical variables (`gender`, `marital_status`, `employment_type`, `region`, `has_dependents`) have a small number of levels and show intuitive distributions (e.g., mix of full-time/part-time, multiple regions).

### Evaluation results

On the held-out 20% test set:

- Accuracy: **1.000**
- Precision (positive class): **0.999**
- Recall (positive class): **1.000**
- F1 score: **1.000**
- ROC-AUC: **0.999**

Because the dataset is synthetic and well‑behaved, the tuned XGBoost model almost perfectly separates enrolled vs non‑enrolled employees on the held‑out test set (F1 ≈ 1.0, ROC‑AUC ≈ 0.999).

These metrics indicate that the model captures the majority of enrolled employees while keeping a relatively low false positive rate. In particular, the F1 score shows a good balance between precision and recall, which aligns with the business goal of prioritizing employees likely to opt in while not spamming everyone.

### Key takeaways and next steps

- Simple, well-engineered features (dependents, tenure, contract status, salary per tenure) combined with a gradient-boosted tree model already provide strong predictive performance on this synthetic dataset.
- The class imbalance is manageable but does impact performance; using `scale_pos_weight` and F1 as the tuning metric helps avoid a trivial “predict everyone enrolls” solution.
- The model is easily productionized: the same feature engineering and preprocessing are encapsulated in `train.py` / `predict.py`, and predictions are exported as probabilities plus hard labels.

With more time, I would:

- **Improve calibration:** Evaluate and, if needed, apply probability calibration (e.g., Platt scaling or isotonic regression) so that `enroll_probability` is better aligned with actual enrollment rates.
- **Richer feature engineering:** Add interactions (e.g., region × employment_type), non-linear transformations of salary/tenure, and possibly external features (e.g., company-level enrollment history if available).
- **Experiment tracking:** Integrate MLflow or Weights & Biases to log experiments, hyperparameters, and metrics, making the tuning process more systematic and reproducible.
- **Model serving API:** Wrap the prediction pipeline in a simple REST API using FastAPI (load the `best_xgb_model.pkl` bundle in memory and expose a `/predict` endpoint that accepts JSON).
- **Monitoring and fairness checks:** Once deployed, track performance over time by segment (region, employment_type, gender) to detect drift or unintended bias and retrain as needed.