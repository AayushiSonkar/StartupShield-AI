# StartupShield AI

StartupShield AI is an explainable machine-learning system that predicts whether a startup is more likely to be **Acquired** or **Closed** from historical startup data.

## Features

- Dataset overview with outcome and missing-value charts
- Comparison of Logistic Regression, Decision Tree, and Random Forest
- Tuned Random Forest outcome prediction with confidence score
- Rule-based startup improvement suggestions
- Bulk CSV upload and batch prediction
- SHAP-based explanation for an individual prediction
- Rule-based project assistant; no external API is used

## Technology Stack

Python, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Streamlit, SHAP, Joblib, Git, and GitHub.

## Target Variable

- `labels = 1`: Acquired
- `labels = 0`: Closed

`status` contains the text form of the target and is not used as a model feature.

## Project Structure

```text
StartupShield-AI/
├── app/app.py                         # Streamlit dashboard
├── data/raw/startup data.csv          # Historical source dataset
├── models/startup_shield_final_bundle.pkl
├── notebooks/                         # Understanding, EDA, and training notebooks
├── reports/                           # Report outline, viva questions, and sample CSV
├── src/train_model.py                 # Reproducible model-training script
├── requirements.txt
└── README.md
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Rebuild the Model

The trained bundle is included so the dashboard works immediately. To rebuild the model from the raw dataset, run:

```bash
python src/train_model.py
```

## Compatible Bulk-Prediction CSV

The bulk CSV feature requires these columns:

```text
funding_total_usd,funding_rounds,relationships,milestones
```

Optional training features such as `state_code`, `category_code`, and investor-related columns may also be supplied. Missing compatible fields use saved training-data defaults.

A ready-to-use example is available at `reports/sample_startup_input.csv`.

## Workflow

1. Load and understand the dataset.
2. Analyze missing values, outcomes, categories, funding, and milestones.
3. Remove identifiers, duplicate target fields, and leakage features.
4. Create funding and advanced-round features.
5. Compare three classification models.
6. Tune Decision Tree and Random Forest with GridSearchCV.
7. Select the best model by F1-score.
8. Explain individual predictions with SHAP.
9. Deploy the trained pipeline in Streamlit.

## Limitations

- This is an academic project using historical data.
- Predictions do not guarantee an actual startup outcome.
- The system is not investment or business advice.
- Feature associations do not prove causation.

## Final-Year Submission Support

- Report structure: `reports/REPORT_OUTLINE.md`
- Viva preparation: `reports/VIVA_QA.md`
- Suggested screenshots: Home, Dataset Overview, Model Performance, Prediction, SHAP Explainability, Bulk CSV Prediction, and AI Assistant.
