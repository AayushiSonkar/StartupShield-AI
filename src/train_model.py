"""Rebuild the StartupShield AI model bundle from the raw dataset.

Run from the project root:
    python src/train_model.py
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from data_preprocessing import build_features
from evaluation import comparison_table, evaluate_classifier
from model_training import build_preprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "startup data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "startup_shield_final_model.pkl"
BUNDLE_PATH = PROJECT_ROOT / "models" / "startup_shield_final_bundle.pkl"


def make_pipeline(preprocessor, classifier):
    """Combine preprocessing and a classifier in a leakage-safe pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("model", classifier),
        ]
    )


def main():
    raw_df = pd.read_csv(DATA_PATH)
    features, target = build_features(raw_df)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )

    preprocessor = build_preprocessor(features)
    baseline_models = {
        "Logistic Regression": make_pipeline(
            preprocessor,
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        ),
        "Decision Tree": make_pipeline(
            preprocessor,
            DecisionTreeClassifier(class_weight="balanced", random_state=42),
        ),
        "Random Forest": make_pipeline(
            preprocessor,
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    }

    trained_models = {}
    results = []
    for name, model in baseline_models.items():
        model.fit(x_train, y_train)
        trained_models[name] = model
        results.append(evaluate_classifier(name, model, x_test, y_test))

    tree_grid = GridSearchCV(
        estimator=make_pipeline(
            preprocessor,
            DecisionTreeClassifier(class_weight="balanced", random_state=42),
        ),
        param_grid={
            "model__max_depth": [3, 5, 10, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
        scoring="f1",
        cv=5,
        n_jobs=-1,
    )
    tree_grid.fit(x_train, y_train)
    trained_models["Tuned Decision Tree"] = tree_grid.best_estimator_
    results.append(
        evaluate_classifier("Tuned Decision Tree", tree_grid.best_estimator_, x_test, y_test)
    )

    forest_grid = GridSearchCV(
        estimator=make_pipeline(
            preprocessor,
            RandomForestClassifier(class_weight="balanced", random_state=42),
        ),
        param_grid={
            "model__n_estimators": [200, 300],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_leaf": [1, 2],
            "model__max_features": ["sqrt", "log2"],
        },
        scoring="f1",
        cv=5,
        n_jobs=-1,
    )
    forest_grid.fit(x_train, y_train)
    trained_models["Tuned Random Forest"] = forest_grid.best_estimator_
    results.append(
        evaluate_classifier("Tuned Random Forest", forest_grid.best_estimator_, x_test, y_test)
    )

    results_df = comparison_table(results)
    best_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_name]

    numeric_defaults = x_train.select_dtypes(include=["int64", "float64"]).median().to_dict()
    categorical_defaults = x_train.select_dtypes(include=["object"]).mode().iloc[0].to_dict()
    bundle = {
        "pipeline": best_model,
        "feature_columns": features.columns.tolist(),
        "defaults": {**numeric_defaults, **categorical_defaults},
        "model_name": best_name,
        "metrics": results_df.to_dict(orient="records"),
    }

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(bundle, BUNDLE_PATH)

    print(results_df.to_string(index=False))
    print(f"\nSaved final model: {MODEL_PATH}")
    print(f"Saved app bundle: {BUNDLE_PATH}")


if __name__ == "__main__":
    main()
