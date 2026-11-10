from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(features):
    """Build leakage-safe numeric and categorical preprocessing."""
    numeric_features = features.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = features.select_dtypes(include=["object"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def build_random_forest_pipeline(features):
    """Build the final Random Forest pipeline used by the project."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(features)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_features="log2",
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
