import numpy as np
import pandas as pd


DROP_COLUMNS = [
    "Unnamed: 0", "Unnamed: 6", "id", "object_id", "name", "city",
    "zip_code", "state_code.1", "status", "labels", "closed_at",
    "founded_at", "first_funding_at", "last_funding_at",
]


def build_features(df):
    """Return model features and the binary outcome target."""
    target = df["labels"].copy()
    features = df.drop(columns=DROP_COLUMNS).copy()

    features["funding_per_round"] = np.where(
        features["funding_rounds"] > 0,
        features["funding_total_usd"] / features["funding_rounds"],
        0,
    )
    features["funding_duration_years"] = (
        features["age_last_funding_year"] - features["age_first_funding_year"]
    )
    features["advanced_funding_rounds"] = features[
        ["has_roundA", "has_roundB", "has_roundC", "has_roundD"]
    ].sum(axis=1)

    return features, target
