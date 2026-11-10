from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st


st.set_page_config(page_title="StartupShield AI", page_icon="Shield", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "startup data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "startup_shield_final_bundle.pkl"


@st.cache_resource
def load_bundle():
    """Load the trained pipeline and prediction metadata once."""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    """Load the historical dataset once for dashboard views."""
    return pd.read_csv(DATA_PATH)


bundle = load_bundle()
df = load_data()
pipeline = bundle["pipeline"]
feature_columns = bundle["feature_columns"]
defaults = bundle["defaults"]
model_name = bundle["model_name"]
metrics_df = pd.DataFrame(bundle.get("metrics", []))


def safe_number(data, column):
    """Return a numeric input safely, using zero for missing values."""
    return pd.to_numeric(pd.Series([data.get(column, 0)]), errors="coerce").fillna(0).iloc[0]


def prepare_prediction_data(input_data):
    """Create a model-ready frame without accepting target or leakage columns."""
    prepared_df = pd.DataFrame(
        {column: [defaults.get(column)] * len(input_data) for column in feature_columns}
    )

    for column in feature_columns:
        if column in input_data.columns:
            prepared_df[column] = input_data[column]

    funding = pd.to_numeric(prepared_df["funding_total_usd"], errors="coerce").fillna(0)
    rounds = pd.to_numeric(prepared_df["funding_rounds"], errors="coerce").fillna(0)
    prepared_df["funding_per_round"] = np.where(rounds > 0, funding / rounds, 0)
    prepared_df["funding_duration_years"] = (
        pd.to_numeric(prepared_df["age_last_funding_year"], errors="coerce").fillna(0)
        - pd.to_numeric(prepared_df["age_first_funding_year"], errors="coerce").fillna(0)
    )

    funding_stage_columns = ["has_roundA", "has_roundB", "has_roundC", "has_roundD"]
    prepared_df["advanced_funding_rounds"] = (
        prepared_df[funding_stage_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .sum(axis=1)
    )
    return prepared_df.reindex(columns=feature_columns)


def get_suggestions(data, acquired_probability):
    """Return transparent rule-based recommendations for a demo prediction."""
    suggestions = []
    funding = safe_number(data, "funding_total_usd")
    rounds = safe_number(data, "funding_rounds")
    milestones = safe_number(data, "milestones")
    relationships = safe_number(data, "relationships")

    if acquired_probability < 0.50:
        suggestions.append("Strengthen product-market fit and validate the business model with measurable traction.")
    if funding == 0:
        suggestions.append("Explore bootstrapping, angel investment, or VC funding after preparing a clear use-of-funds plan.")
    if rounds < 2:
        suggestions.append("Build measurable traction before approaching investors for the next funding round.")
    if milestones < 2:
        suggestions.append("Set product, customer-growth, revenue, and partnership milestones that can be tracked over time.")
    if relationships < 5:
        suggestions.append("Build founder, mentor, investor, and industry relationships to expand support and opportunities.")
    if not suggestions:
        suggestions.append("Maintain growth momentum and monitor funding, customer traction, milestones, and strategic partnerships.")
    return suggestions


def chatbot_reply(question):
    """Provide project-specific answers without an external AI API."""
    question = question.lower()

    if "project" in question or "everything" in question or "all about" in question:
        return (
            "StartupShield AI is an explainable machine-learning project that predicts whether a startup is likely "
            "to be acquired or closed. It compares Logistic Regression, Decision Tree, and Random Forest models, "
            "then uses the tuned Random Forest for predictions, bulk CSV analysis, improvement suggestions, and SHAP explanations."
        )
    if "dataset" in question or "data" in question:
        return (
            "The historical dataset contains startup funding, funding rounds, milestones, relationships, category, "
            "state, investor-related fields, and outcome labels. Label 1 means Acquired and label 0 means Closed."
        )
    if "predict" in question or "prediction" in question:
        return "The prediction page uses the final tuned Random Forest pipeline to estimate Acquired or Closed and shows a confidence score."
    if "shap" in question or "explain" in question:
        return "SHAP explains a prediction by showing which input features pushed the model toward Acquired or Closed."
    if "funding" in question:
        return "Funding can indicate a startup's capacity to build, hire, and scale, but it does not guarantee acquisition or long-term success."
    if "milestone" in question:
        return "Milestones are measurable achievements such as a product launch, customer growth, revenue target, or strategic partnership."
    if "relationship" in question or "network" in question:
        return "Founder, mentor, investor, and industry relationships can support learning, partnerships, fundraising, and market access."
    if "improve" in question or "suggestion" in question:
        return "Improvement suggestions are transparent rule-based recommendations based on funding, milestones, funding rounds, and relationships."
    if "accuracy" in question or "recall" in question or "f1" in question or "model" in question:
        return "Open Model Performance to view Accuracy, Precision, Recall, F1-Score, and ROC-AUC for the compared models."
    if "limit" in question or "risk" in question:
        return "This academic system identifies patterns in historical data only. It is not investment advice and does not guarantee a real startup's future outcome."
    return "Try asking about the project, dataset, prediction, model performance, funding, milestones, SHAP, suggestions, or limitations."


st.sidebar.title("StartupShield AI")
page = st.sidebar.radio(
    "Navigate",
    [
        "Home", "Dataset Overview", "Model Performance", "Startup Prediction",
        "Bulk CSV Prediction", "SHAP Explainability", "AI Assistant", "About and Limitations",
    ],
)


if page == "Home":
    acquired_rate = (df["status"] == "acquired").mean() * 100
    st.title("StartupShield AI")
    st.caption("Explainable Machine Learning System for Startup Outcome Prediction")
    st.divider()

    left_column, right_column = st.columns([3, 2])
    with left_column:
        st.subheader("Predict. Understand. Improve.")
        st.write("Use historical startup signals to estimate an outcome, understand feature impact, and explore transparent improvement suggestions.")
        st.info("Academic project disclaimer: results identify patterns in this historical dataset. They are not investment or business advice.")
    with right_column:
        with st.container(border=True):
            st.subheader("Selected Model")
            st.write(f"**{model_name}**")
            st.caption("Selected after model comparison and hyperparameter tuning.")

    st.subheader("Project Snapshot")
    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Startup Records", f"{len(df):,}")
    metric_2.metric("Dataset Features", df.shape[1])
    metric_3.metric("Acquired Records", f"{acquired_rate:.1f}%")
    st.divider()

    st.subheader("What You Can Explore")
    feature_1, feature_2, feature_3 = st.columns(3)
    with feature_1:
        with st.container(border=True):
            st.markdown("### Dataset Insights")
            st.write("Review startup outcomes, the data preview, and missing-value patterns.")
    with feature_2:
        with st.container(border=True):
            st.markdown("### Outcome Prediction")
            st.write("Enter startup details or upload a compatible CSV for outcome predictions.")
    with feature_3:
        with st.container(border=True):
            st.markdown("### Explainable AI")
            st.write("Use SHAP to see which features influence a specific prediction.")
    st.caption("Use the navigation menu to explore the complete system.")

elif page == "Dataset Overview":
    st.title("Dataset Overview")
    first_metric, second_metric, third_metric = st.columns(3)
    first_metric.metric("Total Startups", f"{len(df):,}")
    second_metric.metric("Columns", df.shape[1])
    third_metric.metric("Duplicate Rows", int(df.duplicated().sum()))
    left_chart, right_chart = st.columns(2)
    with left_chart:
        st.subheader("Startup Outcome Distribution")
        st.bar_chart(df["status"].value_counts())
    with right_chart:
        st.subheader("Missing Values")
        missing_values = df.isnull().sum()
        missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
        st.bar_chart(missing_values)
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

elif page == "Model Performance":
    st.title("Model Performance")
    st.write("Models are compared using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.")
    if metrics_df.empty:
        st.warning("The saved bundle does not contain model metrics.")
    else:
        ordered_metrics = metrics_df.sort_values(by="F1-Score", ascending=False)
        st.dataframe(ordered_metrics, use_container_width=True)
        best_result = ordered_metrics.iloc[0]
        st.success(f"Final model: {best_result['Model']} | F1-Score: {best_result['F1-Score']:.3f}")
        st.bar_chart(ordered_metrics.set_index("Model")[["F1-Score", "ROC-AUC"]])

elif page == "Startup Prediction":
    st.title("Startup Outcome Prediction")
    st.write("Enter startup details, then select **Predict Startup Outcome**.")
    state_options = sorted(df["state_code"].dropna().unique())
    category_options = sorted(df["category_code"].dropna().unique())
    left_column, right_column = st.columns(2)
    with left_column:
        state_code = st.selectbox("State Code", state_options)
        category_code = st.selectbox("Startup Category", category_options)
        funding_total_usd = st.number_input("Total Funding (USD)", min_value=0.0, value=0.0)
        funding_rounds = st.number_input("Funding Rounds", min_value=0, value=0)
    with right_column:
        relationships = st.number_input("Number of Relationships", min_value=0, value=0)
        milestones = st.number_input("Number of Milestones", min_value=0, value=0)
        has_vc = st.checkbox("Has VC Funding")
        has_angel = st.checkbox("Has Angel Funding")
    if st.button("Predict Startup Outcome", type="primary"):
        manual_input = pd.DataFrame([{
            "state_code": state_code, "category_code": category_code,
            "funding_total_usd": funding_total_usd, "funding_rounds": funding_rounds,
            "relationships": relationships, "milestones": milestones,
            "has_VC": int(has_vc), "has_angel": int(has_angel),
        }])
        input_df = prepare_prediction_data(manual_input)
        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0][1]
        st.session_state["input_df"] = input_df
        if prediction == 1:
            st.success(f"Likely Acquired — Confidence: {probability:.1%}")
        else:
            st.error(f"Likely Closed — Confidence: {1 - probability:.1%}")
        st.subheader("Improvement Suggestions")
        for suggestion in get_suggestions(manual_input.iloc[0], probability):
            st.write(f"- {suggestion}")

elif page == "Bulk CSV Prediction":
    st.title("Bulk CSV Upload and Auto Prediction")
    st.write("Upload a compatible startup CSV to predict outcomes for multiple startups.")
    st.caption("Required columns: funding_total_usd, funding_rounds, relationships, and milestones.")
    uploaded_file = st.file_uploader("Upload Startup CSV", type=["csv"])
    required_columns = ["funding_total_usd", "funding_rounds", "relationships", "milestones"]
    if uploaded_file is not None:
        upload_df = pd.read_csv(uploaded_file)
        missing_columns = [column for column in required_columns if column not in upload_df.columns]
        if missing_columns:
            st.error("Incompatible CSV. Missing columns: " + ", ".join(missing_columns))
        else:
            prediction_df = prepare_prediction_data(upload_df)
            predicted_labels = pipeline.predict(prediction_df)
            acquired_probabilities = pipeline.predict_proba(prediction_df)[:, 1]
            result_df = upload_df.copy()
            result_df["Predicted Outcome"] = np.where(predicted_labels == 1, "Likely Acquired", "Likely Closed")
            result_df["Acquired Probability (%)"] = (acquired_probabilities * 100).round(2)
            result_df["Primary Suggestion"] = [
                get_suggestions(upload_df.iloc[position], probability)[0]
                for position, probability in enumerate(acquired_probabilities)
            ]
            st.success(f"Predictions completed for {len(result_df):,} startups.")
            st.dataframe(result_df, use_container_width=True)
            st.download_button("Download Prediction Results CSV", data=result_df.to_csv(index=False).encode("utf-8"), file_name="startupshield_predictions.csv", mime="text/csv")

elif page == "SHAP Explainability":
    st.title("SHAP Explainability")
    if "input_df" not in st.session_state:
        st.info("First make a prediction in the Startup Prediction section.")
    else:
        input_df = st.session_state["input_df"]
        preprocessor = pipeline.named_steps["preprocessor"]
        random_forest = pipeline.named_steps["model"]
        processed_input = preprocessor.transform(input_df)
        processed_features = preprocessor.get_feature_names_out()
        explainer = shap.TreeExplainer(random_forest)
        shap_values = explainer.shap_values(processed_input)
        if isinstance(shap_values, list):
            acquired_shap_values = shap_values[1][0]
        elif shap_values.ndim == 3:
            acquired_shap_values = shap_values[0, :, 1]
        else:
            acquired_shap_values = shap_values[0]
        explanation_df = pd.DataFrame({"Feature": processed_features, "SHAP Impact": acquired_shap_values})
        explanation_df["Absolute Impact"] = explanation_df["SHAP Impact"].abs()
        explanation_df = explanation_df.sort_values(by="Absolute Impact", ascending=False).head(10)
        st.write("Positive impact pushes the prediction toward Acquired; negative impact pushes it toward Closed.")
        st.bar_chart(explanation_df.set_index("Feature")["SHAP Impact"])
        st.dataframe(explanation_df, use_container_width=True)

elif page == "AI Assistant":
    st.title("StartupShield AI Assistant")
    st.caption("This is a project-specific, rule-based assistant. It does not use an external AI API.")
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []
    if st.button("Clear chat"):
        st.session_state["chat_messages"] = []
        st.rerun()
    for message in st.session_state["chat_messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    user_question = st.chat_input("Ask about the project, dataset, prediction, SHAP, or limitations")
    if user_question:
        st.session_state["chat_messages"].append({"role": "user", "content": user_question})
        st.session_state["chat_messages"].append({"role": "assistant", "content": chatbot_reply(user_question)})
        st.rerun()

elif page == "About and Limitations":
    st.title("About StartupShield AI")
    st.write("StartupShield AI is a final-year machine-learning project that combines outcome prediction with explainable AI.")
    left_column, right_column = st.columns(2)
    with left_column:
        st.subheader("Technology Stack")
        st.write("Python, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Streamlit, SHAP, Joblib, Git, and GitHub.")
    with right_column:
        st.subheader("Project Scope")
        st.write("Historical data analysis, startup outcome prediction, bulk prediction, transparent suggestions, and model explainability.")
    st.subheader("Limitations")
    st.write(
        "- The dataset is historical and may not represent every startup.\n"
        "- Predictions are not investment or business advice.\n"
        "- Association does not prove that a feature causes acquisition or closure.\n"
        "- The single-startup demo fills unprovided features with training-data defaults."
    )
