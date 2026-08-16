import os
import cloudpickle
import numpy as np
import pandas as pd
import streamlit as st

ARTIFACT_PATH = os.path.join("model", "artifacts.pkl")

st.set_page_config(page_title="Bank Marketing ML Classifier", page_icon="📊", layout="wide")

@st.cache_resource
def load_artifacts():
    if not os.path.exists(ARTIFACT_PATH):
        return None
    with open(ARTIFACT_PATH, "rb") as f:
        return cloudpickle.load(f)

def prepare_features(df, artifacts):
    numerical_cols = artifacts["numerical_cols"]
    categorical_cols = artifacts["categorical_cols"]
    required = numerical_cols + categorical_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    # StandardScaler was fitted on numerical columns only in the notebook.
    x_num = artifacts["scaler"].transform(df[numerical_cols])
    x_cat = artifacts["encoder"].transform(df[categorical_cols])
    return np.hstack([x_num, x_cat]).astype(float)

def normalize_target(series):
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")
    else:
        values = series.astype(str).str.strip().str.lower().map(
            {"no": 0, "yes": 1, "0": 0, "1": 1}
        )
    if values.isna().any():
        return None
    return values.astype(int).to_numpy()

def safe_auc(y_true, scores):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pos, neg = np.sum(y_true == 1), np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        return np.nan
    order = np.argsort(-scores)
    ys, ss = y_true[order], scores[order]
    tp, fp = np.cumsum(ys == 1), np.cumsum(ys == 0)
    distinct = np.r_[ss[:-1] != ss[1:], True]
    tp, fp = tp[distinct], fp[distinct]
    tpr = np.r_[0.0, tp / pos, 1.0]
    fpr = np.r_[0.0, fp / neg, 1.0]
    return float(np.trapezoid(tpr, fpr))

def evaluate(y_true, y_pred, y_prob):
    y_true, y_pred, y_prob = map(np.asarray, (y_true, y_pred, y_prob))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0

    return {
        "Accuracy": accuracy, "Precision": precision, "Recall": recall,
        "F1": f1, "MCC": mcc, "AUC": safe_auc(y_true, y_prob),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp
    }

st.title("📊 Bank Marketing Classification")
st.caption("Machine Learning Assignment 2 — five classification models implemented from scratch.")

artifacts = load_artifacts()
if artifacts is None:
    st.error("model/artifacts.pkl was not found.")
    st.stop()

uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])
selected_model = st.selectbox("Select a model", list(artifacts["models"].keys()))

if uploaded is None:
    st.info("Upload the test CSV to start evaluation.")
    st.stop()

try:
    data = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

st.subheader("Uploaded data")
st.write(f"Rows: **{len(data):,}**")
st.dataframe(data.head(10), use_container_width=True)

try:
    X_processed = prepare_features(data, artifacts)
except Exception as exc:
    st.error(str(exc))
    st.stop()

model = artifacts["models"][selected_model]

with st.spinner(f"Running {selected_model}..."):
    probabilities = np.asarray(model.predict_proba(X_processed), dtype=float)
    if probabilities.ndim == 2:
        probabilities = probabilities[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

result_df = data.copy()
result_df["predicted_y"] = np.where(predictions == 1, "yes", "no")
result_df["probability_yes"] = probabilities

st.subheader(f"{selected_model} — predictions")
st.dataframe(result_df.head(25), use_container_width=True)
st.download_button(
    "Download predictions CSV",
    result_df.to_csv(index=False).encode("utf-8"),
    file_name="predictions.csv",
    mime="text/csv"
)

if "y" in data.columns:
    y_true = normalize_target(data["y"])
    if y_true is None:
        st.warning("Target y must contain 0/1 or no/yes values.")
    else:
        metrics = evaluate(y_true, predictions, probabilities)
        st.subheader("Evaluation metrics")
        cols = st.columns(6)
        for col, name in zip(cols, ["Accuracy", "Precision", "Recall", "F1", "MCC", "AUC"]):
            value = metrics[name]
            col.metric(name, "N/A" if np.isnan(value) else f"{value:.4f}")

        st.subheader("Confusion Matrix")
        cm = pd.DataFrame(
            [[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]],
            index=["Actual 0 (no)", "Actual 1 (yes)"],
            columns=["Predicted 0 (no)", "Predicted 1 (yes)"]
        )
        st.dataframe(cm, use_container_width=True)
else:
    st.info("No y column found. Predictions are shown, but evaluation requires y.")

st.divider()
st.caption("Models are custom NumPy implementations. Preprocessing uses the notebook's fitted encoder and scaler.")
