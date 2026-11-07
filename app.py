# Streamlit app for ML Assignment 2 - Bank Marketing (UCI)
#
# Upload the test CSV, choose a model, and the app shows the metrics,
# confusion matrix and classification report.
# The models are already trained (train_models.py) - here they are only loaded,
# nothing is retrained on the uploaded file.

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

folder = os.path.dirname(os.path.abspath(__file__))
model_folder = os.path.join(folder, "model")
sample_file = os.path.join(folder, "test_data.csv")

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

feature_cols = ["age", "job", "marital", "education", "default", "balance", "housing",
                "loan", "contact", "day", "month", "duration", "campaign", "pdays",
                "previous", "poutcome"]

# short notes shown under each metric
metric_notes = {
    "Accuracy": "Correct predictions out of all clients",
    "AUC": "How well yes is ranked above no (0.5 = random)",
    "Precision": "Of the predicted yes, how many were right",
    "Recall": "Of the actual yes, how many were found",
    "F1": "Mix of precision and recall",
    "MCC": "Prediction vs truth correlation (-1 to 1)",
}

st.set_page_config(page_title="Bank Marketing Classifier", page_icon="🏦", layout="wide")

# some styling for the header, the summary chips and the metric cards
st.markdown("""
<style>
.block-container {padding-top: 3.5rem; padding-bottom: 1rem;}
.banner {background: linear-gradient(90deg,#0f4c75,#1f6feb); padding:0.8rem 1.3rem;
         border-radius:12px; color:#fff; margin-bottom:0.7rem;}
.banner h1 {margin:0; font-size:1.45rem; color:#fff;}
.banner p  {margin:0.2rem 0 0 0; color:#e6efff; font-size:0.85rem;}
.chips {display:flex; gap:0.5rem; flex-wrap:wrap; margin-bottom:0.4rem;}
.chip {background:#eef4fb; border:1px solid #d6e2f0; border-radius:999px;
       padding:0.2rem 0.8rem; font-size:0.85rem; color:#1b2a3a;}
.card {background:#fff; border:1px solid #d6e2f0; border-left:5px solid #1f6feb;
       border-radius:10px; padding:0.55rem 0.8rem; height:96px; overflow:hidden;}
.card .name {font-size:0.75rem; font-weight:700; color:#1f6feb; text-transform:uppercase;}
.card .val {font-size:1.45rem; font-weight:700; color:#0f2540; line-height:1.25;}
.card .note {font-size:0.7rem; color:#5b6b7d; line-height:1.2;}
.step {background:#eef4fb; border-radius:10px; padding:0.8rem 1rem; color:#1b2a3a;}
[data-testid="stHeaderActionElements"] {display:none;}
h3 {padding-top:0.4rem !important; padding-bottom:0.3rem !important;}
</style>
<div class="banner">
  <h1>🏦 Bank Marketing - Term Deposit Prediction</h1>
  <p>UCI Bank Marketing dataset · Does the client subscribe to a term deposit?
     · 5 trained models tested on the uploaded data</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(fname):
    return joblib.load(os.path.join(model_folder, fname))


def target_to_01(col):
    # target can come as yes/no or already as 1/0
    if col.dtype == object or str(col.dtype).startswith("str"):
        return (col.astype(str).str.strip().str.lower() == "yes").astype(int)
    return col.astype(int)


def get_scores(model, X, y):
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    scores = {
        "Accuracy": accuracy_score(y, pred),
        # AUC needs both classes present in the file
        "AUC": roc_auc_score(y, prob) if y.nunique() > 1 else float("nan"),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }
    return scores, pred


def draw_confusion_matrix(cm, title):
    tags = np.array([["TN", "FP"], ["FN", "TP"]])
    fig, ax = plt.subplots(figsize=(2.9, 2.4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            # white text on the dark cells so it stays readable
            color = "white" if cm[i, j] > cm.max() / 2 else "#0f2540"
            ax.text(j, i, f"{tags[i, j]}\n{cm[i, j]}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)
    ax.set_xticks([0, 1], ["no", "yes"])
    ax.set_yticks([0, 1], ["no", "yes"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---- sidebar ----
with st.sidebar:
    st.header("⚙️ Options")
    uploaded = st.file_uploader("**1. Upload test data (CSV)**", type=["csv"])
    model_name = st.selectbox("**2. Choose a model**", list(model_files.keys()))
    st.caption("The CSV needs all 16 feature columns and the target column `y` (yes/no).")
    if os.path.exists(sample_file):
        with open(sample_file, "rb") as f:
            st.download_button("⬇️ Download sample test_data.csv", f, "test_data.csv",
                               "text/csv", use_container_width=True)
    st.divider()
    st.caption("Models were trained beforehand in `train_models.py` and are only "
               "loaded here.")

# nothing uploaded yet - show a quick how-to
if uploaded is None:
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="step"><b>1.</b> Upload a test CSV from the sidebar '
                '(the sample file can be downloaded there).</div>', unsafe_allow_html=True)
    c2.markdown('<div class="step"><b>2.</b> Choose one of the 5 models.</div>',
                unsafe_allow_html=True)
    c3.markdown('<div class="step"><b>3.</b> Check the metrics, confusion matrix '
                'and the model comparison.</div>', unsafe_allow_html=True)
    st.stop()

# ---- read and check the file ----
try:
    data = pd.read_csv(uploaded, sep=None, engine="python")   # works for , or ;
except Exception as e:
    st.error(f"Couldn't read the CSV file: {e}")
    st.stop()

missing = [c for c in feature_cols + ["y"] if c not in data.columns]
if missing:
    st.error(f"These columns are missing in the file: {missing}")
    st.stop()

X = data[feature_cols]
y = target_to_01(data["y"])

st.markdown(
    f'<div class="chips">'
    f'<span class="chip">Rows: <b>{len(data):,}</b></span>'
    f'<span class="chip">Features: <b>{len(feature_cols)}</b></span>'
    f'<span class="chip">Yes: <b>{int(y.sum())}</b></span>'
    f'<span class="chip">No: <b>{int((1 - y).sum())}</b></span>'
    f'</div>', unsafe_allow_html=True)
with st.expander("📄 See the uploaded data"):
    st.dataframe(data.head(20), use_container_width=True)

# ---- metrics for the chosen model ----
model = load_model(model_files[model_name])
scores, pred = get_scores(model, X, y)

st.subheader(f"📈 Metrics - {model_name}")
cols = st.columns(6)
for col, (metric, value) in zip(cols, scores.items()):
    col.markdown(f'<div class="card"><div class="name">{metric}</div>'
                 f'<div class="val">{value:.4f}</div>'
                 f'<div class="note">{metric_notes[metric]}</div></div>',
                 unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🧮 Confusion matrix & report", "📊 All 5 models",
                            "📉 ROC curves"])

with tab1:
    left, right = st.columns([1, 1.2])
    cm = confusion_matrix(y, pred, labels=[0, 1])
    with left:
        st.pyplot(draw_confusion_matrix(cm, model_name), use_container_width=False)
    with right:
        report = classification_report(y, pred, labels=[0, 1], target_names=["no", "yes"],
                                       output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)
        tn, fp, fn, tp = cm.ravel()
        st.info(f"Out of **{tp + fn}** clients who actually subscribed, the model "
                f"caught **{tp}** and missed **{fn}**. It also wrongly predicted "
                f"yes for **{fp}** clients.")

        # let the user download the predictions for this model
        out = data.copy()
        out["predicted_y"] = np.where(pred == 1, "yes", "no")
        st.download_button("⬇️ Download predictions (CSV)", out.to_csv(index=False),
                           f"predictions_{model_files[model_name].split('.')[0]}.csv",
                           "text/csv")

with tab2:
    rows = []
    for name, fname in model_files.items():
        s, _ = get_scores(load_model(fname), X, y)
        rows.append({"ML Model Name": name, **s})
    comparison = pd.DataFrame(rows).set_index("ML Model Name")
    st.dataframe(
        comparison.style.format("{:.4f}").highlight_max(
            axis=0, props="background-color:#1f6feb; color:white; font-weight:bold;"),
        use_container_width=True)
    best = comparison["MCC"].idxmax()
    st.success(f"🏆 Best model here by MCC: **{best}** "
               f"(MCC = {comparison.loc[best, 'MCC']:.4f})")
    st.caption("Blue = best value in each column. Only about 11.5% of clients say yes, "
               "so accuracy alone can look good even for a weak model - "
               "F1 and MCC give a better picture.")

    # bar chart - easier to compare the models at a glance
    pick = st.multiselect("Metrics to plot", list(metric_notes.keys()),
                          default=["AUC", "F1", "MCC"])
    if pick:
        fig, ax = plt.subplots(figsize=(8, 3))
        width = 0.8 / len(pick)
        x = np.arange(len(comparison))
        for i, m in enumerate(pick):
            ax.bar(x + i * width, comparison[m], width, label=m)
        ax.set_xticks(x + width * (len(pick) - 1) / 2, comparison.index, fontsize=8)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, ncol=len(pick))
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)

with tab3:
    # ROC curve for every model, the selected one drawn thicker
    if y.nunique() < 2:
        st.warning("ROC curve needs both yes and no rows in the file.")
    else:
        fig, ax = plt.subplots(figsize=(5, 3.6))
        for name, fname in model_files.items():
            prob = load_model(fname).predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, prob)
            auc = roc_auc_score(y, prob)
            ax.plot(fpr, tpr, lw=2.8 if name == model_name else 1.2,
                    label=f"{name} (AUC {auc:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(fontsize=7, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        left, right = st.columns([1.4, 1])
        left.pyplot(fig)
        right.markdown("**How to read this**  \nThe closer a curve is to the top-left "
                       "corner, the better the model separates yes from no. The dashed "
                       f"line is random guessing. The selected model (**{model_name}**) "
                       "is drawn thicker.")
