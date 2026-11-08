# Streamlit app for ML Assignment 2 - Bank Marketing (UCI)
#
# Opens with test_data.csv (or an uploaded CSV), lets you pick a model and shows
# the metrics, confusion matrix, classification report and comparisons.
# The models are already trained (train_models.py) - here they are only loaded,
# nothing is retrained on the uploaded file.

import os

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
model_icons = {"Logistic Regression": "📐", "Decision Tree": "🌳", "KNN": "📍",
               "Naive Bayes": "🎲", "Support Vector Machine": "🧭"}

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

# "aurora" jewel-tone palette on a deep midnight background
metric_colors = {"Accuracy": "#5ab8f5", "AUC": "#8b8cf8", "Precision": "#f27fa5",
                 "Recall": "#f6b65c", "F1": "#3dd6a3", "MCC": "#2ec4c9"}
model_colors = {"Logistic Regression": "#5ab8f5", "Decision Tree": "#f6b65c",
                "KNN": "#3dd6a3", "Naive Bayes": "#f27fa5",
                "Support Vector Machine": "#8b8cf8"}
good_color, bad_color = "#14a38b", "#d9596a"   # teal for right, coral for wrong
grid = "rgba(139,140,248,0.15)"

# shared look for all plotly charts
chart_layout = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20,24,52,0.55)",
                    font=dict(color="#e5e7f5"), margin=dict(l=10, r=10, t=40, b=10),
                    hoverlabel=dict(bgcolor="#1b2048", font_color="#ffffff"))

st.set_page_config(page_title="Bank Marketing Classifier", page_icon="🏦", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 3rem; padding-bottom: 1rem;}
/* soft aurora glow behind the whole page */
[data-testid="stAppViewContainer"] {
  background: radial-gradient(1200px 500px at 85% -10%, rgba(139,140,248,0.18), transparent 60%),
              radial-gradient(900px 450px at -10% 30%, rgba(46,196,201,0.12), transparent 60%),
              #0b1020;}
[data-testid="stHeader"] {background: transparent;}
[data-testid="stSidebar"] {background: linear-gradient(180deg,#121735 0%,#1d1747 60%,#26174a 100%);
                           border-right:1px solid rgba(139,140,248,0.20);}
[data-testid="stSidebar"] h2 {color:#f6b65c;}
[data-testid="stHeaderActionElements"] {display:none;}

/* animated aurora header */
@keyframes flow {0% {background-position:0% 50%} 50% {background-position:100% 50%}
                 100% {background-position:0% 50%}}
.hero {background: linear-gradient(115deg,#0e7490,#4f46e5,#9333ea,#db2777,#4f46e5,#0e7490);
       background-size: 300% 300%; animation: flow 18s ease infinite;
       padding:1.05rem 1.5rem; border-radius:18px; color:#fff; margin-bottom:0.9rem;
       box-shadow:0 10px 35px rgba(79,70,229,0.35), inset 0 1px 0 rgba(255,255,255,0.15);}
.hero h1 {margin:0; font-size:1.6rem; color:#fff; letter-spacing:0.4px;
          text-shadow:0 2px 10px rgba(0,0,0,0.25);}
.hero p {margin:0.3rem 0 0 0; color:#eef0ff; font-size:0.87rem;}

.chips {display:flex; gap:0.5rem; flex-wrap:wrap; margin:0.3rem 0 0.6rem 0;}
.chip {border-radius:999px; padding:0.24rem 0.9rem; font-size:0.84rem; color:#fff;
       border:1px solid var(--b); background:var(--f);}
.c0 {--f:rgba(246,182,92,0.16);  --b:rgba(246,182,92,0.55);}
.c1 {--f:rgba(90,184,245,0.16);  --b:rgba(90,184,245,0.55);}
.c2 {--f:rgba(139,140,248,0.18); --b:rgba(139,140,248,0.55);}
.c3 {--f:rgba(61,214,163,0.16);  --b:rgba(61,214,163,0.55);}
.c4 {--f:rgba(242,127,165,0.16); --b:rgba(242,127,165,0.55);}

/* glass metric cards: coloured top edge, glow on hover, progress bar */
.card {position:relative; border:1px solid rgba(255,255,255,0.08);
       border-top:3px solid var(--c);
       background: linear-gradient(160deg, var(--bg) 0%, rgba(20,24,52,0.92) 65%);
       border-radius:16px; padding:0.65rem 0.9rem 0.75rem; height:122px; overflow:hidden;
       box-shadow:0 4px 18px rgba(0,0,0,0.30);
       transition: transform .2s ease, box-shadow .2s ease;}
.card:hover {transform: translateY(-5px); box-shadow:0 12px 30px var(--glow);}
.card .name {font-size:0.72rem; font-weight:800; color:var(--c); letter-spacing:1.2px;
             text-transform:uppercase;}
.card .val {font-size:1.65rem; font-weight:800; color:#ffffff; line-height:1.2;}
.card .note {font-size:0.68rem; color:#a7acd0; line-height:1.2; margin-top:2px;}
.card .bar {position:absolute; left:0; bottom:0; height:5px; width:var(--w);
            background:linear-gradient(90deg, var(--c), #ffffff99);
            border-radius:0 5px 5px 0;}
.card .rank {position:absolute; top:9px; right:10px; font-size:0.68rem; font-weight:700;
             color:var(--c); background:rgba(255,255,255,0.07);
             border:1px solid var(--c); border-radius:999px; padding:0 7px;}

.model-tag {display:inline-block; padding:0.3rem 1rem; border-radius:12px;
            font-weight:800; color:#0b1020; margin-bottom:0.5rem;
            background:linear-gradient(90deg, var(--c), #ffffffcc);
            box-shadow:0 4px 16px var(--c);}
.story {background:linear-gradient(90deg, rgba(61,214,163,0.14), rgba(20,24,52,0.9));
        border:1px solid rgba(61,214,163,0.35); border-left:5px solid #3dd6a3;
        border-radius:12px; padding:0.75rem 0.95rem; color:#eef0ff; font-size:0.9rem;}
.stTabs [data-baseweb="tab"] {font-weight:600;}
h3 {padding-top:0.4rem !important; padding-bottom:0.3rem !important;}
</style>
<div class="hero">
  <h1>🏦 Bank Marketing · Term Deposit Predictor</h1>
  <p>Will the client subscribe? &nbsp;·&nbsp; UCI Bank Marketing data &nbsp;·&nbsp;
     5 pre-trained classifiers evaluated live on your test file</p>
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


def get_scores(prob, y, threshold=0.5):
    pred = (prob >= threshold).astype(int)
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


# ---- sidebar ----
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded = st.file_uploader("**1. Upload test data (CSV)**", type=["csv"])
    model_name = st.radio("**2. Choose a model**", list(model_files.keys()),
                          format_func=lambda m: f"{model_icons[m]}  {m}")
    threshold = st.slider("**3. Decision threshold**", 0.05, 0.95, 0.50, 0.05,
                          help="A client is predicted 'yes' when the model's "
                               "probability is at or above this value. 0.5 is the "
                               "default used for the README results.")
    st.caption("The CSV needs all 16 feature columns and the target column `y` (yes/no).")
    if os.path.exists(sample_file):
        with open(sample_file, "rb") as f:
            st.download_button("⬇️ Download sample test_data.csv", f, "test_data.csv",
                               "text/csv", width="stretch")
    st.divider()
    st.caption("Models were trained beforehand in `train_models.py` and are only "
               "loaded here - nothing is retrained.")

# ---- read and check the file ----
# if nothing is uploaded, open with the bundled test_data.csv so results show straight away
try:
    if uploaded is not None:
        data = pd.read_csv(uploaded, sep=None, engine="python")   # works for , or ;
        source = uploaded.name
    elif os.path.exists(sample_file):
        data = pd.read_csv(sample_file)
        source = "test_data.csv (default)"
    else:
        st.warning("Upload a test CSV from the sidebar to get started.")
        st.stop()
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
    f'<span class="chip c0">📂 {source}</span>'
    f'<span class="chip c1">Rows: <b>{len(data):,}</b></span>'
    f'<span class="chip c2">Features: <b>{len(feature_cols)}</b></span>'
    f'<span class="chip c3">Yes: <b>{int(y.sum())}</b></span>'
    f'<span class="chip c4">No: <b>{int((1 - y).sum())}</b></span>'
    f'<span class="chip c2">Threshold: <b>{threshold:.2f}</b></span>'
    f'</div>', unsafe_allow_html=True)
if uploaded is None:
    st.caption("ℹ️ Showing results on the default test_data.csv - upload your own CSV "
               "from the sidebar to replace it.")

# ---- scores for every model (needed for ranks and comparison) ----
probs = {name: load_model(f).predict_proba(X)[:, 1] for name, f in model_files.items()}
all_scores = {name: get_scores(p, y, threshold)[0] for name, p in probs.items()}
comparison = pd.DataFrame(all_scores).T
comparison.index.name = "ML Model Name"
scores, pred = get_scores(probs[model_name], y, threshold)

# ---- metric cards for the chosen model ----
mc = model_colors[model_name]
st.markdown(f'<span class="model-tag" style="--c:{mc}">{model_icons[model_name]} '
            f'{model_name}</span>', unsafe_allow_html=True)
cols = st.columns(6)
for col, (metric, value) in zip(cols, scores.items()):
    c = metric_colors[metric]
    rank = int(comparison[metric].rank(ascending=False, method="min")[model_name])
    width = max(0, min(1, value)) * 100
    col.markdown(f'<div class="card" style="--c:{c}; --bg:{c}33; --glow:{c}66; '
                 f'--w:{width:.0f}%;">'
                 f'<div class="rank">#{rank} of 5</div>'
                 f'<div class="name">{metric}</div>'
                 f'<div class="val">{value:.4f}</div>'
                 f'<div class="note">{metric_notes[metric]}</div>'
                 f'<div class="bar"></div></div>', unsafe_allow_html=True)

st.write("")
tab1, tab2, tab3, tab4 = st.tabs(["🧮 Confusion matrix & report", "🏁 Compare models",
                                  "📉 ROC curves", "🔍 Explore predictions"])

# ---- tab 1: confusion matrix + report ----
with tab1:
    left, right = st.columns([1, 1.15])
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    with left:
        text = [[f"TN<br><b>{tn}</b>", f"FP<br><b>{fp}</b>"],
                [f"FN<br><b>{fn}</b>", f"TP<br><b>{tp}</b>"]]
        # green = correct, red = mistake
        fig = go.Figure(go.Heatmap(
            z=[[1, 0], [0, 1]], x=["Predicted no", "Predicted yes"],
            y=["Actual no", "Actual yes"],
            text=text, texttemplate="%{text}", textfont=dict(size=18, color="white"),
            colorscale=[[0, bad_color], [1, good_color]], showscale=False,
            customdata=cm,
            hovertemplate="%{y} · %{x}<br>Count: %{customdata}<extra></extra>",
            xgap=4, ygap=4))
        fig.update_layout(**chart_layout, height=330,
                          title=f"Confusion matrix · {model_name}")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width="stretch")
    with right:
        report = classification_report(y, pred, labels=[0, 1], target_names=["no", "yes"],
                                       output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")
        st.markdown(f'<div class="story">Out of <b>{tp + fn}</b> clients who actually '
                    f'subscribed, the model caught <b>{tp}</b> and missed <b>{fn}</b>. '
                    f'It also wrongly predicted yes for <b>{fp}</b> clients.</div>',
                    unsafe_allow_html=True)
        st.write("")
        out = data.copy()
        out["predicted_y"] = np.where(pred == 1, "yes", "no")
        out["prob_yes"] = probs[model_name].round(4)
        st.download_button("⬇️ Download predictions (CSV)", out.to_csv(index=False),
                           f"predictions_{model_files[model_name].split('.')[0]}.csv",
                           "text/csv")

# ---- tab 2: compare all models ----
with tab2:
    st.dataframe(
        comparison.style.format("{:.4f}").highlight_max(
            axis=0, props=f"background-color:{good_color}; color:white; font-weight:bold;"),
        width="stretch")
    best = comparison["MCC"].idxmax()
    st.success(f"🏆 Best model here by MCC: **{best}** "
               f"(MCC = {comparison.loc[best, 'MCC']:.4f})")
    st.caption("Teal = best value in each column. Only about 11.5% of clients say yes, "
               "so accuracy alone can look good even for a weak model - "
               "F1 and MCC give a better picture.")

    view = st.segmented_control("Chart type", ["Bars", "Radar"], default="Bars")
    pick = st.multiselect("Metrics to plot", list(metric_notes.keys()),
                          default=["AUC", "F1", "MCC"])
    if pick and view == "Radar":
        fig = go.Figure()
        for name in comparison.index:
            vals = comparison.loc[name, pick].tolist()
            fig.add_trace(go.Scatterpolar(
                r=vals + vals[:1], theta=pick + pick[:1], name=name, fill="toself",
                opacity=0.85 if name == model_name else 0.35,
                line=dict(color=model_colors[name], width=3 if name == model_name else 1)))
        fig.update_layout(**chart_layout, height=430,
                          polar=dict(bgcolor="rgba(20,24,52,0.55)",
                                     radialaxis=dict(range=[0, 1], gridcolor=grid),
                                     angularaxis=dict(gridcolor=grid)))
        st.plotly_chart(fig, width="stretch")
    elif pick:
        fig = go.Figure()
        for m in pick:
            fig.add_trace(go.Bar(x=comparison.index, y=comparison[m], name=m,
                                 marker_color=metric_colors[m],
                                 hovertemplate="%{x}<br>" + m + ": %{y:.4f}<extra></extra>"))
        fig.update_layout(**chart_layout, barmode="group", height=380,
                          yaxis=dict(range=[0, 1], gridcolor=grid),
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, width="stretch")

# ---- tab 3: ROC curves ----
with tab3:
    if y.nunique() < 2:
        st.warning("ROC curve needs both yes and no rows in the file.")
    else:
        fig = go.Figure()
        for name, p in probs.items():
            fpr, tpr, _ = roc_curve(y, p)
            auc = roc_auc_score(y, p)
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"{name} (AUC {auc:.3f})",
                line=dict(color=model_colors[name], width=4 if name == model_name else 1.5),
                hovertemplate="FPR %{x:.3f}<br>TPR %{y:.3f}<extra>" + name + "</extra>"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                                 line=dict(color="grey", dash="dash", width=1)))
        fig.update_layout(**chart_layout, height=440,
                          xaxis=dict(title="False positive rate", gridcolor=grid),
                          yaxis=dict(title="True positive rate", gridcolor=grid),
                          legend=dict(x=0.55, y=0.05, bgcolor="rgba(27,32,72,0.85)"))
        st.plotly_chart(fig, width="stretch")
        st.caption("Hover over a curve to see the exact rates, click a legend entry to "
                   f"hide or show a model. The selected model (**{model_name}**) is "
                   "drawn thicker.")

# ---- tab 4: explore individual predictions ----
with tab4:
    p = probs[model_name]
    fig = go.Figure()
    for label, colour, name in [(0, bad_color, "Actual no"), (1, good_color, "Actual yes")]:
        fig.add_trace(go.Histogram(x=p[y.values == label], name=name, nbinsx=40,
                                   marker_color=colour, opacity=0.7))
    fig.add_vline(x=threshold, line_dash="dash", line_color="#f6b65c",
                  annotation_text=f"threshold {threshold:.2f}",
                  annotation_font_color="#f6b65c")
    fig.update_layout(**chart_layout, barmode="overlay", height=330,
                      title=f"Predicted probability of 'yes' · {model_name}",
                      xaxis=dict(title="Probability of yes", gridcolor=grid),
                      yaxis=dict(title="Clients", gridcolor=grid, type="log"),
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, width="stretch")
    st.caption("Move the threshold in the sidebar: lowering it catches more subscribers "
               "(higher recall) but adds false alarms (lower precision).")

    show = st.pills("Show rows", ["All", "Correct", "Wrong", "Missed subscribers"],
                    default="All")
    view_df = data.copy()
    view_df.insert(0, "prob_yes", p.round(4))
    view_df.insert(1, "predicted_y", np.where(pred == 1, "yes", "no"))
    if show == "Correct":
        view_df = view_df[pred == y.values]
    elif show == "Wrong":
        view_df = view_df[pred != y.values]
    elif show == "Missed subscribers":
        view_df = view_df[(pred == 0) & (y.values == 1)]
    st.caption(f"{len(view_df):,} rows")
    st.dataframe(view_df, width="stretch", height=320,
                 column_config={"prob_yes": st.column_config.ProgressColumn(
                     "prob_yes", min_value=0.0, max_value=1.0, format="%.3f")})
