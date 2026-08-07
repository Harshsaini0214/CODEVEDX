"""
app.py  —  Iris Flower Classification · Streamlit Dashboard
────────────────────────────────────────────────────────────
Run with:  streamlit run app.py
"""

import sys
import os
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from src.preprocessing import preprocess_pipeline, FEATURE_COLUMNS
from src.model import train_all_models, get_best_model, MODELS
from src.evaluation import (
    all_model_metrics, compute_metrics,
    plot_confusion_matrix, plot_roc_curves,
    plot_feature_importance, plot_model_comparison,
    plot_pairplot_plotly, plot_distribution,
)
from src.utils import (
    save_model, save_artifact, load_model, load_artifact,
    model_exists, save_metrics_report, species_info, format_confidence,
    logger,
)

# ═══════════════════════════════ Page Config ══════════════════════════════════
st.set_page_config(
    page_title="Iris Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════ Custom CSS ════════════════════════════════════
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #6C63FF 0%, #a855f7 50%, #ec4899 100%);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 20px 60px rgba(108, 99, 255, 0.35);
}
.hero-banner h1 {
    color: white;
    font-size: 2.8rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero-banner p {
    color: rgba(255,255,255,0.85);
    font-size: 1.05rem;
    margin: 0.5rem 0 0;
}

/* ── Metric Cards ── */
.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 35px rgba(108,99,255,0.3);
}
.metric-card .metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #6C63FF;
}
.metric-card .metric-label {
    font-size: 0.82rem;
    color: rgba(255,255,255,0.55);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.25rem;
}

/* ── Section headers ── */
.section-title {
    font-size: 1.35rem;
    font-weight: 600;
    color: white;
    border-left: 4px solid #6C63FF;
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem;
}

/* ── Prediction Result Card ── */
.pred-card {
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin-top: 1rem;
    animation: fadeInUp 0.5s ease;
}
.pred-card-setosa     { background: linear-gradient(135deg,#6C63FF22,#6C63FF55); border: 2px solid #6C63FF; }
.pred-card-versicolor { background: linear-gradient(135deg,#FF658422,#FF658455); border: 2px solid #FF6584; }
.pred-card-virginica  { background: linear-gradient(135deg,#43E97B22,#43E97B55); border: 2px solid #43E97B; }
.pred-species {
    font-size: 2rem;
    font-weight: 700;
    color: white;
    margin: 0.5rem 0;
}
.pred-emoji { font-size: 3.5rem; }
.pred-conf  { font-size: 1rem; color: rgba(255,255,255,0.7); margin-top: 0.3rem; }

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Sidebar styling ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"]  { background: rgba(255,255,255,0.04); border-radius: 12px; padding: 4px; }
.stTabs [data-baseweb="tab"]       { border-radius: 8px; color: rgba(255,255,255,0.6) !important; }
.stTabs [aria-selected="true"]     { background: #6C63FF !important; color: white !important; }

/* ── Sliders & Inputs ── */
.stSlider > div > div { color: white !important; }

/* ── Dataframe ── */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* ── Info/Success boxes ── */
.stAlert { border-radius: 12px; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6C63FF, #a855f7);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 1.8rem;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.2s ease;
    box-shadow: 0 4px 15px rgba(108,99,255,0.4);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(108,99,255,0.5);
}

/* ── Plotly chart background ── */
.js-plotly-plot .plotly { border-radius: 16px; }

/* ── General text ── */
p, li, label { color: rgba(255,255,255,0.85) !important; }
h1,h2,h3,h4  { color: white !important; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════ Session State ════════════════════════════════════
if "trained"       not in st.session_state: st.session_state.trained        = False
if "trained_models" not in st.session_state: st.session_state.trained_models = {}
if "metrics_df"    not in st.session_state: st.session_state.metrics_df     = None
if "best_name"     not in st.session_state: st.session_state.best_name      = None
if "df"            not in st.session_state: st.session_state.df             = None
if "pipeline"      not in st.session_state: st.session_state.pipeline       = None


@st.cache_resource(show_spinner=False)
def get_pipeline():
    return preprocess_pipeline()


@st.cache_resource(show_spinner=False)
def get_trained_models(tune: bool):
    X_train, X_test, y_train, y_test, scaler, le, df = get_pipeline()
    models = train_all_models(X_train, y_train, tune=tune)
    return models


# ══════════════════════════════ Sidebar ═══════════════════════════════════════
with st.sidebar:
    st.markdown("### 🌸 Iris Classifier")
    st.markdown("---")

    st.markdown("**⚙️ Training Options**")
    tune = st.checkbox("Hyperparameter Tuning (GridSearchCV)", value=False)

    st.markdown("---")
    if st.button("🚀 Train Models", use_container_width=True):
        with st.spinner("Training all models …"):
            X_train, X_test, y_train, y_test, scaler, le, df = get_pipeline()
            trained_models = get_trained_models(tune)
            metrics_df = all_model_metrics(trained_models, X_test, y_test, le)
            best_name, best_model = get_best_model(trained_models, X_test, y_test)

            # Persist
            save_model(best_model, "best_model.pkl")
            save_artifact(scaler, "scaler.pkl")
            save_artifact(le,     "label_encoder.pkl")
            for nm, m in trained_models.items():
                save_model(m, nm.lower().replace(" ", "_") + ".pkl")
            save_metrics_report(metrics_df, best_name)

            st.session_state.trained        = True
            st.session_state.trained_models = trained_models
            st.session_state.metrics_df     = metrics_df
            st.session_state.best_name      = best_name
            st.session_state.df             = df

        st.success(f"✅ Done! Best: **{best_name}**")

    st.markdown("---")

    # Load from disk if already trained
    if not st.session_state.trained and model_exists():
        if st.button("📂 Load Saved Model", use_container_width=True):
            with st.spinner("Loading …"):
                _, X_test, _, y_test, scaler, le, df = get_pipeline()
                best_model = load_model("best_model.pkl")
                m = compute_metrics(best_model, X_test, y_test, le)
                st.session_state.trained   = True
                st.session_state.best_name = "Loaded from disk"
                st.session_state.df        = df
            st.success("Model loaded!")

    st.markdown("---")
    st.caption("Made with ❤️ using Streamlit & scikit-learn")


# ══════════════════════════════ Hero Banner ════════════════════════════════════
st.markdown("""
<div class="hero-banner">
  <h1>🌸 Iris Flower Classification</h1>
  <p>A complete machine learning pipeline — train, evaluate, and predict Iris species in real time</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════ Tabs ═════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["🔬 Predict", "📊 Analysis", "🤖 Model Evaluation", "📋 Dataset"])


# ─────────────────────────────── Tab 1: Predict ───────────────────────────────
with tab1:
    st.markdown('<div class="section-title">🌿 Single-Sample Prediction</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("#### Adjust Measurements")
        sl = st.slider("🌿 Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1, key="sl")
        sw = st.slider("🌿 Sepal Width  (cm)", 2.0, 4.5, 3.5, 0.1, key="sw")
        pl = st.slider("🌺 Petal Length (cm)", 1.0, 7.0, 1.4, 0.1, key="pl")
        pw = st.slider("🌺 Petal Width  (cm)", 0.1, 2.5, 0.2, 0.1, key="pw")

        st.markdown("")
        predict_btn = st.button("🔍 Predict Species", use_container_width=True)

    with col_r:
        if predict_btn:
            if not (st.session_state.trained or model_exists()):
                st.warning("⚠️ Please train or load a model first (sidebar).")
            else:
                try:
                    model  = load_model("best_model.pkl")
                    scaler = load_artifact("scaler.pkl")
                    le     = load_artifact("label_encoder.pkl")

                    X = np.array([[sl, sw, pl, pw]])
                    X_scaled = scaler.transform(X)

                    pred_idx = model.predict(X_scaled)[0]
                    species  = le.inverse_transform([pred_idx])[0]
                    proba    = model.predict_proba(X_scaled)[0] if hasattr(model, "predict_proba") else None
                    conf     = float(proba[pred_idx]) if proba is not None else None

                    info = species_info().get(species, {})
                    emoji = info.get("emoji", "🌸")
                    color_cls = species.lower().split("-")[-1]

                    st.markdown(f"""
                    <div class="pred-card pred-card-{color_cls}">
                      <div class="pred-emoji">{emoji}</div>
                      <div class="pred-species">{species}</div>
                      <div class="pred-conf">{format_confidence(conf) if conf else "Prediction complete"}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    if info.get("description"):
                        st.info(f"ℹ️ {info['description']}")

                    if proba is not None:
                        st.markdown("#### Probability Distribution")
                        import plotly.graph_objects as go
                        colors = {"Iris-setosa": "#6C63FF", "Iris-versicolor": "#FF6584", "Iris-virginica": "#43E97B"}
                        fig = go.Figure(go.Bar(
                            x=le.classes_,
                            y=proba,
                            marker_color=[colors.get(c, "#6C63FF") for c in le.classes_],
                            text=[f"{p:.1%}" for p in proba],
                            textposition="outside",
                        ))
                        fig.update_layout(
                            yaxis_range=[0, 1.1],
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(family="Inter", color="white"),
                            margin=dict(t=20),
                            showlegend=False,
                        )
                        fig.update_yaxes(tickformat=".0%", color="white", gridcolor="rgba(255,255,255,0.1)")
                        fig.update_xaxes(color="white")
                        st.plotly_chart(fig, use_container_width=True)

                except FileNotFoundError:
                    st.warning("⚠️ No saved model found. Please train first (sidebar → 🚀 Train Models).")
        else:
            st.markdown("""
            <div style="background:rgba(255,255,255,0.04);border-radius:16px;padding:2rem;text-align:center;margin-top:2rem;border:1px dashed rgba(255,255,255,0.15);">
              <div style="font-size:3rem;">🌸</div>
              <div style="color:rgba(255,255,255,0.5);margin-top:0.5rem;">
                Adjust sliders and click <strong>Predict Species</strong>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # Batch prediction
    st.markdown("---")
    st.markdown('<div class="section-title">📂 Batch Prediction (CSV Upload)</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a CSV with columns: sepal_length, sepal_width, petal_length, petal_width",
                                 type=["csv"])
    if uploaded is not None:
        if not (st.session_state.trained or model_exists()):
            st.warning("Please train/load a model first.")
        else:
            try:
                batch_df = pd.read_csv(uploaded)
                batch_df.columns = batch_df.columns.str.strip().str.lower()
                model  = load_model("best_model.pkl")
                scaler = load_artifact("scaler.pkl")
                le     = load_artifact("label_encoder.pkl")
                X_b    = batch_df[FEATURE_COLUMNS].values
                X_b_s  = scaler.transform(X_b)
                preds   = le.inverse_transform(model.predict(X_b_s))
                batch_df["Predicted Species"] = preds
                if hasattr(model, "predict_proba"):
                    probas = model.predict_proba(X_b_s)
                    for i, cls in enumerate(le.classes_):
                        batch_df[f"P({cls})"] = probas[:, i].round(4)
                st.dataframe(batch_df, use_container_width=True)
                csv = batch_df.to_csv(index=False).encode()
                st.download_button("⬇️ Download Results", csv, "predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Error: {e}")


# ─────────────────────────────── Tab 2: EDA ───────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    # Load data if not yet available
    if st.session_state.df is None:
        _, _, _, _, _, _, df_eda = get_pipeline()
    else:
        df_eda = st.session_state.df

    # Dataset summary metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><div class="metric-value">150</div><div class="metric-label">Total Samples</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="metric-value">4</div><div class="metric-label">Features</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="metric-value">3</div><div class="metric-label">Classes</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><div class="metric-value">0</div><div class="metric-label">Missing Values</div></div>', unsafe_allow_html=True)

    st.markdown("")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Feature Pair Plot")
        st.plotly_chart(plot_pairplot_plotly(df_eda), use_container_width=True)

    with col2:
        st.markdown("#### Feature Distribution")
        feat = st.selectbox("Select Feature", FEATURE_COLUMNS, key="eda_feat")
        st.plotly_chart(plot_distribution(df_eda, feat), use_container_width=True)

    # Correlation heatmap
    st.markdown("#### Correlation Heatmap")
    import plotly.express as px
    corr = df_eda[FEATURE_COLUMNS].corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Feature Correlation Matrix",
    )
    fig_corr.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="white", size=13),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Class balance
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### Class Distribution")
        vc = df_eda["species"].value_counts().reset_index()
        vc.columns = ["Species", "Count"]
        fig_pie = px.pie(vc, names="Species", values="Count",
                         color_discrete_sequence=["#6C63FF", "#FF6584", "#43E97B"],
                         hole=0.45)
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                              font=dict(family="Inter", color="white"))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col4:
        st.markdown("#### Descriptive Statistics")
        st.dataframe(df_eda[FEATURE_COLUMNS].describe().round(3), use_container_width=True)


# ─────────────────────────── Tab 3: Model Eval ────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">🤖 Model Performance</div>', unsafe_allow_html=True)

    if not st.session_state.trained:
        st.info("👈 Click **🚀 Train Models** in the sidebar to train and evaluate all classifiers.")
    else:
        metrics_df    = st.session_state.metrics_df
        trained_models = st.session_state.trained_models
        best_name     = st.session_state.best_name

        # Best model highlight
        best_row = metrics_df[metrics_df["Model"] == best_name]
        if not best_row.empty:
            acc = best_row["Accuracy"].values[0]
            f1  = best_row["F1 Score"].values[0]
        else:
            acc, f1 = 0, 0

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{acc:.1%}</div><div class="metric-label">Best Accuracy</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{f1:.1%}</div><div class="metric-label">Best F1 Score</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{len(trained_models)}</div><div class="metric-label">Models Trained</div></div>', unsafe_allow_html=True)

        st.markdown("")

        # Model comparison chart
        st.markdown("#### Model Comparison")
        st.plotly_chart(plot_model_comparison(metrics_df), use_container_width=True)

        # Metrics table
        st.markdown("#### Detailed Metrics")
        st.dataframe(metrics_df.style.highlight_max(
            subset=["Accuracy", "Precision", "Recall", "F1 Score"],
            color="#6C63FF44"
        ), use_container_width=True)

        # Per-model deep dive
        st.markdown("#### Model Deep Dive")
        _, X_test, _, y_test, _, le, _ = get_pipeline()
        selected_model_name = st.selectbox("Select Model", list(trained_models.keys()), key="deep_model")
        selected_model = trained_models[selected_model_name]

        col_cm, col_roc = st.columns(2)
        with col_cm:
            st.markdown("**Confusion Matrix**")
            st.plotly_chart(
                plot_confusion_matrix(selected_model, X_test, y_test, le),
                use_container_width=True
            )
        with col_roc:
            st.markdown("**ROC Curves**")
            st.plotly_chart(
                plot_roc_curves(selected_model, X_test, y_test, le),
                use_container_width=True
            )

        # Feature importance
        fi_fig = plot_feature_importance(selected_model, selected_model_name)
        if fi_fig:
            st.markdown("**Feature Importance**")
            st.plotly_chart(fi_fig, use_container_width=True)
        else:
            st.caption(f"ℹ️ {selected_model_name} does not expose feature importances.")

        # Classification report
        with st.expander("📄 Full Classification Report"):
            m = compute_metrics(selected_model, X_test, y_test, le)
            st.code(m["report"], language="text")


# ─────────────────────────── Tab 4: Dataset ───────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">📋 Raw Dataset</div>', unsafe_allow_html=True)

    if st.session_state.df is None:
        _, _, _, _, _, _, df_raw = get_pipeline()
    else:
        df_raw = st.session_state.df

    # Filters
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        species_filter = st.multiselect(
            "Filter by Species",
            options=df_raw["species"].unique().tolist(),
            default=df_raw["species"].unique().tolist(),
        )
    with col_f2:
        sl_range = st.slider(
            "Sepal Length Range",
            float(df_raw["sepal_length"].min()),
            float(df_raw["sepal_length"].max()),
            (float(df_raw["sepal_length"].min()), float(df_raw["sepal_length"].max())),
        )

    df_filtered = df_raw[
        (df_raw["species"].isin(species_filter)) &
        (df_raw["sepal_length"] >= sl_range[0]) &
        (df_raw["sepal_length"] <= sl_range[1])
    ]

    st.markdown(f"**Showing {len(df_filtered)} of {len(df_raw)} records**")
    st.dataframe(df_filtered.reset_index(drop=True), use_container_width=True, height=450)

    csv_raw = df_filtered.to_csv(index=False).encode()
    st.download_button("⬇️ Download Filtered Data", csv_raw, "iris_filtered.csv", "text/csv")
