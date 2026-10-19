"""
app.py
------
Streamlit web application for Sales Prediction Using Machine Learning.

Pages:
- Home
- Dataset Preview
- Exploratory Data Analysis
- Model Training & Evaluation
- Real-Time Prediction
- Prediction History

Run:
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import config
import utils

# ====================================================================
# PAGE CONFIG
# ====================================================================
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout=config.PAGE_LAYOUT,
    initial_sidebar_state="expanded",
)

# ====================================================================
# CUSTOM CSS
# ====================================================================
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #FF4B4B, #F7931E);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 0.2rem;
    }
    .sub-header {
        text-align: center;
        color: #6c757d;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 1.1rem;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        font-weight: 600;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
    }
    .stButton>button:hover {
        background-color: #d63d3d;
        color: white;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ====================================================================
# SESSION STATE INIT
# ====================================================================
if "trained_results" not in st.session_state:
    st.session_state.trained_results = None
if "trained_models" not in st.session_state:
    st.session_state.trained_models = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "best_model_name" not in st.session_state:
    st.session_state.best_model_name = None
if "df" not in st.session_state:
    st.session_state.df = None


# ====================================================================
# DATA LOADING (cached)
# ====================================================================
@st.cache_data(show_spinner=False)
def get_dataset():
    df = utils.load_dataset()
    df, removed = utils.clean_dataset(df)
    return df, removed


def dataset_available():
    return os.path.exists(config.DATASET_PATH)


def model_available():
    return os.path.exists(config.MODEL_PATH) and os.path.exists(config.SCALER_PATH)


# ====================================================================
# SIDEBAR NAVIGATION
# ====================================================================
st.sidebar.markdown("## 📈 Sales Prediction")
st.sidebar.markdown("Navigate through the app")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📁 Dataset Preview",
        "📈 Exploratory Data Analysis",
        "🤖 Model Training & Evaluation",
        "🔮 Real-Time Prediction",
        "📥 Prediction History",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info(
    "This app predicts product sales based on advertising spend "
    "across TV, Radio, and Newspaper channels using Machine Learning."
)
st.sidebar.markdown("**Author:** Harsh Saini")
st.sidebar.markdown("BCA (Artificial Intelligence & Machine Learning)")
st.sidebar.markdown("Data Science Intern")


# ====================================================================
# HOME PAGE
# ====================================================================
if page == "🏠 Home":
    st.markdown(f'<div class="main-header">{config.APP_TITLE}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Predict Future Sales Using Artificial Intelligence & Machine Learning</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            '<div class="metric-card">⭐<br><b style="color:black">Interactive Dashboard</b></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="metric-card">📊<br><b style="color:black">Data Visualization</b></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="metric-card">🤖<br><b style="color:black">Machine Learning</b></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            '<div class="metric-card">🚀<br><b style="color:black">Real-Time Prediction</b></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.markdown("### 📖 Overview")
    st.write(
        "Sales Prediction Using Machine Learning is a regression-based predictive "
        "analytics application that estimates future product sales from advertising "
        "expenditures. The application analyzes historical marketing data, trains "
        "multiple regression models, evaluates their performance, and provides an "
        "interactive dashboard for making accurate sales predictions."
    )

    st.markdown("### 🎯 Objectives")
    obj_col1, obj_col2 = st.columns(2)
    with obj_col1:
        st.markdown(
            """
            - Predict future product sales accurately
            - Analyze the effectiveness of advertising channels
            - Compare multiple Machine Learning algorithms
            """
        )
    with obj_col2:
        st.markdown(
            """
            - Assist businesses in optimizing marketing budgets
            - Provide an intuitive web-based prediction system
            - Demonstrate a complete end-to-end ML workflow
            """
        )

    st.markdown("### ✨ Features")
    feat_df = pd.DataFrame(
        {
            "Feature": [
                "📊 Interactive Dashboard",
                "📁 Dataset Preview",
                "🧹 Data Cleaning",
                "📈 Exploratory Data Analysis",
                "🔥 Correlation Heatmap",
                "🤖 Multiple ML Models",
                "🏆 Best Model Selection",
                "📊 Model Evaluation",
                "💾 Model Saving",
                "🔮 Real-Time Prediction",
                "📥 Export Predictions",
                "⚠️ Input Validation",
            ],
            "Description": [
                "Modern Streamlit-based interface",
                "View uploaded advertising dataset",
                "Missing value & duplicate handling",
                "Interactive charts and visualizations",
                "Understand feature relationships",
                "Train and compare regression algorithms",
                "Automatically selects the highest-performing model",
                "MAE, RMSE, MSE & R² Score",
                "Save trained model using Joblib",
                "Predict sales instantly",
                "Download prediction results",
                "Prevent invalid user inputs",
            ],
        }
    )
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    if not dataset_available():
        st.warning(
            f"⚠️ No dataset found at `{config.DATASET_PATH}`. "
            f"Please add your `advertising.csv` file to the `dataset/` folder to get started."
        )
    else:
        st.success("✅ Dataset detected! Head to **Dataset Preview** or **Model Training** to continue.")


# ====================================================================
# DATASET PREVIEW PAGE
# ====================================================================
elif page == "📁 Dataset Preview":
    st.markdown("## 📁 Dataset Preview")

    if not dataset_available():
        st.error(
            f"Dataset not found at `{config.DATASET_PATH}`. "
            f"Please place your `advertising.csv` file inside the `dataset/` folder."
        )
        st.stop()

    with st.spinner("Loading dataset..."):
        try:
            df, removed = get_dataset()
            st.session_state.df = df
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            st.stop()

    st.success(f"Dataset loaded successfully! {removed} duplicate/invalid row(s) removed during cleaning.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", len(df))
    col2.metric("Total Columns", len(df.columns))
    col3.metric("Missing Values", int(df.isnull().sum().sum()))
    col4.metric("Duplicate Rows Removed", removed)

    st.markdown("### 🔍 Data Sample")
    n_rows = st.slider("Rows to display", 5, min(100, len(df)), 10)
    st.dataframe(df.head(n_rows), use_container_width=True)

    st.markdown("### 📊 Statistical Summary")
    st.dataframe(df.describe().T, use_container_width=True)

    st.markdown("### 🧬 Data Types")
    dtype_df = pd.DataFrame({"Column": df.dtypes.index, "Type": df.dtypes.values.astype(str)})
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Cleaned Dataset",
        data=csv,
        file_name="cleaned_advertising.csv",
        mime="text/csv",
    )


# ====================================================================
# EDA PAGE
# ====================================================================
elif page == "📈 Exploratory Data Analysis":
    st.markdown("## 📈 Exploratory Data Analysis")

    if not dataset_available():
        st.error(
            f"Dataset not found at `{config.DATASET_PATH}`. "
            f"Please place your `advertising.csv` file inside the `dataset/` folder."
        )
        st.stop()

    try:
        df, _ = get_dataset()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📉 Histograms", "📈 Scatter Plots", "🔥 Correlation Heatmap", "📦 Box Plots", "📊 Pair Plot"]
    )

    with tab1:
        st.markdown("#### Distribution of Features")
        col = st.selectbox("Select a column", df.columns, key="hist_col")
        fig = px.histogram(df, x=col, nbins=30, marginal="box", color_discrete_sequence=["#FF4B4B"])
        fig.update_layout(bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("#### Relationship with Sales")
        x_col = st.selectbox("Select feature (X-axis)", config.FEATURE_COLUMNS, key="scatter_col")

        # Build scatter plot without statsmodels dependency
        fig = px.scatter(
            df,
            x=x_col,
            y=config.TARGET_COLUMN,
            color_discrete_sequence=["#F7931E"],
            title=f"{x_col} vs {config.TARGET_COLUMN}",
        )

        # Add OLS trendline manually using numpy (no statsmodels required)
        x_vals = df[x_col].dropna().values.astype(float)
        y_vals = df.loc[df[x_col].notna(), config.TARGET_COLUMN].values.astype(float)
        if len(x_vals) > 1:
            coeffs = np.polyfit(x_vals, y_vals, 1)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
            y_line = np.polyval(coeffs, x_line)
            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name="OLS Trendline",
                    line=dict(color="#FF4B4B", width=2, dash="dash"),
                )
            )

        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("#### Correlation Heatmap")
        corr = df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, annot=True, cmap="Reds", fmt=".2f", ax=ax, linewidths=0.5)
        st.pyplot(fig)

    with tab4:
        st.markdown("#### Box Plots (Outlier Detection)")
        col = st.selectbox("Select a column", df.columns, key="box_col")
        fig = px.box(df, y=col, color_discrete_sequence=["#150458"])
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.markdown("#### Pair Plot")
        st.caption("Shows pairwise relationships between all features (may take a moment).")
        if st.button("Generate Pair Plot"):
            with st.spinner("Generating pair plot..."):
                fig = sns.pairplot(df, diag_kind="kde", corner=True, palette="flare")
                st.pyplot(fig)


# ====================================================================
# MODEL TRAINING PAGE
# ====================================================================
elif page == "🤖 Model Training & Evaluation":
    st.markdown("## 🤖 Model Training & Evaluation")

    if not dataset_available():
        st.error(
            f"Dataset not found at `{config.DATASET_PATH}`. "
            f"Please place your `advertising.csv` file inside the `dataset/` folder."
        )
        st.stop()

    try:
        df, removed = get_dataset()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

    st.info(f"Dataset ready: {len(df)} rows (after removing {removed} duplicate/invalid rows).")

    col_a, col_b = st.columns([1, 3])
    with col_a:
        test_size = st.slider("Test set size", 0.1, 0.4, config.TEST_SIZE, 0.05)
        train_btn = st.button("🚀 Train Models")

    if train_btn:
        with st.spinner("Training models, please wait..."):
            X = df[config.FEATURE_COLUMNS]
            y = df[config.TARGET_COLUMN]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=config.RANDOM_STATE
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree Regressor": DecisionTreeRegressor(random_state=config.RANDOM_STATE),
                "Random Forest Regressor": RandomForestRegressor(
                    n_estimators=200, random_state=config.RANDOM_STATE
                ),
            }

            results = {}
            trained_models = {}
            predictions = {}

            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                metrics = utils.evaluate_model(y_test, y_pred)
                results[name] = metrics
                trained_models[name] = model
                predictions[name] = y_pred

            best_model_name = max(results, key=lambda k: results[k]["R2 Score"])

            st.session_state.trained_results = results
            st.session_state.trained_models = trained_models
            st.session_state.scaler = scaler
            st.session_state.best_model_name = best_model_name
            st.session_state.y_test = y_test
            st.session_state.predictions = predictions
            st.session_state.X_train_scaled = X_train_scaled

            # Save best model to disk
            utils.save_artifact(trained_models[best_model_name], config.MODEL_PATH)
            utils.save_artifact(scaler, config.SCALER_PATH)
            utils.save_artifact(
                best_model_name, config.MODEL_PATH.replace(".pkl", "_name.pkl")
            )

            from train_model import write_report

            write_report(results, best_model_name, len(df), removed)

        st.success(f"✅ Training complete! Best model: **{best_model_name}**")

    if st.session_state.trained_results:
        results = st.session_state.trained_results
        best_model_name = st.session_state.best_model_name

        st.markdown("### 🏆 Best Model")
        st.success(f"**{best_model_name}** — R² Score: {results[best_model_name]['R2 Score']}")

        st.markdown("### 📊 Model Comparison")
        results_df = pd.DataFrame(results).T
        results_df.insert(0, "Model", results_df.index)
        results_df = results_df.reset_index(drop=True)

        def highlight_best(row):
            color = "background-color: #d4f7d4" if row["Model"] == best_model_name else ""
            return [color] * len(row)

        st.dataframe(
            results_df.style.apply(highlight_best, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### 📉 R² Score Comparison")
        fig = px.bar(
            results_df,
            x="Model",
            y="R2 Score",
            color="Model",
            text="R2 Score",
            color_discrete_sequence=["#FF4B4B", "#F7931E", "#150458"],
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🌳 Feature Importance (Tree-based Models)")
        tree_models = {
            k: v
            for k, v in st.session_state.trained_models.items()
            if hasattr(v, "feature_importances_")
        }
        if tree_models:
            selected_tree = st.selectbox("Select model", list(tree_models.keys()))
            importances = tree_models[selected_tree].feature_importances_
            imp_df = pd.DataFrame(
                {"Feature": config.FEATURE_COLUMNS, "Importance": importances}
            ).sort_values("Importance", ascending=True)
            fig = px.bar(
                imp_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color_discrete_sequence=["#F7931E"],
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📈 Predicted vs Actual")
        selected_model_pva = st.selectbox(
            "Select model for Predicted vs Actual", list(results.keys()), key="pva_model"
        )
        y_test = st.session_state.y_test
        y_pred = st.session_state.predictions[selected_model_pva]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=y_test, y=y_pred, mode="markers", name="Predictions",
                       marker=dict(color="#FF4B4B", size=8, opacity=0.7))
        )
        min_val, max_val = float(min(y_test.min(), y_pred.min())), float(max(y_test.max(), y_pred.max()))
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode="lines",
                       name="Perfect Prediction", line=dict(color="black", dash="dash"))
        )
        fig.update_layout(xaxis_title="Actual Sales", yaxis_title="Predicted Sales")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📉 Residual Plot")
        residuals = y_test - y_pred
        fig = px.scatter(
            x=y_pred, y=residuals,
            labels={"x": "Predicted Sales", "y": "Residuals"},
            color_discrete_sequence=["#150458"],
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

        with open(config.MODEL_REPORT_PATH, "r") as f:
            report_text = f.read()
        st.download_button(
            "📄 Download Model Report", data=report_text,
            file_name="model_report.txt", mime="text/plain"
        )
    else:
        if model_available():
            st.info("A previously trained model was found on disk. Click **Train Models** above to retrain, "
                     "or go to **Real-Time Prediction** to use the existing model.")
        else:
            st.info("Click **Train Models** to begin training.")


# ====================================================================
# REAL-TIME PREDICTION PAGE
# ====================================================================
elif page == "🔮 Real-Time Prediction":
    st.markdown("## 🔮 Real-Time Sales Prediction")

    if not model_available():
        st.warning(
            "⚠️ No trained model found. Please go to **Model Training & Evaluation** "
            "and click **Train Models** first."
        )
        st.stop()

    model = utils.load_artifact(config.MODEL_PATH)
    scaler = utils.load_artifact(config.SCALER_PATH)
    try:
        model_name = utils.load_artifact(config.MODEL_PATH.replace(".pkl", "_name.pkl"))
    except FileNotFoundError:
        model_name = "Trained Model"

    st.info(f"Using model: **{model_name}**")

    col1, col2, col3 = st.columns(3)
    with col1:
        tv = st.number_input("📺 TV Advertising Budget", min_value=0.0, max_value=1000.0, value=150.0, step=1.0)
    with col2:
        radio = st.number_input("📻 Radio Advertising Budget", min_value=0.0, max_value=1000.0, value=25.0, step=1.0)
    with col3:
        newspaper = st.number_input("📰 Newspaper Advertising Budget", min_value=0.0, max_value=1000.0, value=30.0, step=1.0)

    if st.button("🔮 Predict Sales"):
        errors = utils.validate_inputs(tv, radio, newspaper)
        if errors:
            for e in errors:
                st.error(e)
        else:
            X = pd.DataFrame([[tv, radio, newspaper]], columns=config.FEATURE_COLUMNS)
            X_scaled = scaler.transform(X)
            prediction = model.predict(X_scaled)[0]

            st.markdown("### 📊 Prediction Result")
            st.markdown(
                f"""
                <div style="background: linear-gradient(90deg, #FF4B4B, #F7931E);
                            padding: 2rem; border-radius: 16px; text-align: center;">
                    <h2 style="color:white; margin:0;">Predicted Sales</h2>
                    <h1 style="color:white; font-size:3rem; margin:0;">{prediction:.2f} Units</h1>
                </div>
                """,
                unsafe_allow_html=True,
            )

            utils.log_prediction(tv, radio, newspaper, prediction, model_name)
            st.success("✅ Prediction logged to history.")

            st.markdown("#### Input Summary")
            fig = px.bar(
                x=["TV", "Radio", "Newspaper"], y=[tv, radio, newspaper],
                labels={"x": "Channel", "y": "Budget"},
                color=["TV", "Radio", "Newspaper"],
                color_discrete_sequence=["#FF4B4B", "#F7931E", "#150458"],
            )
            st.plotly_chart(fig, use_container_width=True)


# ====================================================================
# PREDICTION HISTORY PAGE
# ====================================================================
elif page == "📥 Prediction History":
    st.markdown("## 📥 Prediction History")

    history = utils.load_prediction_history()

    if history.empty:
        st.info("No predictions made yet. Go to **Real-Time Prediction** to make your first prediction.")
    else:
        st.dataframe(history.sort_values("Timestamp", ascending=False), use_container_width=True, hide_index=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Predictions", len(history))
        col2.metric("Average Predicted Sales", f"{history['Predicted_Sales'].mean():.2f}")
        col3.metric("Max Predicted Sales", f"{history['Predicted_Sales'].max():.2f}")

        fig = px.line(
            history, y="Predicted_Sales", markers=True,
            title="Prediction History Trend",
            color_discrete_sequence=["#FF4B4B"],
        )
        st.plotly_chart(fig, use_container_width=True)

        csv = history.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Prediction History", data=csv,
            file_name="prediction_history.csv", mime="text/csv"
        )

        if st.button("🗑️ Clear History"):
            os.remove(config.PREDICTION_HISTORY_PATH)
            st.success("History cleared. Refresh the page to see changes.")
