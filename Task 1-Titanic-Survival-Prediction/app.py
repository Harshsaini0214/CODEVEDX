import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend – works in any environment

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05) !important;
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255,255,255,0.1);
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

/* Headings */
h1 { color: #ffffff !important; font-weight: 700 !important; letter-spacing: -0.5px; }
h2, h3 { color: #c9d1e0 !important; }

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] { color: #a0aec0 !important; font-size: 0.85rem !important; }
[data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.7rem !important; }

/* Success / Info / Warning */
.stSuccess { background: rgba(72,187,120,0.15) !important; border: 1px solid #48bb78 !important; color: #9ae6b4 !important; border-radius: 10px !important; }
.stInfo    { background: rgba(66,153,225,0.15) !important; border: 1px solid #4299e1 !important; color: #90cdf4 !important; border-radius: 10px !important; }
.stWarning { background: rgba(237,137,54,0.15) !important; border: 1px solid #ed8936 !important; color: #fbd38d !important; border-radius: 10px !important; }

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #667eea, #764ba2);
    color: white !important;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.5rem;
    font-weight: 600;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* Sliders & Selects label colour */
label { color: #c9d1e0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Titanic-Dataset.csv")
    return df


# ── Pre-processing (cached) ───────────────────────────────────────────────────
@st.cache_data
def preprocess(df: pd.DataFrame):
    data = df.copy()
    data["Age"] = data["Age"].fillna(data["Age"].median())
    data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])
    data["Fare"] = data["Fare"].fillna(data["Fare"].median())
    data["Sex"] = data["Sex"].map({"male": 0, "female": 1})
    data["Embarked"] = data["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    return data


# ── Model training (cached) ───────────────────────────────────────────────────
@st.cache_resource
def train_model(data: pd.DataFrame):
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    X = data[features]
    y = data["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return clf, X_test, y_test, preds, acc


# ── Load data once ────────────────────────────────────────────────────────────
df = load_data()
data = preprocess(df)
model, X_test, y_test, preds, acc = train_model(data)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🚢 Navigation")
menu = st.sidebar.radio(
    "Go to",
    ["🏠 Overview", "📊 Dataset", "📈 Visualization", "🤖 Model", "🔮 Prediction"],
)

st.title("🚢 Titanic Survival Prediction")
st.markdown("---")


# ═════════════════════════════════════════════════════════════════════════════
# 1. OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if menu == "🏠 Overview":
    col1, col2, col3, col4 = st.columns(4)
    total = len(df)
    survived = int(df["Survived"].sum())
    died = total - survived
    survival_rate = survived / total * 100

    col1.metric("Total Passengers", f"{total:,}")
    col2.metric("Survived", f"{survived:,}")
    col3.metric("Did Not Survive", f"{died:,}")
    col4.metric("Survival Rate", f"{survival_rate:.1f}%")

    st.markdown("### About the App")
    st.info(
        "This app trains a **Random Forest** classifier on the Titanic dataset "
        "to predict whether a passenger would have survived. "
        "Navigate using the sidebar to explore the data, visualisations, model metrics, "
        "or make your own prediction."
    )

    st.markdown("### Model Performance at a Glance")
    st.metric("Current Model Accuracy", f"{acc * 100:.2f}%")


# ═════════════════════════════════════════════════════════════════════════════
# 2. DATASET
# ═════════════════════════════════════════════════════════════════════════════
elif menu == "📊 Dataset":
    st.subheader("📊 Raw Dataset")
    st.write(f"Shape: **{df.shape[0]} rows × {df.shape[1]} columns**")
    st.dataframe(df, use_container_width=True)

    st.subheader("📋 Summary Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("❓ Missing Values")
    missing = df.isnull().sum()
    missing = missing[missing > 0].reset_index()
    missing.columns = ["Column", "Missing Count"]
    missing["Missing %"] = (missing["Missing Count"] / len(df) * 100).round(2)
    if missing.empty:
        st.success("No missing values found!")
    else:
        st.dataframe(missing, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# 3. VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════
elif menu == "📈 Visualization":
    st.subheader("📈 Data Visualizations")

    plot_bg = "#1a1a2e"
    text_col = "#c9d1e0"
    palette = ["#e74c3c", "#2ecc71"]

    def style_fig(fig, ax_or_axes):
        """Apply dark theme to a figure."""
        fig.patch.set_facecolor(plot_bg)
        axes = ax_or_axes if isinstance(ax_or_axes, (list, np.ndarray)) else [ax_or_axes]
        for ax in np.array(axes).flatten():
            ax.set_facecolor(plot_bg)
            ax.tick_params(colors=text_col)
            ax.xaxis.label.set_color(text_col)
            ax.yaxis.label.set_color(text_col)
            ax.title.set_color(text_col)
            for spine in ax.spines.values():
                # matplotlib does not accept CSS rgba() strings – use an RGBA tuple instead
                spine.set_edgecolor((1.0, 1.0, 1.0, 0.1))

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        # hue parameter avoids the FutureWarning in seaborn ≥ 0.13
        sns.countplot(data=df, x="Survived", hue="Survived",
                      palette=palette, legend=False, ax=ax)
        ax.set_title("Survival Count")
        # Use FixedLocator before set_xticklabels to avoid UserWarning
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Did Not Survive", "Survived"])
        style_fig(fig, ax)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x="Pclass", hue="Survived",
                      palette=palette, ax=ax)
        ax.set_title("Survival by Passenger Class")
        leg = ax.legend(["Did Not Survive", "Survived"],
                        facecolor=plot_bg, edgecolor="gray")
        for text in leg.get_texts():
            text.set_color(text_col)
        style_fig(fig, ax)
        st.pyplot(fig)
        plt.close(fig)

    col3, col4 = st.columns(2)

    with col3:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data=df, x="Age", hue="Survived", bins=30,
                     palette=palette, kde=True, ax=ax)
        ax.set_title("Age Distribution by Survival")
        leg = ax.legend(["Did Not Survive", "Survived"],
                        facecolor=plot_bg, edgecolor="gray")
        for text in leg.get_texts():
            text.set_color(text_col)
        style_fig(fig, ax)
        st.pyplot(fig)
        plt.close(fig)

    with col4:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x="Sex", hue="Survived",
                      palette=palette, ax=ax)
        ax.set_title("Survival by Gender")
        leg = ax.legend(["Did Not Survive", "Survived"],
                        facecolor=plot_bg, edgecolor="gray")
        for text in leg.get_texts():
            text.set_color(text_col)
        style_fig(fig, ax)
        st.pyplot(fig)
        plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# 4. MODEL
# ═════════════════════════════════════════════════════════════════════════════
elif menu == "🤖 Model":
    st.subheader("🤖 Model Training & Evaluation")
    st.success(f"✅ Model Accuracy: **{acc * 100:.2f}%**")

    st.markdown("#### Classification Report")
    report = classification_report(y_test, preds, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    # Format only numeric columns; na_rep handles any NaN cells gracefully
    st.dataframe(
        report_df.style.format("{:.2f}", na_rep="-"),
        use_container_width=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Did Not Survive", "Survived"],
                    yticklabels=["Did Not Survive", "Survived"], ax=ax)
        ax.set_title("Confusion Matrix", color="#c9d1e0")
        ax.tick_params(colors="#c9d1e0")
        ax.xaxis.label.set_color("#c9d1e0")
        ax.yaxis.label.set_color("#c9d1e0")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.markdown("#### Feature Importance")
        features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
        importances = pd.Series(model.feature_importances_, index=features).sort_values()
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        importances.plot(kind="barh", ax=ax,
                         color=["#667eea"] * len(importances))
        ax.set_title("Feature Importances", color="#c9d1e0")
        ax.tick_params(colors="#c9d1e0")
        ax.xaxis.label.set_color("#c9d1e0")
        for spine in ax.spines.values():
            # matplotlib does not accept CSS rgba() strings – use an RGBA tuple instead
            spine.set_edgecolor((0.0, 0.0, 0.0, 0.0))
        st.pyplot(fig)
        plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# 5. PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
else:
    st.subheader("🔮 Predict Survival")
    st.markdown("Fill in the passenger details below and click **Predict** to see the result.")

    col1, col2, col3 = st.columns(3)

    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3],
                              help="1 = First, 2 = Second, 3 = Third")
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.slider("Age", min_value=1, max_value=80, value=28)

    with col2:
        sibsp = st.number_input("Siblings / Spouses Aboard", min_value=0, max_value=8, value=0)
        parch = st.number_input("Parents / Children Aboard", min_value=0, max_value=6, value=0)

    with col3:
        fare = st.number_input("Fare Paid (£)", min_value=0.0, max_value=600.0, value=32.0, step=0.5)
        embarked = st.selectbox("Port of Embarkation",
                                ["S – Southampton", "C – Cherbourg", "Q – Queenstown"])

    # Encode inputs
    sex_enc = 0 if sex == "male" else 1
    embarked_enc = {"S – Southampton": 0, "C – Cherbourg": 1, "Q – Queenstown": 2}[embarked]

    if st.button("🔮 Predict Survival"):
        input_df = pd.DataFrame([[pclass, sex_enc, age, int(sibsp), int(parch), fare, embarked_enc]],
                                columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"])
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        if prediction == 1:
            st.success(f"🎉 **Survived!** The model predicts this passenger would have survived "
                       f"with a confidence of **{probability[1] * 100:.1f}%**.")
        else:
            st.error(f"💀 **Did Not Survive.** The model predicts this passenger would not have survived "
                     f"(confidence: **{probability[0] * 100:.1f}%**).")

        col_a, col_b = st.columns(2)
        col_a.metric("Survival Probability", f"{probability[1] * 100:.1f}%")
        col_b.metric("Non-Survival Probability", f"{probability[0] * 100:.1f}%")
