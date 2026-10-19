# evaluation module
"""
Model evaluation utilities: metrics, confusion matrix, ROC, feature importance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# ─────────────────── Metric Helpers ───────────────────────────────────────────

def compute_metrics(model, X_test, y_test, label_encoder) -> dict:
    """Return a dict of common classification metrics."""
    y_pred = model.predict(X_test)
    classes = label_encoder.classes_
    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall":    recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1":        f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "report":    classification_report(y_test, y_pred, target_names=classes),
        "y_pred":    y_pred,
    }


def all_model_metrics(trained_models: dict, X_test, y_test, label_encoder) -> pd.DataFrame:
    """Compute accuracy, precision, recall, F1 for every trained model."""
    rows = []
    for name, model in trained_models.items():
        m = compute_metrics(model, X_test, y_test, label_encoder)
        rows.append({
            "Model":     name,
            "Accuracy":  round(m["accuracy"], 4),
            "Precision": round(m["precision"], 4),
            "Recall":    round(m["recall"], 4),
            "F1 Score":  round(m["f1"], 4),
        })
    return pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)


# ─────────────────── Plotting Helpers ─────────────────────────────────────────

def plot_confusion_matrix(model, X_test, y_test, label_encoder):
    """Return a Plotly heatmap of the confusion matrix."""
    y_pred = model.predict(X_test)
    classes = label_encoder.classes_
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(
        cm,
        text_auto=True,
        x=classes, y=classes,
        color_continuous_scale="Blues",
        title="Confusion Matrix",
        labels={"x": "Predicted", "y": "Actual"},
    )
    fig.update_layout(
        title_font_size=18,
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        coloraxis_showscale=False,
        font=dict(family="Inter", size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_roc_curves(model, X_test, y_test, label_encoder):
    """Return a Plotly figure of one-vs-rest ROC curves."""
    classes = label_encoder.classes_
    n_classes = len(classes)
    y_bin = label_binarize(y_test, classes=list(range(n_classes)))

    # probability scores
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    else:
        y_score = label_binarize(model.predict(X_test), classes=list(range(n_classes)))

    colors = ["#6C63FF", "#FF6584", "#43E97B"]
    fig = go.Figure()
    for i, (cls, color) in enumerate(zip(classes, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{cls} (AUC={roc_auc:.3f})",
            line=dict(color=color, width=2.5),
        ))
    fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=1,
                  line=dict(dash="dash", color="gray"))
    fig.update_layout(
        title="ROC Curves (One-vs-Rest)",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        font=dict(family="Inter", size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def plot_feature_importance(model, model_name: str):
    """Return a Plotly bar chart of feature importances (if available)."""
    from src.preprocessing import FEATURE_COLUMNS
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).mean(axis=0)

    if importances is None:
        return None

    df = pd.DataFrame({
        "Feature":    FEATURE_COLUMNS,
        "Importance": importances,
    }).sort_values("Importance", ascending=True)

    fig = px.bar(
        df, x="Importance", y="Feature", orientation="h",
        title=f"Feature Importance — {model_name}",
        color="Importance",
        color_continuous_scale="Purples",
    )
    fig.update_layout(
        font=dict(family="Inter", size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
    )
    return fig


def plot_model_comparison(metrics_df: pd.DataFrame):
    """Return a grouped Plotly bar chart comparing models across metrics."""
    melt = metrics_df.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision", "Recall", "F1 Score"],
        var_name="Metric", value_name="Score",
    )
    fig = px.bar(
        melt, x="Model", y="Score", color="Metric",
        barmode="group",
        title="Model Comparison",
        color_discrete_sequence=["#6C63FF", "#FF6584", "#43E97B", "#FFB347"],
    )
    fig.update_layout(
        yaxis_range=[0, 1.05],
        font=dict(family="Inter", size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis_tickangle=-25,
    )
    return fig


def plot_pairplot_plotly(df: pd.DataFrame):
    """Return a Plotly scatter matrix (pair plot) of all features."""
    from src.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN
    fig = px.scatter_matrix(
        df,
        dimensions=FEATURE_COLUMNS,
        color=TARGET_COLUMN,
        color_discrete_sequence=["#6C63FF", "#FF6584", "#43E97B"],
        title="Feature Pair Plot",
    )
    fig.update_traces(diagonal_visible=False, showupperhalf=False)
    fig.update_layout(
        font=dict(family="Inter", size=11),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_distribution(df: pd.DataFrame, feature: str):
    """Return a Plotly violin+box plot for a single feature split by species."""
    from src.preprocessing import TARGET_COLUMN
    fig = px.violin(
        df, x=TARGET_COLUMN, y=feature, color=TARGET_COLUMN,
        box=True, points="all",
        color_discrete_sequence=["#6C63FF", "#FF6584", "#43E97B"],
        title=f"Distribution of {feature.replace('_', ' ').title()} by Species",
    )
    fig.update_layout(
        font=dict(family="Inter", size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig
