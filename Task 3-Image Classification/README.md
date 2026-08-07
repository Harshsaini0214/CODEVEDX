# 🌸 Iris Flower Classification

A complete machine learning pipeline built with **scikit-learn** and **Streamlit** to classify Iris flowers into three species:
- *Iris setosa*
- *Iris versicolor*
- *Iris virginica*

---

## 📁 Project Structure

```
Iris-Flower-Classification/
├── app.py                   # Streamlit web application
├── train_model.py           # CLI training script
├── predict.py               # CLI prediction script
├── requirements.txt         # Python dependencies
├── dataset/
│   └── IRIS.csv             # Raw dataset (150 samples)
├── models/                  # Saved model artefacts (auto-generated)
├── reports/                 # Metrics reports (auto-generated)
├── src/
│   ├── preprocessing.py     # Data loading, encoding, scaling
│   ├── model.py             # Model registry (6 classifiers)
│   ├── evaluation.py        # Metrics & Plotly visualisations
│   └── utils.py             # Persistence, logging, helpers
└── notebooks/               # Jupyter notebooks (optional)
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train models (CLI)
```bash
python train_model.py             # basic training
python train_model.py --tune      # with GridSearchCV tuning
```

### 3. Launch the web app
```bash
streamlit run app.py
```

### 4. Predict from CLI
```bash
python predict.py --sl 5.1 --sw 3.5 --pl 1.4 --pw 0.2
python predict.py --interactive
```

---

## 🤖 Models Included

| Model | Notes |
|---|---|
| Logistic Regression | Baseline linear model |
| Decision Tree | Interpretable, tree-based |
| Random Forest | Ensemble, highest accuracy |
| SVM | Kernel-based, robust |
| K-Nearest Neighbors | Distance-based |
| Gradient Boosting | Boosting ensemble |

---

## 🖥️ App Features

- **🔬 Predict tab** — Interactive sliders for real-time single prediction + batch CSV upload
- **📊 Analysis tab** — Pair plots, violin plots, correlation heatmap, class distribution
- **🤖 Model Evaluation tab** — Model comparison, confusion matrix, ROC curves, feature importance
- **📋 Dataset tab** — Filtered view of the raw data with download

## 👤 Author

**Harsh Saini** 
**Data Science Intern**
*Built with ❤️ using Python & Streamlit*