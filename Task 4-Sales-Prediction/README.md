<div a ="center">

# 📈 Sales Prediction Using Machine Learning

### Predict Future Sales Using Artificial Intelligence & Machine Learning

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical_Computing-013243?style=for-the-badge&logo=numpy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)

**An interactive Machine Learning application that predicts product sales based on advertising investments across TV, Radio, and Newspaper marketing channels.**

---

⭐ **Interactive Dashboard** • 📊 **Data Visualization** • 🤖 **Machine Learning** • 🚀 **Real-Time Prediction**

</div>

---

# 📖 Overview

Sales Prediction Using Machine Learning is a regression-based predictive analytics project that estimates future product sales from advertising expenditures. The application analyzes historical marketing data, trains multiple regression models, evaluates their performance, and provides an interactive dashboard for making accurate sales predictions.

The project demonstrates the complete Machine Learning lifecycle, including data preprocessing, exploratory data analysis (EDA), model training, evaluation, and deployment through a modern Streamlit web application.

---

# 🎯 Objectives

- Predict future product sales accurately.
- Analyze the effectiveness of advertising channels.
- Compare multiple Machine Learning algorithms.
- Assist businesses in optimizing marketing budgets.
- Provide an intuitive web-based prediction system.
- Demonstrate a complete end-to-end ML workflow.

---

# ✨ Features

| 🚀 Feature | Description |
|------------|-------------|
| 📊 Interactive Dashboard | Modern Streamlit-based interface |
| 📁 Dataset Preview | View uploaded advertising dataset |
| 🧹 Data Cleaning | Missing value & duplicate handling |
| 📈 Exploratory Data Analysis | Interactive charts and visualizations |
| 🔥 Correlation Heatmap | Understand feature relationships |
| 🤖 Multiple ML Models | Train and compare regression algorithms |
| 🏆 Best Model Selection | Automatically selects the highest-performing model |
| 📊 Model Evaluation | MAE, RMSE, MSE & R² Score |
| 💾 Model Saving | Save trained model using Joblib |
| 🔮 Real-Time Prediction | Predict sales instantly |
| 📥 Export Predictions | Download prediction results |
| ⚠️ Input Validation | Prevent invalid user inputs |

---

# 🖥️ Dashboard Preview

> Run the app and add your own screenshots to the `screenshots/` folder to replace these placeholders.

| Home | Dashboard |
|------|-----------|
| ![](screenshots/home.png) | ![](screenshots/dashboard.png) |

| Prediction | Analytics |
|------------|-----------|
| ![](screenshots/prediction.png) | ![](screenshots/analytics.png) |

---

# 📂 Project Structure

```text
📦 Sales-Prediction
│
├── 📂 assets
│   ├── logo.png
│   ├── banner.png
│   └── background.png
│
├── 📂 dataset
│   └── advertising.csv        <-- add your dataset here
│
├── 📂 model
│   ├── sales_model.pkl         (generated after training)
│   └── scaler.pkl              (generated after training)
│
├── 📂 notebooks
│   └── Sales_Analysis.ipynb
│
├── 📂 reports
│   ├── model_report.txt        (generated after training)
│   └── prediction_history.csv  (generated after predictions)
│
├── 📂 screenshots
│   ├── home.png
│   ├── dashboard.png
│   ├── analytics.png
│   └── prediction.png
│
├── app.py
├── train_model.py
├── predict.py
├── utils.py
├── config.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

# 📊 Dataset

The project uses an advertising dataset containing marketing expenditure across different media channels. Place your `advertising.csv` file inside the `dataset/` folder.

| Feature | Description |
|----------|-------------|
| 📺 TV | Advertising Budget |
| 📻 Radio | Advertising Budget |
| 📰 Newspaper | Advertising Budget |
| 💰 Sales | Target Variable |

Column names are normalized automatically (e.g. `TV Ad Budget ($)` is mapped to `TV`), so common variations of the classic advertising dataset will work out of the box.

---

# ⚙️ Machine Learning Workflow

```text
                 SALES PREDICTION PIPELINE

                Advertising Dataset
                         │
                         ▼
                Data Preprocessing
                         │
                         ▼
          Exploratory Data Analysis (EDA)
                         │
                         ▼
                Feature Selection
                         │
                         ▼
                 Train/Test Split
                         │
                         ▼
         Train Multiple Regression Models
                         │
                         ▼
              Performance Comparison
                         │
                         ▼
                Select Best Model
                         │
                         ▼
               Save Trained Model (.pkl)
                         │
                         ▼
               Streamlit Web Dashboard
                         │
                         ▼
             Predict Future Product Sales
```

---

# 🤖 Machine Learning Models

The project compares multiple regression algorithms.

| Model | Purpose |
|--------|----------|
| 📈 Linear Regression | Baseline Model |
| 🌳 Decision Tree Regressor | Non-linear Prediction |
| 🌲 Random Forest Regressor | Ensemble Learning |

The application automatically selects the model with the best R² Score.

---

# 📊 Model Evaluation

Performance is measured using:

| Metric | Description |
|---------|-------------|
| R² Score | Goodness of Fit |
| MAE | Mean Absolute Error |
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |

---

# 📈 Sample Prediction

### Input

| Feature | Value |
|----------|------:|
| TV | 200 |
| Radio | 35 |
| Newspaper | 50 |

### Output

```text
Predicted Sales

20.53 Units
```

*(Actual value will depend on your dataset and the model selected during training.)*

---

# 📊 Visualizations Included

- 📉 Histogram
- 📈 Scatter Plot
- 🔥 Correlation Heatmap
- 📦 Box Plot
- 📊 Pair Plot
- 📉 Residual Plot
- 📈 Predicted vs Actual
- 🌳 Feature Importance

---

# 🛠️ Technologies Used

| Category | Technology |
|-----------|------------|
| Programming Language | Python |
| Machine Learning | Scikit-Learn |
| Data Analysis | Pandas |
| Numerical Computing | NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Web Framework | Streamlit |
| Model Storage | Joblib |
| IDE | Visual Studio Code |
| Version Control | Git & GitHub |

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/Sales-Prediction.git
cd Sales-Prediction
```

## Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Add Your Dataset

Place your `advertising.csv` file inside the `dataset/` folder (required columns: `TV`, `Radio`, `Newspaper`, `Sales`).

## Train the Model

```bash
python train_model.py
```

This trains Linear Regression, Decision Tree, and Random Forest models, compares them, and saves the best-performing model to `model/sales_model.pkl`.

## Run the Application

```bash
streamlit run app.py
```

The dashboard opens in your browser at `http://localhost:8501`. From there you can:
- Preview and clean the dataset
- Explore interactive EDA visualizations
- Train and compare models directly from the UI
- Make real-time predictions
- View and export prediction history

## Command-Line Prediction (optional)

```bash
python predict.py --tv 200 --radio 35 --newspaper 50
```

Or run it without arguments for interactive prompts.

---

# 📦 Python Libraries

```text
numpy
pandas
matplotlib
seaborn
plotly
streamlit
scikit-learn
joblib
```

---

# 🌟 Advantages

- Fast prediction
- Easy to use
- Professional dashboard
- Interactive visualization
- High prediction accuracy
- Lightweight application
- Beginner friendly
- Easy deployment

---

# ⚠️ Limitations

- Relies on historical advertising data.
- External market conditions are not included.
- Performance depends on data quality.
- Does not account for seasonal demand.

---

# 🔮 Future Scope

- Deep Learning Models
- Time Series Forecasting
- Real-Time Sales Dashboard
- Multi-Product Prediction
- Cloud Deployment
- Customer Segmentation
- Marketing ROI Analysis
- Business Intelligence Integration

---

# 🧪 Testing

- ✅ Unit Testing
- ✅ Integration Testing
- ✅ System Testing
- ✅ User Acceptance Testing

---

# 📌 Project Statistics

| Item | Value |
|------|-------|
| 🐍 Programming Language | Python |
| 🤖 ML Models | 3 |
| 📊 Visualizations | 8+ |
| 📂 Modules | 10+ |
| 💻 Framework | Streamlit |
| 📄 Documentation | Complete |

---

# 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

# 👨‍💻 Author

**Harsh Saini**

**🎓 BCA (Artificial Intelligence & Machine Learning)**

**💻 Machine Learning • Data Science • Python**

---

<div a="center">

## ⭐ If you found this project useful, consider giving it a Star!

### Made with ❤️ using Python, Streamlit & Scikit-Learn

</div>
