# 🚢 Titanic Survival Prediction

A **Streamlit web application** that performs exploratory data analysis and machine learning on the classic Titanic dataset to predict passenger survival.

---

## 📌 Overview

This project is part of a **Data Science Internship (Task 1)**. It demonstrates an end-to-end machine learning workflow — from data loading and visualization to model training and prediction — all wrapped in an interactive web UI powered by Streamlit.

---

## 🚀 Features

| Section | Description |
|---|---|
| 📊 **Dataset** | View the raw Titanic dataset in a sortable, filterable table |
| 📈 **Visualization** | Explore survival distribution with charts powered by Seaborn & Matplotlib |
| 🤖 **Train Model** | Train a Random Forest Classifier and view model accuracy |
| 🔮 **Prediction** | Overview of the prediction workflow |

---

## 🛠️ Tech Stack

- **Python 3.12+**
- **Streamlit** — Interactive web application framework
- **Pandas** — Data manipulation and analysis
- **Seaborn & Matplotlib** — Data visualization
- **Scikit-learn** — Machine learning (Random Forest Classifier)

---

## 📁 Project Structure

```
Titanic-Survival-Prediction/
│
├── app.py                  # Main Streamlit application
├── Titanic-Dataset.csv     # Dataset (not tracked by Git — see .gitignore)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Titanic-Survival-Prediction.git
cd Titanic-Survival-Prediction
```

### 2. Create a virtual environment (if needed)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Download the [Titanic dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset) from Kaggle and place the CSV file in the project root directory:

```
Titanic-Dataset.csv
```

### 5. Run the app

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 📊 Model Details

- **Algorithm**: Random Forest Classifier
- **Features Used**: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`
- **Preprocessing**:
  - Missing `Age` values filled with the **median**
  - Missing `Embarked` values filled with the **mode**
  - `Sex` encoded: `male → 0`, `female → 1`
  - `Embarked` encoded: `S → 0`, `C → 1`, `Q → 2`
- **Train/Test Split**: 80% / 20% (random_state=42)

---

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👤 Author

## Harsh Saini
**Bachelor of Computer Applications(Artificial Intelligence & Machine Learning)** 
**Data Science Intern**
*Built with ❤️ using Python & Streamlit*
