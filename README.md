# 📊 Vendor Invoice Intelligence (VII)

### End-to-End Machine Learning System for Freight Cost Prediction & Invoice Risk Detection

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![DuckDB](https://img.shields.io/badge/DuckDB-FDC100?style=flat&logo=duckdb&logoColor=black)](https://duckdb.org/)

---

## 🏢 Business Problem

In large-scale enterprise resource planning, vendor invoice management is often manual, error-prone, and susceptible to fraud. This project addresses two major business friction points:

| Problem | Impact |
|---------|--------|
| **Unpredictable Freight Costs** | Logistics budgeting is guesswork, leading to variance in quarterly reports |
| **Manual Invoice Auditing** | Humans can't audit 10,000+ invoices monthly; fraud slips through |
| **Financial Leakage** | Overpayments and fraudulent invoices drain revenue silently |

---

## 💡 Solution

A dual-model ML system that:

1. **Predicts Freight Costs** - Regression model forecasting exact shipping costs
2. **Detects Risky Invoices** - Classification model flagging anomalies for audit

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Programming** | Python 3.9+ |
| **Machine Learning** | Scikit-Learn, Random Forest |
| **Data Processing** | Pandas, NumPy |
| **SQL Engine** | DuckDB |
| **Web App** | Streamlit |
| **Explainability** | SHAP |
| **Visualization** | Matplotlib, Seaborn, Plotly |

---

## 📂 Project Structure

```
freight_cost/
├── app/
│   └── streamlit_app.py          # Streamlit web application
├── data/
│   ├── raw_invoices.csv          # Raw invoice dataset
│   └── featured_invoices.csv     # Engineered features
├── models/
│   ├── freight_regressor.pkl     # Trained regression model
│   ├── risk_classifier.pkl       # Trained classification model
│   ├── preprocessor.pkl          # Data preprocessing pipeline
│   └── metrics.json              # Model evaluation metrics
├── outputs/
│   ├── shap_summary.png          # SHAP summary plot
│   └── shap_bar.png              # SHAP feature importance
├── src/
│   ├── generate_data.py          # Synthetic data generator
│   ├── database_manager.py       # SQL feature engineering
│   ├── data_preprocessing.py     # Data cleaning & validation
│   ├── train_model.py            # Model training pipeline
│   ├── model_explainability.py   # SHAP explanations
│   └── utils.py                  # Helper functions
├── Dockerfile                    # Container configuration
├── Makefile                      # Workflow automation
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Aditya2714/Freight_Cost.git
   cd Freight_Cost
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate synthetic data**
   ```bash
   python src/generate_data.py
   ```

4. **Run SQL feature engineering**
   ```bash
   python src/database_manager.py
   ```

5. **Preprocess data**
   ```bash
   python src/data_preprocessing.py
   ```

6. **Train models**
   ```bash
   python src/train_model.py
   ```

7. **Launch the app**
   ```bash
   streamlit run app/streamlit_app.py
   ```

### Using Makefile (Optional)

```bash
make all          # Run complete pipeline
make app          # Launch Streamlit app
make clean        # Clean generated files
```

---

## 📊 Model Performance

### Regression Model (Freight Cost Prediction)

| Metric | Value | Description |
|--------|-------|-------------|
| **R² Score** | 0.9991 | Model explains 99.9% of cost variation |
| **MAE** | $8,183 | Average prediction error |
| **RMSE** | $21,985 | Root mean squared error |

### Classification Model (Risk Detection)

| Metric | Value | Description |
|--------|-------|-------------|
| **Accuracy** | 99.3% | Overall correct predictions |
| **Precision** | 99.5% | Correct when predicting "risky" |
| **F1-Score** | 99.4% | Balance of precision & recall |

---

## 🖥️ Application Features

### 📊 Presentation Dashboard
- Executive summary with business impact metrics
- Monthly cost trends and vendor breakdown
- Risk analysis by vendor
- Cost waterfall chart

### 🏠 Dashboard
- Key performance indicators (KPIs)
- Freight cost distribution
- Risk distribution by carrier

### 📈 Predictions
- Interactive invoice input form
- Real-time freight cost prediction
- Risk probability assessment

### 🔍 Model Insights
- Model performance metrics
- Feature importance visualization
- SHAP explainability plots

### 📋 Data Explorer
- Filter data by vendor, carrier, risk level
- Export filtered data as CSV

---

## 🔍 How It Works

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Raw Invoice    │───▶│  SQL Feature     │───▶│  Preprocessing  │
│  Data           │    │  Engineering     │    │  & Validation   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                      │
                       ┌──────────────────┐           │
                       │  Streamlit App   │◀──────────┤
                       └──────────────────┘           │
                              ▲                       ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │  Trained ML      │◀───│  Model Training │
                       │  Models (.pkl)   │    │  & Evaluation   │
                       └──────────────────┘    └─────────────────┘
```

---

## 📈 Key Features

- **SQL-First Feature Engineering** - Uses DuckDB for professional SQL transformations
- **Dual-Model Architecture** - Regression for cost, Classification for risk
- **Explainable AI** - SHAP values for transparent decision-making
- **Interactive UI** - Streamlit dashboard for real-time predictions
- **Production-Ready** - Docker support and modular codebase

---

## 🎯 Business Impact

| Metric | Value |
|--------|-------|
| **Money Saved** | Prevents fraudulent overpayments |
| **Time Saved** | 354+ hours/month in manual review |
| **Efficiency Gain** | 42% reduction in audit workload |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Aditya Prakash**
- GitHub: [Aditya2714](https://github.com/Aditya2714)
- LinkedIn: [Connect with me](https://linkedin.com/in/your-profile)

---

## 🙏 Acknowledgments

- Built with Scikit-Learn, Streamlit, and DuckDB
- SHAP for model explainability
