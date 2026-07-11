# 📊 Telco Customer Churn Prediction

🔗 **[Try the Live App](https://telco-customer-churn-prediction-myfirstproject.streamlit.app)**

A machine learning project that predicts customer churn for a telecom company using Random Forest and XGBoost — with a strong focus on identifying and eliminating **data leakage** to produce a realistic, production-honest model.

## 🎯 Project Overview

Customer churn (customers leaving a service) is one of the most costly problems telecom companies face. This project builds a classification model to predict which customers are likely to churn, based on their service usage, contract details, and demographics — enabling proactive retention strategies.

## ⚠️ The Data Leakage Story (Key Highlight)

The first version of this model achieved **95%+ accuracy** — which, for a real-world churn dataset, is a red flag rather than a win.

By analyzing feature importance, I discovered that a **`Satisfaction Score`** column was responsible for over 40% of the model's predictive power. This field is recorded *after* a customer has effectively already decided to churn — meaning the model was leaking future information rather than learning genuine predictive patterns.

After removing `Satisfaction Score` along with other leaked features (`Churn Score`, `CLTV`), the model's performance dropped to a much more realistic and trustworthy **85.3% accuracy**, confirming the fix worked as expected.

## 📈 Final Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Random Forest | 83.04% | 90.11% |
| **XGBoost (Best)** | **85.31%** | **90.98%** |

## 🔍 Key Business Insights

- **Contract type** is the strongest churn driver — month-to-month customers churn far more than those on 1–2 year contracts
- **Fiber optic internet** users show higher churn rates than DSL/Cable users
- **Referrals & automatic payment methods** correlate with higher customer loyalty
- Customers with **longer tenure** are significantly less likely to churn

## 🛠️ Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Modeling:** Scikit-learn, XGBoost
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Google Colab

## 📁 Project Structure

```
telco-customer-churn-prediction/
├── README.md
├── Telco_Customer_Churn.ipynb     # Full analysis notebook
├── images/
│   ├── business_insights.png      # EDA visualizations
│   └── model_performance.png      # ROC curve, confusion matrix, feature importance
├── model/
│   ├── telco_churn_model.pkl      # Trained XGBoost model
│   └── model_columns.pkl          # Feature column order for inference
└── requirements.txt
```

## 🔄 Methodology

1. **Data Cleaning** — Handled missing values (`Offer`, `Internet Type`), validated numeric data types
2. **Leakage & Irrelevant Feature Removal** — Dropped `Churn Score`, `CLTV`, `Satisfaction Score` (leakage) and `Customer ID`, `City`, `Zip Code`, etc. (non-predictive)
3. **Encoding** — Binary mapping for Yes/No features, one-hot encoding for multi-class categoricals
4. **Train-Test Split** — 80/20 stratified split to preserve churn ratio
5. **Model Training** — Compared Random Forest vs. XGBoost
6. **Evaluation** — Accuracy, ROC-AUC, confusion matrix, feature importance analysis

## 📊 Visualizations

**Business Insights**
![Business Insights](images/business_insights.png)

**Model Performance**
![Model Performance](images/model_performance.png)

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/telco-customer-churn-prediction.git
cd telco-customer-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Open the notebook
jupyter notebook Telco_Customer_Churn.ipynb
```

## 📌 Dataset

[IBM Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 50 original features.

## 👤 Author

**Fahad Shabbir** — BS Data Science student | Data Scientist | AI & ML Engineer 

---
*Built as part of ongoing Data Science coursework, with a focus on rigorous, leakage-free ML practices.*
