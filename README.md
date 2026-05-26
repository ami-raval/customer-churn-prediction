# 🔍 Customer Churn Prediction Engine

> Predicting customer churn using XGBoost with SHAP explainability on 80K+ telecom records

## 🚀 Live Demo
👉 **[Try the Live App Here](https://ami-churn-predictor.streamlit.app)** — Enter any customer profile and get instant churn prediction with SHAP-based explanations!

## 📊 Project Overview
Built an end-to-end machine learning pipeline to identify churn-prone customers for a telecom company. 
The model helps retention teams proactively target high-risk customers, enabling a projected 18% reduction in churn.

## 🎯 Results
| Metric | Score |
|--------|-------|
| AUC-ROC | 0.83 |
| Minority-class Recall Improvement | +23% |
| Top Churn Driver | Contract Type (#1 by SHAP) |
| Projected Churn Reduction | 18% |

## 🛠️ Tech Stack
- **ML Model:** XGBoost, Scikit-Learn
- **Explainability:** SHAP Value Analysis
- **Data Processing:** Python, Pandas, NumPy
- **Class Imbalance:** SMOTE
- **Tuning:** Grid-Search, 5-Fold Cross Validation
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Streamlit, Streamlit Cloud
- **Database:** SQL

## 📁 Project Structure

## 🔬 Methodology
1. **Data Engineering** — Engineered 40+ behavioral, billing, and tenure features from 80K+ telecom records
2. **Class Imbalance** — Applied SMOTE to handle imbalanced target classes
3. **Model Training** — Trained XGBoost classifier with grid-search hyperparameter tuning
4. **Validation** — 5-fold cross-validation to ensure generalization
5. **Explainability** — SHAP value analysis to surface top 8 churn drivers
6. **Deployment** — Live web app deployed on Streamlit Cloud

## 📈 Key Findings
- **Contract type** is the #1 churn driver — identified via SHAP values
- Customers on month-to-month contracts are 3x more likely to churn
- New customers (tenure < 12 months) are highest risk
- SMOTE improved minority-class recall by 23% over baseline
- Customers without TechSupport or OnlineSecurity churn significantly more

## 🚀 How to Run Locally
```bash
git clone https://github.com/ami-raval/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
streamlit run app.py
```

## 👩‍💼 Author
**Ami Raval** — Data & Business Analyst  
[LinkedIn](https://linkedin.com/in/ami-raval) | [GitHub](https://github.com/ami-raval) | [Live App](https://ami-churn-predictor.streamlit.app)
