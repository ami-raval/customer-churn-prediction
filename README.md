# customer-churn-prediction
XGBoost model achieving 89% precision &amp; 0.92 AUC-ROC to predict customer churn from 80K+ telecom records using SHAP explainability

## 📊 Project Overview
Built an end-to-end machine learning pipeline to identify churn-prone customers for a telecom company. 
The model helps retention teams proactively target high-risk customers, enabling a projected 18% reduction in churn.

## 🎯 Results
| Metric | Score |
|--------|-------|
| Precision | 89% |
| AUC-ROC | 0.92 |
| Minority-class Recall Improvement | +23% |
| Projected Churn Reduction | 18% |

## 🛠️ Tech Stack
- **ML Model:** XGBoost, Scikit-Learn
- **Explainability:** SHAP Value Analysis
- **Data Processing:** Python, Pandas, NumPy
- **Class Imbalance:** SMOTE
- **Tuning:** Grid-Search, 5-Fold Cross Validation
- **Visualization:** Power BI, Matplotlib, Seaborn
- **Database:** SQL

## 📁 Project Structure
## 🔬 Methodology
1. **Data Engineering** — Engineered 40+ behavioral, billing, and tenure features from 80K+ telecom records
2. **Class Imbalance** — Applied SMOTE to handle imbalanced target classes
3. **Model Training** — Trained XGBoost classifier with grid-search hyperparameter tuning
4. **Validation** — 5-fold cross-validation to ensure generalization
5. **Explainability** — SHAP value analysis to surface top 8 churn drivers
6. **Dashboard** — Interactive Power BI dashboard with drill-through cohort analysis

## 📈 Key Findings
- Top churn drivers identified via SHAP: contract type, tenure, monthly charges, tech support
- Customers on month-to-month contracts are 3x more likely to churn
- SMOTE improved minority-class recall by 23% over baseline

## 🚀 How to Run
```bash
git clone https://github.com/ami-raval/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
jupyter notebook notebooks/churn_prediction.ipynb
```

## 👩‍💼 Author
**Ami Raval** — Data & Business Analyst  
[LinkedIn](https://linkedin.com/in/ami-raval) | [GitHub](https://github.com/ami-raval)
