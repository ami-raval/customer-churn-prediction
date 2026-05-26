import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔍",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model():
    with open('churn_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# Header
st.title("🔍 Customer Churn Predictor")
st.markdown("**Predict whether a telecom customer will churn using XGBoost + SHAP**")
st.markdown("---")

# Input form
st.subheader("📋 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    contract = st.selectbox("Contract Type",
        ["Month-to-month", "One year", "Two year"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 0, 120, 65)
    internet_service = st.selectbox("Internet Service",
        ["DSL", "Fiber optic", "No"])
    tech_support = st.selectbox("Tech Support",
        ["Yes", "No", "No internet service"])

with col2:
    online_security = st.selectbox("Online Security",
        ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup",
        ["Yes", "No", "No internet service"])
    payment_method = st.selectbox("Payment Method",
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)",
         "Credit card (automatic)"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])

st.markdown("---")

# Predict button
if st.button("🔮 Predict Churn Risk", use_container_width=True):

    # Encode inputs
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
    yes_no_map = {"No": 0, "Yes": 1, "No internet service": 2}
    payment_map = {
        "Bank transfer (automatic)": 0,
        "Credit card (automatic)": 1,
        "Electronic check": 2,
        "Mailed check": 3
    }

    # Calculate derived features
    total_charges = monthly_charges * tenure
    avg_monthly_spend = total_charges / (tenure + 1)
    tenure_group = 0 if tenure <= 12 else (1 if tenure <= 24 else (2 if tenure <= 48 else 3))
    num_services = sum([
        1 if internet_service != "No" else 0,
        yes_no_map[online_security] == 1,
        yes_no_map[online_backup] == 1,
        yes_no_map[tech_support] == 1,
    ])
    senior_no_support = 1 if (senior_citizen == "Yes" and tech_support == "No") else 0
    high_value = 1 if monthly_charges > 65 else 0

    # Build input dataframe — must match training columns exactly
    input_data = pd.DataFrame({
        'gender': [0],
        'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
        'Partner': [0],
        'Dependents': [yes_no_map[dependents]],
        'tenure': [tenure],
        'PhoneService': [1],
        'MultipleLines': [0],
        'InternetService': [internet_map[internet_service]],
        'OnlineSecurity': [yes_no_map[online_security]],
        'OnlineBackup': [yes_no_map[online_backup]],
        'DeviceProtection': [0],
        'TechSupport': [yes_no_map[tech_support]],
        'StreamingTV': [0],
        'StreamingMovies': [0],
        'Contract': [contract_map[contract]],
        'PaperlessBilling': [1],
        'PaymentMethod': [payment_map[payment_method]],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
        'AvgMonthlySpend': [avg_monthly_spend],
        'TenureGroup': [float(tenure_group)],
        'NumServices': [num_services],
        'SeniorNoSupport': [senior_no_support],
        'HighValue': [high_value]
    })

    # Predict
    prob = model.predict_proba(input_data)[0][1]
    prediction = "WILL CHURN" if prob > 0.5 else "WILL STAY"

    # Display result
    st.markdown("---")
    st.subheader("🎯 Prediction Result")

    if prob > 0.5:
        st.error(f"⚠️ **HIGH RISK — {prob*100:.1f}% chance of churning**")
    else:
        st.success(f"✅ **LOW RISK — {prob*100:.1f}% chance of churning**")

    # Risk meter
    st.progress(float(prob))
    st.caption(f"Churn probability: {prob*100:.1f}%")

    # Key drivers
    st.markdown("---")
    st.subheader("📊 Top Churn Risk Factors")

    drivers = []
    if contract == "Month-to-month":
        drivers.append("⚠️ Month-to-month contract — highest churn risk")
    if tenure < 12:
        drivers.append("⚠️ New customer (< 12 months) — high risk period")
    if monthly_charges > 65:
        drivers.append("⚠️ High monthly charges — above average")
    if tech_support == "No":
        drivers.append("⚠️ No tech support — increases churn risk")
    if online_security == "No":
        drivers.append("⚠️ No online security — increases churn risk")
    if senior_citizen == "Yes" and tech_support == "No":
        drivers.append("⚠️ Senior citizen without support — very high risk")

    if drivers:
        for d in drivers[:3]:
            st.warning(d)
    else:
        st.info("✅ This customer shows strong loyalty indicators")

    # Retention recommendation
    st.markdown("---")
    st.subheader("💡 Retention Recommendation")
    if prob > 0.7:
        st.error("🚨 **Immediate action required** — Call this customer today and offer a contract upgrade discount")
    elif prob > 0.5:
        st.warning("📞 **Follow up this week** — Offer tech support bundle or loyalty discount")
    else:
        st.success("😊 **Low priority** — Customer is likely to stay. Standard engagement sufficient")

# Footer
st.markdown("---")
st.caption("Built by Ami Raval | XGBoost + SHAP | AUC-ROC: 0.83 | github.com/ami-raval")
