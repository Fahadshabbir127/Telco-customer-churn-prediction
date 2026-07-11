import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb

# Load model (native XGBoost format - avoids version mismatch issues)
model = xgb.XGBClassifier()
model.load_model('churn_xgboost_model.json')

# Load scaler and column order
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

st.title("📉 Telco Customer Churn Predictor")
st.write("Customer ki details fill karein aur predict karein ke woh churn karega ya nahi.")

st.divider()

# ---------------- Demographics ----------------
st.subheader("👤 Demographics")
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    under_30 = st.selectbox("Under 30?", ["No", "Yes"])
    senior_citizen = st.selectbox("Senior Citizen?", ["No", "Yes"])
with col2:
    married = st.selectbox("Married?", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
    number_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
    population = st.number_input("Population (area)", min_value=0, max_value=200000, value=25000)

st.divider()

# ---------------- Account Info ----------------
st.subheader("📋 Account Information")
col3, col4 = st.columns(2)
with col3:
    referred_friend = st.selectbox("Referred a Friend?", ["No", "Yes"])
    number_of_referrals = st.number_input("Number of Referrals", min_value=0, max_value=20, value=0)
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
    contract = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
with col4:
    paperless_billing = st.selectbox("Paperless Billing?", ["No", "Yes"])
    payment_method = st.selectbox("Payment Method", ["Bank Withdrawal", "Credit Card", "Mailed Check"])
    offer = st.selectbox("Offer", ["None", "Offer A", "Offer B", "Offer C", "Offer D", "Offer E"])

st.divider()

# ---------------- Services ----------------
st.subheader("📞 Phone & Internet Services")
col5, col6 = st.columns(2)
with col5:
    phone_service = st.selectbox("Phone Service?", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines?", ["No", "Yes"])
    avg_long_distance = st.number_input("Avg Monthly Long Distance Charges ($)", min_value=0.0, max_value=100.0, value=20.0)
    internet_service = st.selectbox("Internet Service?", ["No", "Yes"])
with col6:
    internet_type = st.selectbox("Internet Type", ["Cable", "DSL", "Fiber Optic", "None"])
    avg_gb_download = st.number_input("Avg Monthly GB Download", min_value=0, max_value=200, value=20)
    unlimited_data = st.selectbox("Unlimited Data?", ["No", "Yes"])

st.divider()

# ---------------- Add-on Services ----------------
st.subheader("🛠️ Add-on Services")
col7, col8 = st.columns(2)
with col7:
    online_security = st.selectbox("Online Security?", ["No", "Yes"])
    online_backup = st.selectbox("Online Backup?", ["No", "Yes"])
    device_protection = st.selectbox("Device Protection Plan?", ["No", "Yes"])
with col8:
    premium_tech_support = st.selectbox("Premium Tech Support?", ["No", "Yes"])
    streaming_tv = st.selectbox("Streaming TV?", ["No", "Yes"])
    streaming_movies = st.selectbox("Streaming Movies?", ["No", "Yes"])
    streaming_music = st.selectbox("Streaming Music?", ["No", "Yes"])

st.divider()

# ---------------- Charges ----------------
st.subheader("💰 Charges")
col9, col10 = st.columns(2)
with col9:
    monthly_charge = st.number_input("Monthly Charge ($)", min_value=0.0, max_value=200.0, value=65.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=800.0)
    total_refunds = st.number_input("Total Refunds ($)", min_value=0.0, max_value=1000.0, value=0.0)
with col10:
    total_extra_data = st.number_input("Total Extra Data Charges ($)", min_value=0.0, max_value=1000.0, value=0.0)
    total_long_distance = st.number_input("Total Long Distance Charges ($)", min_value=0.0, max_value=5000.0, value=200.0)
    total_revenue = st.number_input("Total Revenue ($)", min_value=0.0, max_value=15000.0, value=1000.0)

st.divider()

# ---------------- Predict ----------------
if st.button("🔍 Predict Churn", use_container_width=True):

    # Build raw input row (matches the original dataframe structure before encoding)
    raw_input = {
        'Gender': gender,
        'Age': age,
        'Under 30': under_30,
        'Senior Citizen': senior_citizen,
        'Married': married,
        'Dependents': dependents,
        'Number of Dependents': number_of_dependents,
        'Population': population,
        'Referred a Friend': referred_friend,
        'Number of Referrals': number_of_referrals,
        'Tenure in Months': tenure,
        'Phone Service': phone_service,
        'Avg Monthly Long Distance Charges': avg_long_distance,
        'Multiple Lines': multiple_lines,
        'Internet Service': internet_service,
        'Avg Monthly GB Download': avg_gb_download,
        'Online Security': online_security,
        'Online Backup': online_backup,
        'Device Protection Plan': device_protection,
        'Premium Tech Support': premium_tech_support,
        'Streaming TV': streaming_tv,
        'Streaming Movies': streaming_movies,
        'Streaming Music': streaming_music,
        'Unlimited Data': unlimited_data,
        'Paperless Billing': paperless_billing,
        'Monthly Charge': monthly_charge,
        'Total Charges': total_charges,
        'Total Refunds': total_refunds,
        'Total Extra Data Charges': total_extra_data,
        'Total Long Distance Charges': total_long_distance,
        'Total Revenue': total_revenue,
        'Offer': offer,
        'Internet Type': internet_type,
        'Contract': contract,
        'Payment Method': payment_method,
    }

    df_input = pd.DataFrame([raw_input])

    # Step 1: Gender mapping
    df_input['Gender'] = df_input['Gender'].map({'Male': 1, 'Female': 0})

    # Step 2: Binary Yes/No columns
    binary_cols = ['Under 30', 'Senior Citizen', 'Married', 'Dependents',
                   'Referred a Friend', 'Phone Service', 'Multiple Lines',
                   'Internet Service', 'Online Security', 'Online Backup',
                   'Device Protection Plan', 'Premium Tech Support', 'Streaming TV',
                   'Streaming Movies', 'Streaming Music', 'Unlimited Data',
                   'Paperless Billing']
    for col in binary_cols:
        df_input[col] = df_input[col].map({'Yes': 1, 'No': 0})

    # Step 3: One-Hot Encode multi-class columns
    multi_class_cols = ['Offer', 'Internet Type', 'Contract', 'Payment Method']
    df_input = pd.get_dummies(df_input, columns=multi_class_cols, drop_first=True)

    # Step 4: Align columns with training data (add any missing dummy columns as 0)
    df_input = df_input.reindex(columns=model_columns, fill_value=0)

    # Step 5: Scale
    df_scaled = scaler.transform(df_input)

    # Step 6: Predict
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"### ⚠️ This customer is LIKELY TO CHURN\n**Churn Probability: {probability*100:.1f}%**")
    else:
        st.success(f"### ✅ This customer is LIKELY TO STAY\n**Churn Probability: {probability*100:.1f}%**")
