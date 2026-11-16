
import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------
st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# -----------------------------------
# LOAD TRAINED MODEL
# -----------------------------------
MODEL_PATH = "models/xgboost.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

# -----------------------------------
# HEADER
# -----------------------------------
st.title("📊 Telecom Customer Churn Prediction")
st.caption("AI-powered customer retention analysis using XGBoost")

st.divider()

if model is None:
    st.error("Trained model not found. Please run python main.py first.")
    st.stop()

# -----------------------------------
# SIDEBAR
# -----------------------------------
st.sidebar.title("About Project")
st.sidebar.info(
    """
    Telecom Customer Churn Prediction

    Machine Learning Models:
    - Random Forest
    - XGBoost

    Best Model: XGBoost
    ROC-AUC: 84.12%
    """
)

st.sidebar.markdown("---")
st.sidebar.write("Developed using Python, Pandas, Scikit-learn, XGBoost and Streamlit.")

# -----------------------------------
# CUSTOMER INPUT FORM
# -----------------------------------
st.subheader("👤 Enter Customer Details")

with st.form("customer_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Has Partner?", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
        tenure = st.number_input(
            "Tenure (Months)", min_value=0, max_value=72, value=12
        )
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"]
        )

    with col2:
        internet = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"]
        )

    with col3:
        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            value=70.0,
            step=1.0
        )
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            value=840.0,
            step=1.0
        )

    submitted = st.form_submit_button(
        "🔍 Predict Customer Churn",
        use_container_width=True
    )

# -----------------------------------
# PREDICTION
# -----------------------------------
if submitted:

    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.divider()
        st.subheader("📈 Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            if prediction == 1:
                st.error("⚠️ Customer is likely to Churn")
            else:
                st.success("✅ Customer is likely to Stay")

        with result_col2:
            st.metric(
                "Churn Probability",
                f"{probability * 100:.2f}%"
            )

        st.progress(float(probability))

        if prediction == 1:
            st.warning(
                "Suggested Action: Contact this customer and consider "
                "offering suitable retention benefits."
            )
        else:
            st.info(
                "Suggested Action: Continue regular customer engagement."
            )

        with st.expander("View Customer Input Data"):
            st.dataframe(input_data, use_container_width=True)

    except Exception as error:
        st.error(f"Prediction error: {error}")

# -----------------------------------
# FOOTER
# -----------------------------------
st.divider()
st.caption("Telecom Customer Churn Prediction | Powered by XGBoost")
