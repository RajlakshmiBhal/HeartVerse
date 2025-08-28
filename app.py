import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
from fpdf import FPDF

# 🎨 Page Config and Styling
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.markdown("""
    <style>
    html, body, [class*="css"] {
        background-color: #8B0000;
        color: black;
        font-family: 'Georgia', serif;
    }
    h1, h2, h3 {
        text-align: center;
        color: black;
    }
    .stButton>button, .stDownloadButton>button {
        background-color: black;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1.5em;
        font-size: 16px;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

# 🏥 Header
st.markdown("<h1>Heart Disease Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>\"Let your heartbeat speak—where science meets clarity, healing begins.\"</p>", unsafe_allow_html=True)

# 🔍 Load Model & Scaler
model = joblib.load('heart_rf_model.pkl')
scaler = joblib.load('heart_scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# 📝 Manual Form
st.markdown("### Fill Out Patient Details")
with st.form("manual_input"):
    name = st.text_input("Patient Name")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120)
        trestbps = st.number_input("Resting Blood Pressure")
        chol = st.number_input("Cholesterol")
        thalch = st.number_input("Max Heart Rate Achieved")
    with col2:
        oldpeak = st.number_input("ST Depression")
        ca = st.number_input("Number of Major Vessels", min_value=0, max_value=3)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
        thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversable defect"])
    submitted = st.form_submit_button("Predict")

if submitted:
    manual_df = pd.DataFrame([{
        'age': age, 'trestbps': trestbps, 'chol': chol, 'thalch': thalch,
        'oldpeak': oldpeak, 'ca': ca, 'sex': sex, 'cp': cp, 'thal': thal
    }])
    manual_df = pd.get_dummies(manual_df)
    manual_df = manual_df.reindex(columns=feature_columns, fill_value=0)
    scaled_manual = scaler.transform(manual_df)
    pred = model.predict(scaled_manual)[0]
    prob = model.predict_proba(scaled_manual)[0][1]
    risk = "High" if prob > 0.75 else "Moderate" if prob > 0.4 else "Low"

    note = ("This patient exhibits clinical indicators consistent with elevated cardiac risk. "
            "Immediate consultation with a cardiologist is strongly recommended. Lifestyle modifications and diagnostic follow-up are advised."
            if pred == 1 else
            "No immediate cardiac risk detected based on current parameters. Recommend maintaining a heart-healthy lifestyle and scheduling regular checkups.")

    precautions = (
        "• Maintain a balanced diet low in saturated fats and sodium\n"
        "• Engage in regular physical activity (30 minutes/day)\n"
        "• Avoid tobacco and excessive alcohol\n"
        "• Monitor blood pressure and cholesterol levels\n"
        "• Follow up with a healthcare provider for personalized guidance"
    )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 🛡️ Safe Text Function
    def safe_text(text):
        return text.encode('latin-1', 'replace').decode('latin-1')

    # 📄 Generate PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(139, 0, 0)
    pdf.rect(0, 0, 210, 297, 'F')
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", size=12)
    pdf.set_xy(10, 10)
    pdf.multi_cell(0, 10, safe_text(f"""
Heart Disease Risk Assessment Report

Patient Name: {name}
Generated On: {timestamp}

----------------------------------------

Patient Details:
Age: {age}
Sex: {sex}
Chest Pain Type: {cp}
Resting Blood Pressure: {trestbps} mmHg
Cholesterol: {chol} mg/dL
Max Heart Rate Achieved: {thalch} bpm
ST Depression: {oldpeak}
Major Vessels: {ca}
Thalassemia: {thal}

----------------------------------------

Prediction: {'Heart Disease Detected' if pred == 1 else 'No Disease Detected'}
Confidence Score: {prob:.2%}
Risk Category: {risk}

----------------------------------------

Doctor's Note:
{note}

----------------------------------------

Precautionary Advice:
{precautions}

----------------------------------------

This report is generated with care and poetry by Rajlakshmi's HeartVerse.
"""))

    pdf.output("clinical_heart_report.pdf")

    with open("clinical_heart_report.pdf", "rb") as f:
        st.download_button("Download Clinical PDF Report", f.read(), f"{name}_heart_report.pdf", "application/pdf")
