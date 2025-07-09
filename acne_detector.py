# --- Health Condition Predictor App ---
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
from fpdf import FPDF
import requests

# --- Set up Streamlit page ---
st.set_page_config(page_title="🩺 Health Condition Predictor", layout="wide")

# --- Title Banner ---
st.markdown("""
    <h1 style='text-align: center; color: #e91e63;'>
        ⚡️  *Predictly Health Check* – स्मार्ट लाइफस्टाइल सलाहकार
    </h1>
    <h4 style='text-align: center; color: gray;'>Made with ❤️ by Deepali Verma</h4>
    <hr style='border: 1px solid #e91e63;'>
""", unsafe_allow_html=True)

# --- Model and configuration ---
models = {
    "Acne": "acne_model.pkl",
    "Hairfall": "hairfall_model.pkl",
    "Weight Gain": "weight_model.pkl"
}

condition_factors = {
    "Acne": [
        ("Age", (10, 45)),
        ("Sleep Hours", (4, 10)),
        ("Water Intake (L/day)", (1, 5)),
        ("Stress Level (0=Low, 1=Medium, 2=High)", (0, 2)),
        ("Skin Type (0=Oily, 1=Dry, 2=Combination)", (0, 2)),
        ("Diet (0=Junk, 1=Healthy)", (0, 1)),
        ("Routine (0=No, 1=Yes)", (0, 1))
    ],
    "Hairfall": [
        ("Age", (10, 60)),
        ("Protein Intake (g/day)", (20, 150)),
        ("Stress Level (0=Low, 1=Medium, 2=High)", (0, 2)),
        ("Shampoo Frequency/week", (0, 7)),
        ("Sleep Hours", (4, 10)),
        ("Pollution Level (0=Low, 1=Medium, 2=High)", (0, 2))
    ],
    "Weight Gain": [
        ("Age", (10, 70)),
        ("Daily Calorie Intake", (1000, 4000)),
        ("Physical Activity Level (0=Low, 1=Medium, 2=High)", (0, 2)),
        ("Sleep Hours", (4, 10)),
        ("Metabolism Rate (0=Slow, 1=Normal, 2=Fast)", (0, 2))
    ]
}

@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def suggest_doctors_nearby():
    st.subheader("🏥 Nearby Dermatologists or Clinics")
    try:
        location_info = requests.get("https://ipinfo.io").json()
        city = location_info.get("city", "your area")
        latlon = location_info.get("loc", "28.61,77.20")
        st.markdown(f"Showing doctors near **{city}**")
        query = f"https://www.google.com/maps/search/dermatologist+near+{city}/@{latlon},14z"
        st.markdown(f"[🔗 Click here to view on Google Maps]({query})", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not fetch location: {e}")

# --- UI ---
condition = st.selectbox("Select Health Issue to Predict:", list(models.keys()))
model_path = models[condition]
model = load_model(model_path)

factors = condition_factors[condition]
user_input = []

st.subheader(f"📝 Please fill in the information for {condition} prediction")

for label, (min_val, max_val) in factors:
    if max_val <= 2:
        value = st.selectbox(label, list(range(min_val, max_val + 1)))
    else:
        value = st.slider(label, min_val, max_val, int((min_val + max_val) / 2))
    user_input.append(value)

input_array = np.array(user_input).reshape(1, -1)

if st.button("🔍 Predict Severity"):
    prediction = model.predict(input_array)[0]
    st.success(f"Predicted {condition} severity score (0–10): {prediction:.2f}")

    st.markdown("---")
    st.subheader("📋 Suggestions")
    tips = [f"Improve {label.lower()} to reduce severity." for (label, _), val in zip(factors, user_input) if isinstance(val, (int, float)) and val < 5]
    tips.append("Consult a specialist if score > 7.") if prediction > 7 else tips.append("Maintain current routine.")

    for tip in tips:
        st.markdown(f"- {tip}")

    st.markdown("---")
    st.subheader("📊 Input Chart")
    fig, ax = plt.subplots(figsize=(6, 3))
    labels = [label for label, _ in factors]
    ax.barh(labels, user_input, color="#4caf50")
    ax.set_xlim(left=0)
    st.pyplot(fig)

    # PDF Report Download
    from io import BytesIO
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"{condition} Prediction Report", ln=True, align='C')
    pdf.ln(10)
    for label, value in zip(labels, user_input):
        pdf.cell(200, 10, f"{label}: {value}", ln=True)
    pdf.cell(200, 10, f"Predicted Severity: {prediction:.2f}", ln=True)
    for tip in tips:
        pdf.cell(200, 10, f"Tip: {tip}", ln=True)

    pdf_data = pdf.output(dest='S').encode('latin1')

    st.download_button(
        label="⬇️ Download PDF Report",
        data=pdf_data,
        file_name=f"{condition.lower()}_report.pdf",
        mime="application/pdf"
    )

    # WhatsApp Sharing (link based)
    message = f"{condition} Severity Report\nScore: {prediction:.2f}\n" + "\n".join(tips)
    encoded_msg = requests.utils.quote(message)
    whatsapp_url = f"https://wa.me/?text={encoded_msg}"
    st.markdown("---")
    st.subheader("📤 Share on WhatsApp")
    st.markdown(f"[📲 Click here to share on WhatsApp]({whatsapp_url})", unsafe_allow_html=True)

    st.markdown("---")
    suggest_doctors_nearby()
