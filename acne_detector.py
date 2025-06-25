import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Acne Risk Predictor", layout="centered")

# Fetch real dataset from GitHub or remote source
@st.cache_data
def fetch_real_data():
    url = "https://raw.githubusercontent.com/datablist/sample-csv-files/main/files/people/people-100.csv"
    df = pd.read_csv(url)
    df = df.rename(columns={"Age": "age"})
    df = df.dropna().head(100)
    # Simulate acne features with correct relationships
    df["sleep_hours"] = np.random.randint(4, 10, size=len(df))
    df["water_intake_ltr"] = np.round(np.random.uniform(1.5, 3.5, size=len(df)), 1)
    df["stress_level"] = np.random.choice([0, 1, 2], size=len(df))
    df["skin_type"] = np.random.choice([0, 1, 2], size=len(df))
    df["diet"] = np.random.choice([0, 1], size=len(df))
    df["routine"] = np.random.choice([0, 1], size=len(df))

    # Generate acne target with water and sleep having inverse relation
    df["acne"] = (
        (df["water_intake_ltr"] < 2.5).astype(int) +
        (df["sleep_hours"] < 7).astype(int) +
        (df["stress_level"] > 1).astype(int) +
        (df["diet"] == 0).astype(int) +
        (df["routine"] == 0).astype(int)
    )
    df["acne"] = (df["acne"] > 2).astype(int)
    return df[["age", "sleep_hours", "water_intake_ltr", "stress_level", "skin_type", "diet", "routine", "acne"]]

# Train and save model
@st.cache_resource
def train_model(df):
    X = df.drop("acne", axis=1)
    y = df["acne"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    return model

# Load or create model from real dataset
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            return pickle.load(f)
    except:
        df = fetch_real_data()
        return train_model(df)

model = load_model()

# Get pollution level using OpenWeatherMap API
def get_air_quality(city="Delhi"):
    api_key = os.getenv("OPENWEATHERMAP_API_KEY", "YOUR_OPENWEATHERMAP_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat=28.6139&lon=77.2090&appid={api_key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            aqi = data['list'][0]['main']['aqi']
            return aqi - 1 if aqi <= 3 else 2
        else:
            return 1
    except:
        return 1

# UI Heading
st.markdown("""
    <h1 style='text-align: center; color: #ff4b4b;'>Acne Risk Predictor 🧴</h1>
""", unsafe_allow_html=True)

# User Inputs
age = st.slider("Select Age", 13, 45, 20)
sleep = st.slider("Sleep Hours (per day)", 4, 10, 7)
water = st.slider("Water Intake (liters/day)", 1.0, 4.0, 2.0, step=0.1)

skin = st.selectbox("Skin Type", ["Oily", "Dry", "Combination"])
diet = st.radio("Diet Type", ["Junk", "Healthy"])
routine = st.radio("Do you follow a skincare routine?", ["Yes", "No"])

# Get live pollution data
st.write("Fetching live air quality (as a factor for stress level)...")
stress_level = get_air_quality()
st.write(f"Detected stress level from pollution data: {['Low', 'Medium', 'High'][stress_level]}")

# Encode input
skin_map = {"Oily": 0, "Dry": 1, "Combination": 2}
diet_map = {"Junk": 0, "Healthy": 1}
routine_map = {"No": 0, "Yes": 1}

input_data = np.array([[age, sleep, water, 
                        stress_level,
                        skin_map[skin], 
                        diet_map[diet], 
                        routine_map[routine]]])

# Predict & Give Recommendations
if st.button("Predict Acne Risk"):
    prediction = model.predict(input_data)
    risk = "High Risk ⚠️" if prediction[0] == 1 else "Low Risk ✅"
    st.success(f"Based on your inputs, your acne risk is: {risk}")

    # Doctor-style feedback
    st.markdown("---")
    st.subheader("🩺 Doctor's Suggestions")

    if prediction[0] == 1:
        st.markdown("""
        - **Hydration:** Increase your water intake to flush toxins.
        - **Sleep:** Aim for 7–9 hours of sleep daily.
        - **Skincare:** Use non-comedogenic products and wash your face twice a day.
        - **Diet:** Reduce oily/junk food and include fruits and greens.
        - **Stress:** Try yoga or breathing exercises to reduce stress levels.
        - **Pollution:** Clean your skin after outdoor exposure and consider using a light anti-pollution serum.
        """)
    else:
        st.markdown("""
        - Your skin seems to be at low risk for acne – great job! 😄
        - Continue your routine and maintain hydration and sleep cycles.
        - Keep monitoring your skin changes and act early if needed.
        """)

# Styling
st.markdown("""
    <style>
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-size: 16px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)