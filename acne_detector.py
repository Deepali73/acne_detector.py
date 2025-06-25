# Acne Severity Predictor (Regression Version)
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Acne Severity Predictor", layout="centered")

@st.cache_data
def fetch_real_data():
    np.random.seed(42)
    df = pd.DataFrame({
        "age": np.random.randint(13, 45, 300),
        "sleep_hours": np.random.randint(4, 10, 300),
        "water_intake_ltr": np.round(np.random.uniform(1.5, 4.0, 300), 1),
        "stress_level": np.random.choice([0, 1, 2], 300),
        "skin_type": np.random.choice([0, 1, 2], 300),
        "diet": np.random.choice([0, 1], 300),
        "routine": np.random.choice([0, 1], 300),
    })
    df["acne_severity"] = (
        -2.0 * df["water_intake_ltr"] +
        (7 - df["sleep_hours"]) * 1.5 +
        df["stress_level"] * 2.5 +
        (1 - df["diet"]) * 2.0 +
        (1 - df["routine"]) * 2.0 +
        np.random.normal(0, 1, len(df)) + 4
    )
    df["acne_severity"] = df["acne_severity"].clip(0, 10)
    return df

@st.cache_resource
def train_model(df):
    X = df.drop("acne_severity", axis=1)
    y = df["acne_severity"]
    model = LinearRegression()
    model.fit(X, y)
    with open("regression_model.pkl", "wb") as f:
        pickle.dump(model, f)
    return model

def load_model():
    try:
        with open("regression_model.pkl", "rb") as f:
            return pickle.load(f)
    except:
        df = fetch_real_data()
        return train_model(df)

model = load_model()


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

st.markdown("""
    <h1 style='text-align: center; color: #ff4b4b;'>Acne Severity Predictor 🎯</h1>
""", unsafe_allow_html=True)

age = st.slider("Select Age", 13, 45, 20)
sleep = st.slider("Sleep Hours (per day)", 4, 10, 7)
water = st.slider("Water Intake (liters/day)", 1.0, 4.0, 2.0, step=0.1)

skin = st.selectbox("Skin Type", ["Oily", "Dry", "Combination"])
diet = st.radio("Diet Type", ["Junk", "Healthy"])
routine = st.radio("Do you follow a skincare routine?", ["Yes", "No"])

st.write("Fetching live air quality (as a factor for stress level)...")
stress_level = get_air_quality()
st.write(f"Detected stress level from pollution data: {['Low', 'Medium', 'High'][stress_level]}")

skin_map = {"Oily": 0, "Dry": 1, "Combination": 2}
diet_map = {"Junk": 0, "Healthy": 1}
routine_map = {"No": 0, "Yes": 1}

input_data = np.array([[age, sleep, water,
                        stress_level,
                        skin_map[skin],
                        diet_map[diet],
                        routine_map[routine]]])

if st.button("Predict Acne Severity"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted acne severity score (0–10): {prediction:.2f}")

    st.markdown("---")
    st.subheader("🩺 Doctor's Suggestions")

    suggestions = []

    if water < 2.5:
        suggestions.append("- Increase water intake to keep your skin hydrated.")
    else:
        suggestions.append("- Your hydration level is good. Keep it up!")

    if sleep < 7:
        suggestions.append("- Improve sleep quality. Aim for 7–8 hours per night.")
    else:
        suggestions.append("- Sleep hours look sufficient.")

    if stress_level > 0:
        suggestions.append("- Practice relaxation techniques to manage stress levels.")

    if diet_map[diet] == 0:
        suggestions.append("- Reduce junk food. Prefer a balanced, nutritious diet.")
    else:
        suggestions.append("- Your diet appears healthy.")

    if routine_map[routine] == 0:
        suggestions.append("- Start a consistent skincare routine for better results.")
    else:
        suggestions.append("- Skincare routine is in place. Great job!")

    if prediction > 6:
        suggestions.append("- Severe acne detected. Consider consulting a dermatologist.")
    elif prediction > 3:
        suggestions.append("- Moderate acne. Monitor your habits and adjust accordingly.")
    else:
        suggestions.append("- Low severity. Keep maintaining your current lifestyle.")

    for s in suggestions:
        st.markdown(s)

    # Plot user input and prediction visually
    st.markdown("---")
    st.subheader("📊 Input Summary & Prediction")
    fig, ax = plt.subplots()
    feature_names = ["Age", "Sleep", "Water", "Stress", "Skin Type", "Diet", "Routine"]
    input_values = input_data[0].tolist()
    bars = ax.barh(feature_names, input_values, color="#ff8080")
    ax.set_xlabel("Input Values")
    st.pyplot(fig)

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
