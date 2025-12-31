import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Calorie Burn Predictor", page_icon="🔥")

st.title("🔥 Calorie Burn Prediction App")
st.write("Estimate calories burned during exercise using a machine learning model.")


@st.cache_data
def load_data():
    df = pd.read_csv("/calories.csv") 
    return df

df = load_data()

# Encode Gender
label_encoder = LabelEncoder()
df["Gender"] = label_encoder.fit_transform(df["Gender"])

X = df.drop(['Calories', 'User_ID'], axis=1)
y = df["Calories"]

# Train model (fast)
@st.cache_resource
def train_model(X, y):
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

model = train_model(X, y)


st.subheader("User Details")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 10, 80, 25)
height = st.slider("Height (cm)", 130, 210, 170)
weight = st.slider("Weight (kg)", 30, 150, 70)

st.subheader(" Exercise Details")

duration = st.slider("Exercise Duration (minutes)", 1, 120, 30)
heart_rate = st.slider("Heart Rate", 60, 200, 120)
body_temp = st.slider("Body Temperature (°C)", 35.0, 42.0, 37.0)


gender_encoded = label_encoder.transform([gender.lower()])[0]

features = np.array([[gender_encoded, age, height, weight,
                      duration, heart_rate, body_temp]])


if st.button("🔥 Predict Calories Burned"):
    calories = model.predict(features)[0]
    st.success(f"🔥 Estimated Calories Burned: **{calories:.2f} kcal**")
