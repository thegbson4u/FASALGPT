# ================= IMPORTS =================
import streamlit as st
import tensorflow as tf
import numpy as np
import os

from huggingface_hub import hf_hub_download
from weather import get_weather


# ================= STREAMLIT CONFIG =================
st.set_page_config(
    page_title="FASALGPT | Smart Agriculture Assistant",
    page_icon="🌾",
    layout="wide",
)


# ================= GLOBAL CSS =================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #0f1f17;
    color: #eaeaea;
}
#MainMenu, footer, header {visibility: hidden;}

.glass {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(14px);
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.15);
    padding: 28px;
    margin-bottom: 20px;
}

.metric-box {
    padding: 16px;
    background: rgba(255,255,255,0.06);
    border-radius: 12px;
    text-align: center;
    font-size: 15px;
}

.alert {
    padding: 14px;
    border-left: 5px solid #FFC107;
    background: rgba(255,193,7,0.18);
    border-radius: 8px;
    margin-top: 10px;
}

.stButton>button {
    background: linear-gradient(135deg, #4CAF50, #2e7d32);
    color: white;
    border-radius: 10px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ================= TOP NAVBAR =================
st.markdown("""
<div class="glass">
  <h2>🌾 FasalGPT</h2>
  <p>AI-Powered Smart Agriculture Assistant</p>
</div>
""", unsafe_allow_html=True)


# ================= LOAD DISEASE MODEL =================
@st.cache_resource
def load_disease_model():
    model_path = hf_hub_download(
        repo_id="THEGBSON/fasalgpt-disease-model",
        filename="trained_model_keras.keras"
    )
    return tf.keras.models.load_model(model_path)


disease_model = load_disease_model()


# ================= RULE-BASED CROP RECOMMENDATION =================
def recommend_crop(N, P, K, temp, hum, ph, rain):
    if rain > 200 and temp >= 25:
        return "Rice 🌾"
    if rain < 100 and temp >= 20:
        return "Wheat 🌾"
    if ph < 6.5:
        return "Potato 🥔"
    if K > 150:
        return "Sugarcane 🎋"
    return "Maize 🌽"


# ================= WEATHER LOGIC =================
def weather_advisory(city):
    weather = get_weather(city)
    advice = []

    if weather["rainfall"] > 5:
        advice.append("🌧️ आज सिंचाई की आवश्यकता नहीं है")
    if weather["temperature"] > 35:
        advice.append("🔥 लू का खतरा – फसल को नुकसान हो सकता है")
    if weather["humidity"] > 80:
        advice.append("🦠 फंगल रोग का खतरा अधिक")

    return weather, advice


# ================= DISEASE PREDICTION =================
def predict_disease(img):
    image = tf.keras.preprocessing.image.load_img(img, target_size=(128, 128))
    arr = tf.keras.preprocessing.image.img_to_array(image)
    arr = np.expand_dims(arr, axis=0)

    preds = disease_model.predict(arr)
    return int(np.argmax(preds)), float(np.max(preds))


# ================= SIDEBAR =================
st.sidebar.title("🌱 Navigation")
app_mode = st.sidebar.radio(
    "",
    ["Home", "Weather", "Disease Detection", "Crop Recommendation", "About"]
)


# ================= HOME =================
if app_mode == "Home":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.header("📊 Farmer Dashboard")
    st.markdown("""
    ✔ Real-time weather  
    ✔ AI disease detection  
    ✔ Smart crop recommendation  
    """)
    st.markdown('</div>', unsafe_allow_html=True)


# ================= WEATHER =================
elif app_mode == "Weather":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    city = st.text_input("शहर का नाम", "Delhi")

    if st.button("Get Weather"):
        weather, advice = weather_advisory(city)

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='metric-box'>🌡 {weather['temperature']} °C</div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-box'>💧 {weather['humidity']} %</div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-box'>🌧 {weather['rainfall']} mm</div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-box'>☁ {weather['condition']}</div>", unsafe_allow_html=True)

        for a in advice:
            st.markdown(f"<div class='alert'>{a}</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ================= DISEASE DETECTION =================
elif app_mode == "Disease Detection":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    img = st.file_uploader("पत्ती की फोटो अपलोड करें", ["jpg", "png", "jpeg"])

    if img and st.button("Analyze"):
        idx, conf = predict_disease(img)
        st.success(f"रोग पहचान (Class ID): {idx}")
        st.info(f"Confidence: {conf * 100:.2f}%")

    st.markdown('</div>', unsafe_allow_html=True)


# ================= CROP RECOMMENDATION =================
elif app_mode == "Crop Recommendation":
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    N = st.slider("Nitrogen", 0, 200, 50)
    P = st.slider("Phosphorus", 0, 200, 50)
    K = st.slider("Potassium", 0, 200, 50)
    ph = st.slider("Soil pH", 0.0, 14.0, 7.0)
    temp = st.slider("Temperature", 0, 50, 25)
    hum = st.slider("Humidity", 0, 100, 60)
    rain = st.slider("Rainfall", 0, 300, 100)

    if st.button("Recommend"):
        crop = recommend_crop(N, P, K, temp, hum, ph, rain)
        st.success(f"✔ अनुशंसित फसल: {crop}")

    st.markdown('</div>', unsafe_allow_html=True)


# ================= ABOUT =================
elif app_mode == "About":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("FasalGPT – AI-powered agriculture assistant for Indian farmers.")
    st.markdown('</div>', unsafe_allow_html=True)
