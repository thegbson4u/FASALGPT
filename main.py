# ================= IMPORTS =================
import streamlit as st
import tensorflow as tf
import numpy as np
import requests
import os

from huggingface_hub import hf_hub_download


# ================= STREAMLIT CONFIG =================
st.set_page_config(
    page_title="FASALGPT | Smart Agriculture Assistant",
    page_icon="🌾",
    layout="wide",
)

# ================= HIDE STREAMLIT FOOTER & MENU =================
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ================= GLOBAL CSS =================
st.markdown("""
<style>
html, body {
    background: #0f1f17;
    color: #eaeaea;
    font-family: 'Inter', sans-serif;
}

.glass {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(14px);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.15);
}

.metric {
    padding: 16px;
    background: rgba(255,255,255,0.06);
    border-radius: 12px;
    text-align: center;
}

.nav button {
    background: transparent;
    color: #eaeaea;
    border: none;
    font-size: 16px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ================= TOP NAVIGATION =================
tabs = st.tabs([
    "🏠 Home",
    "🌦 Weather",
    "🦠 Disease Detection",
    "🌾 Crop Recommendation",
    "ℹ About"
])


# ================= LOAD DISEASE MODEL (HF) =================
@st.cache_resource
def load_disease_model():
    model_path = hf_hub_download(
        repo_id="THEGBSON/fasalgpt-disease-model",
        filename="trained_model.h5",   # ✅ EXACT NAME
        revision="main"
    )
    return tf.keras.models.load_model(model_path, compile=False)


disease_model = load_disease_model()


# ================= DISEASE LABELS =================
DISEASE_CLASSES = {
    0: "Healthy Leaf",
    1: "Bacterial Blight",
    2: "Leaf Smut",
    3: "Brown Spot",
    4: "Powdery Mildew"
}


# ================= WEATHER FUNCTION =================
def get_weather(city):
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return None

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}
    res = requests.get(url, params=params)

    if res.status_code != 200:
        return None

    data = res.json()
    return {
        "temp": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "condition": data["weather"][0]["description"],
        "rain": data.get("rain", {}).get("1h", 0)
    }


# ================= HOME =================
with tabs[0]:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.header("🌾 FASALGPT")
    st.write("AI-Powered Smart Agriculture Assistant")
    st.markdown("""
    ✔ Disease detection  
    ✔ Weather advisory  
    ✔ Crop recommendation  
    ✔ Government-ready platform  
    """)
    st.markdown("</div>", unsafe_allow_html=True)


# ================= WEATHER =================
with tabs[1]:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.header("🌦 Weather Advisory")

    city = st.text_input("Enter City", "Delhi")

    if st.button("Get Weather"):
        weather = get_weather(city)
        if weather:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🌡 Temp", f"{weather['temp']} °C")
            c2.metric("💧 Humidity", f"{weather['humidity']} %")
            c3.metric("🌧 Rain", f"{weather['rain']} mm")
            c4.metric("☁ Condition", weather["condition"])
        else:
            st.error("Weather data unavailable")

    st.markdown("</div>", unsafe_allow_html=True)


# ================= DISEASE DETECTION =================
with tabs[2]:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.header("🦠 Crop Disease Detection")

    img = st.file_uploader("Upload leaf image", ["jpg", "png", "jpeg"])

    if img:
        st.image(img, caption="Uploaded Image", use_container_width=True)

        if st.button("Analyze"):
            image = tf.keras.preprocessing.image.load_img(img, target_size=(128, 128))
            arr = tf.keras.preprocessing.image.img_to_array(image)
            arr = np.expand_dims(arr, axis=0)

            preds = disease_model.predict(arr)
            idx = int(np.argmax(preds))
            conf = float(np.max(preds)) * 100

            st.success(f"🧪 Disease: **{DISEASE_CLASSES.get(idx, 'Unknown')}**")
            st.info(f"Confidence: {conf:.2f}%")

    st.markdown("</div>", unsafe_allow_html=True)


# ================= CROP RECOMMENDATION =================
with tabs[3]:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.header("🌾 Crop Recommendation")

    N = st.slider("Nitrogen", 0, 200, 50)
    P = st.slider("Phosphorus", 0, 200, 50)
    K = st.slider("Potassium", 0, 200, 50)
    temp = st.slider("Temperature", 0, 50, 25)
    rain = st.slider("Rainfall", 0, 300, 100)

    if st.button("Recommend Crop"):
        if rain > 200:
            crop = "Rice"
        elif temp < 20:
            crop = "Wheat"
        else:
            crop = "Maize"

        st.success(f"✔ Recommended Crop: **{crop}**")

    st.markdown("</div>", unsafe_allow_html=True)


# ================= ABOUT =================
with tabs[4]:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.header("ℹ About FASALGPT")
    st.write("A secure AI platform for smart agriculture. No personal data exposed.")
    st.markdown("</div>", unsafe_allow_html=True)
