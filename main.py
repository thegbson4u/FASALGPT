# ================= IMPORTS =================
import streamlit as st
import tensorflow as tf
import numpy as np
import requests
import os

from huggingface_hub import hf_hub_download

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="FASALGPT",
    page_icon="🌾",
    layout="wide",
)

# ================= HIDE STREAMLIT BRANDING & FOOTER =================
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
body {
    background-color: #0f1f17;
    color: #eaeaea;
    font-family: Inter, sans-serif;
}

.navbar {
    display: flex;
    gap: 18px;
    padding: 12px 20px;
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    border-radius: 14px;
    margin-bottom: 20px;
}

.nav-item {
    font-weight: 600;
    cursor: pointer;
}

.box {
    background: rgba(255,255,255,0.08);
    padding: 22px;
    border-radius: 14px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ================= TOP NAV =================
if "page" not in st.session_state:
    st.session_state.page = "Home"

def nav_button(label):
    if st.button(label):
        st.session_state.page = label

with st.container():
    cols = st.columns(8)
    labels = [
        "Home", "Weather", "Disease",
        "Crop", "Soil", "Irrigation",
        "Schemes", "About"
    ]
    for i, label in enumerate(labels):
        with cols[i]:
            nav_button(label)

page = st.session_state.page

# ================= DISEASE LABELS =================
DISEASE_CLASSES = [
    "Healthy",
    "Leaf Blight",
    "Powdery Mildew",
    "Rust",
    "Leaf Spot"
]

# ================= LOAD MODELS =================
@st.cache_resource
def load_disease_model():
    model_path = hf_hub_download(
        repo_id="THEGBSON/fasalgpt-disease-model",
        filename="trained_model_keras.keras"
    )
    return tf.keras.models.load_model(model_path, compile=False)

disease_model = load_disease_model()

# ================= WEATHER FUNCTION =================
def get_weather(city):
    api_key = st.secrets["OPENWEATHER_API_KEY"]

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}

    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()

    return {
        "temp": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "condition": data["weather"][0]["description"],
        "rain": data.get("rain", {}).get("1h", 0)
    }

# ================= PREDICT DISEASE =================
def predict_disease(img):
    image = tf.keras.preprocessing.image.load_img(img, target_size=(128,128))
    arr = tf.keras.preprocessing.image.img_to_array(image)
    arr = np.expand_dims(arr, axis=0) / 255.0

    preds = disease_model.predict(arr)
    idx = int(np.argmax(preds))
    confidence = float(np.max(preds))

    return DISEASE_CLASSES[idx], confidence

# ================= PAGES =================
if page == "Home":
    st.markdown("<div class='box'>", unsafe_allow_html=True)
    st.header("🌾 FasalGPT")
    st.write("AI-powered smart agriculture assistant")
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Weather":
    st.markdown("<div class='box'>", unsafe_allow_html=True)
    st.header("🌦 Weather Advisory")

    city = st.text_input("Enter city", "Delhi")
    if st.button("Get Weather"):
        w = get_weather(city)
        st.metric("🌡 Temperature", f"{w['temp']} °C")
        st.metric("💧 Humidity", f"{w['humidity']} %")
        st.metric("🌧 Rainfall", f"{w['rain']} mm")
        st.write(f"☁ Condition: {w['condition']}")

    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Disease":
    st.markdown("<div class='box'>", unsafe_allow_html=True)
    st.header("🦠 Disease Detection")

    img = st.file_uploader("Upload leaf image", ["jpg","png","jpeg"])
    if img:
        st.image(img, use_container_width=True)
        if st.button("Analyze"):
            disease, conf = predict_disease(img)
            st.success(f"Disease: {disease}")
            st.info(f"Confidence: {conf*100:.2f}%")

    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Crop":
    st.markdown("<div class='box'>", unsafe_allow_html=True)
    st.header("🌾 Crop Recommendation")

    rain = st.slider("Rainfall", 0, 300, 100)
    temp = st.slider("Temperature", 0, 50, 25)
    ph = st.slider("Soil pH", 0.0, 14.0, 7.0)

    if st.button("Recommend"):
        if rain > 200:
            st.success("Rice")
        elif ph < 6.5:
            st.success("Potato")
        else:
            st.success("Wheat")

    st.markdown("</div>", unsafe_allow_html=True)

elif page in ["Soil", "Irrigation", "Schemes"]:
    st.markdown("<div class='box'>", unsafe_allow_html=True)
    st.info("Module coming soon")
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "About":
    st.markdown("<div class='box'>", unsafe_allow_html=True)
    st.write("FasalGPT – Smart agriculture assistant for farmers.")
    st.markdown("</div>", unsafe_allow_html=True)
