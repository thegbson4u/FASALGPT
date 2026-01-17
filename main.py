# ================= IMPORTS =================
import streamlit as st
import tensorflow as tf
import numpy as np
import requests
from huggingface_hub import hf_hub_download


# ================= STREAMLIT CONFIG =================
st.set_page_config(
    page_title="FASALGPT | Digital Agriculture Advisory",
    page_icon="🌾",
    layout="wide",
)

# ================= HIDE STREAMLIT BRANDING =================
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
    background: linear-gradient(135deg, #0b0f1a, #111827);
    color: #eaeaea;
    font-family: 'Inter', sans-serif;
}

/* Glass cards */
.glass {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(14px);
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.15);
}

/* BIG BOLD HEADINGS (3x) */
.big-title {
    font-size: 3rem;
    font-weight: 800;
    margin-bottom: 10px;
}

/* BIG RESULT TEXT */
.big-result {
    font-size: 2.2rem;
    font-weight: 800;
    color: #7CFC98;
}

/* Metric text */
.metric-text {
    font-size: 1.5rem;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


# ================= TOP NAVIGATION =================
tabs = st.tabs([
    "🏠 Home",
    "🌦 Weather Advisory",
    "🦠 Disease Detection",
    "🌾 Crop Recommendation",
    "ℹ About"
])


# ================= LOAD DISEASE MODEL (HF) =================
@st.cache_resource
def load_disease_model():
    model_path = hf_hub_download(
        repo_id="THEGBSON/fasalgpt-disease-model",
        filename="trained_model.h5",
        revision="main"
    )
    return tf.keras.models.load_model(model_path, compile=False)

disease_model = load_disease_model()


# ================= DISEASE CLASSES (38) =================
CLASS_NAMES = [
    'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust',
    'Apple___healthy','Blueberry___healthy','Cherry___Powdery_mildew',
    'Cherry___healthy','Corn___Cercospora_leaf_spot','Corn___Common_rust',
    'Corn___Northern_Leaf_Blight','Corn___healthy','Grape___Black_rot',
    'Grape___Esca','Grape___Leaf_blight','Grape___healthy',
    'Orange___Haunglongbing','Peach___Bacterial_spot','Peach___healthy',
    'Pepper___Bacterial_spot','Pepper___healthy','Potato___Early_blight',
    'Potato___Late_blight','Potato___healthy','Raspberry___healthy',
    'Soybean___healthy','Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch','Strawberry___healthy',
    'Tomato___Bacterial_spot','Tomato___Early_blight',
    'Tomato___Late_blight','Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot','Tomato___Spider_mites',
    'Tomato___Target_Spot','Tomato___Yellow_Leaf_Curl_Virus',
    'Tomato___Mosaic_virus','Tomato___healthy'
]


# ================= WEATHER FUNCTION =================
def get_weather(city):
    try:
        api_key = st.secrets["OPENWEATHER_API_KEY"]
    except KeyError:
        return None

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}

    r = requests.get(url, params=params, timeout=10)
    if r.status_code != 200:
        return None

    data = r.json()
    return {
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "rainfall": data.get("rain", {}).get("1h", 0),
        "condition": data["weather"][0]["description"]
    }


# ================= DISEASE PREDICTION =================
def predict_disease(img):
    image = tf.keras.preprocessing.image.load_img(img, target_size=(128, 128))
    arr = tf.keras.preprocessing.image.img_to_array(image)
    arr = np.expand_dims(arr, axis=0)  # ❗ NO normalization

    preds = disease_model.predict(arr)
    idx = int(np.argmax(preds))
    conf = float(np.max(preds)) * 100

    return CLASS_NAMES[idx], conf


# ================= HOME =================
with tabs[0]:
    st.image("assets/banner.jpg", use_container_width=True)
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("<div class='big-title'>🌾 FASALGPT</div>", unsafe_allow_html=True)
    st.write("Government-grade AI platform for Indian agriculture.")
    st.markdown("</div>", unsafe_allow_html=True)


# ================= WEATHER =================
with tabs[1]:
    st.image("assets/banner1.jpg", use_container_width=True)
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("<div class='big-title'>🌦 Weather Advisory</div>", unsafe_allow_html=True)

    city = st.text_input("Enter City Name", "Delhi")
    if st.button("Get Weather"):
        weather = get_weather(city)
        if weather:
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"<div class='metric-text'>🌡 {weather['temperature']} °C</div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-text'>💧 {weather['humidity']} %</div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='metric-text'>🌧 {weather['rainfall']} mm</div>", unsafe_allow_html=True)
            c4.markdown(f"<div class='metric-text'>☁ {weather['condition']}</div>", unsafe_allow_html=True)
        else:
            st.error("Weather data unavailable")

    st.markdown("</div>", unsafe_allow_html=True)


# ================= DISEASE =================
with tabs[2]:
    st.image("assets/banner2.jpg", use_container_width=True)
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("<div class='big-title'>🦠 Disease Detection</div>", unsafe_allow_html=True)

    img = st.file_uploader("Upload Leaf Image", ["jpg", "png", "jpeg"])
    if img:
        st.image(img, use_container_width=True)
        if st.button("Analyze Disease"):
            disease, conf = predict_disease(img)
            st.markdown(f"<div class='big-result'>{disease}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-text'>Confidence: {conf:.2f}%</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ================= CROP =================
with tabs[3]:
    st.image("assets/banner3.jpg", use_container_width=True)
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("<div class='big-title'>🌾 Crop Recommendation</div>", unsafe_allow_html=True)

    N = st.slider("Nitrogen", 0, 200, 50)
    P = st.slider("Phosphorus", 0, 200, 50)
    K = st.slider("Potassium", 0, 200, 50)
    temp = st.slider("Temperature (°C)", 0, 50, 25)
    rain = st.slider("Rainfall (mm)", 0, 300, 100)

    if st.button("Recommend Crop"):
        crop = "Rice" if rain > 200 else "Wheat" if temp < 20 else "Maize"
        st.markdown(f"<div class='big-result'>{crop}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ================= ABOUT =================
with tabs[4]:
    st.image("assets/banner.jpg", use_container_width=True)
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("<div class='big-title'>ℹ About FASALGPT</div>", unsafe_allow_html=True)
    st.write("Secure AI-powered agriculture advisory system. No developer identity exposed.")
    st.markdown("</div>", unsafe_allow_html=True)
