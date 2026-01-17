# ================= IMPORTS =================
import streamlit as st
import tensorflow as tf
import numpy as np
import requests
import os
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


# ================= LOAD DISEASE MODEL =================
@st.cache_resource
def load_disease_model():
    model_path = hf_hub_download(
        repo_id="THEGBSON/fasalgpt-disease-model",
        filename="trained_model.h5",   # ✅ exact file name on HF
        revision="main"
    )
    return tf.keras.models.load_model(model_path, compile=False)

disease_model = load_disease_model()


# ================= DISEASE CLASS NAMES (38 CLASSES) =================
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


# ================= WEATHER FUNCTION (STREAMLIT CLOUD SAFE) =================
def get_weather(city):
    try:
        api_key = st.secrets["OPENWEATHER_API_KEY"]
    except KeyError:
        st.error("Weather API key not configured")
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
    image = tf.keras.preprocessing.image.load_img(
        img, target_size=(128, 128)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(image)

    # ❌ DO NOT NORMALIZE
    # img_array = img_array / 255.0  <-- REMOVE THIS

    img_array = np.expand_dims(img_array, axis=0)

    preds = disease_model.predict(img_array)
    class_index = int(np.argmax(preds))
    confidence = float(np.max(preds)) * 100

    return CLASS_NAMES[class_index], confidence


# ================= HOME =================
with tabs[0]:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.header("🌾 FASALGPT")
    st.write("Government-grade AI platform for Indian agriculture.")
    st.markdown("""
    ✔ Real-time weather advisory  
    ✔ AI-based crop disease detection  
    ✔ Intelligent crop recommendation  
    ✔ Secure & privacy-safe  
    """)
    st.markdown("</div>", unsafe_allow_html=True)


# ================= WEATHER =================
with tabs[1]:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.header("🌦 Weather Advisory")

    city = st.text_input("Enter City Name", "Delhi")

    if st.button("Get Weather"):
        weather = get_weather(city)
        if weather:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🌡 Temperature", f"{weather['temperature']} °C")
            c2.metric("💧 Humidity", f"{weather['humidity']} %")
            c3.metric("🌧 Rainfall", f"{weather['rainfall']} mm")
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
        st.image(img, caption="Uploaded Leaf Image", use_container_width=True)

        if st.button("Analyze Disease"):
            disease, conf = predict_disease(img)
            st.success(f"🌱 Detected Disease: **{disease}**")
            st.info(f"Confidence: {conf:.2f}%")

    st.markdown("</div>", unsafe_allow_html=True)


# ================= CROP RECOMMENDATION =================
with tabs[3]:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.header("🌾 Crop Recommendation")

    N = st.slider("Nitrogen", 0, 200, 50)
    P = st.slider("Phosphorus", 0, 200, 50)
    K = st.slider("Potassium", 0, 200, 50)
    temp = st.slider("Temperature (°C)", 0, 50, 25)
    rain = st.slider("Rainfall (mm)", 0, 300, 100)

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
    st.write("""
    FASALGPT is a secure AI-powered agriculture advisory system.
    No personal data, identity, or developer information is exposed.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

