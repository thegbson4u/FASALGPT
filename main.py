# ================= IMPORTS =================
import streamlit as st
import tensorflow as tf
import numpy as np
import requests
from huggingface_hub import hf_hub_download

# ================= CONFIG =================
st.set_page_config(
    page_title="FASALGPT | Digital Agriculture Advisory",
    page_icon="🌾",
    layout="wide",
)

# ================= HIDE STREAMLIT BRANDING =================
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ================= GLOBAL CSS (INSPIRED UI) =================
st.markdown("""
<style>
:root {
    --green: #2e7d32;
    --glow: #00ff88;
    --dark: #121212;
    --card: #1e1e1e;
}

html, body {
    background: var(--dark);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

/* SECTION TITLES */
.big-title {
    font-size: 48px;
    font-weight: 900;
    margin-bottom: 10px;
}

.subtext {
    color: #b0b0b0;
    font-size: 18px;
}

/* GLASS CARD */
.card {
    background: var(--card);
    padding: 30px;
    border-radius: 16px;
    margin-bottom: 30px;
}

/* METRIC BOX */
.metric {
    background: rgba(255,255,255,0.06);
    padding: 16px;
    border-radius: 12px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ================= TOP NAV (FUNCTIONAL) =================
tabs = st.tabs([
    "🏠 Home",
    "🌦 Weather Advisory",
    "🦠 Disease Detection",
    "🌾 Crop Recommendation",
    "ℹ About"
])

# ================= LOAD MODEL =================
@st.cache_resource
def load_disease_model():
    path = hf_hub_download(
        repo_id="THEGBSON/fasalgpt-disease-model",
        filename="trained_model.h5"
    )
    return tf.keras.models.load_model(path, compile=False)

disease_model = load_disease_model()

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
    except:
        st.error("Weather API key not configured")
        return None

    url = "https://api.openweathermap.org/data/2.5/weather"
    r = requests.get(url, params={
        "q": city,
        "appid": api_key,
        "units": "metric"
    }, timeout=10)

    if r.status_code != 200:
        return None

    d = r.json()
    return {
        "temp": d["main"]["temp"],
        "humidity": d["main"]["humidity"],
        "rain": d.get("rain", {}).get("1h", 0),
        "condition": d["weather"][0]["description"]
    }

# ================= HOME =================
with tabs[0]:
    st.image("assets/banner.jpg", use_container_width=True)
    st.markdown("<div class='big-title'>FASALGPT</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtext'>AI-powered precision agriculture platform</div>", unsafe_allow_html=True)

    st.markdown("""
    ✔ Disease detection  
    ✔ Weather advisory  
    ✔ Crop recommendation  
    ✔ Government-ready system  
    """)

# ================= WEATHER =================
with tabs[1]:
    st.image("assets/banner1.jpg", use_container_width=True)
    st.markdown("<div class='big-title'>Weather Advisory</div>", unsafe_allow_html=True)

    city = st.text_input("Enter City", "Delhi")
    if st.button("Get Weather"):
        w = get_weather(city)
        if w:
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("🌡 Temp", f"{w['temp']} °C")
            c2.metric("💧 Humidity", f"{w['humidity']} %")
            c3.metric("🌧 Rain", f"{w['rain']} mm")
            c4.metric("☁ Condition", w["condition"])
        else:
            st.error("Unable to fetch weather")

# ================= DISEASE =================
with tabs[2]:
    st.image("assets/banner2.jpg", use_container_width=True)
    st.markdown("<div class='big-title'>Disease Detection</div>", unsafe_allow_html=True)

    img = st.file_uploader("Upload leaf image", ["jpg","png","jpeg"])
    if img:
        st.image(img, use_container_width=True)
        image = tf.keras.preprocessing.image.load_img(img, target_size=(128,128))
        arr = tf.keras.preprocessing.image.img_to_array(image)
        arr = np.expand_dims(arr, axis=0)

        preds = disease_model.predict(arr)
        idx = int(np.argmax(preds))
        conf = float(np.max(preds))*100

        st.success(f"Detected Disease: **{CLASS_NAMES[idx]}**")
        st.info(f"Confidence: {conf:.2f}%")

# ================= CROP =================
with tabs[3]:
    st.markdown("<div class='big-title'>Crop Recommendation</div>", unsafe_allow_html=True)

    N = st.slider("Nitrogen", 0, 200, 50)
    P = st.slider("Phosphorus", 0, 200, 50)
    K = st.slider("Potassium", 0, 200, 50)
    temp = st.slider("Temperature", 0, 50, 25)
    rain = st.slider("Rainfall", 0, 300, 100)

    if st.button("Recommend"):
        if rain > 200:
            crop = "Rice"
        elif temp < 20:
            crop = "Wheat"
        else:
            crop = "Maize"
        st.success(f"Recommended Crop: **{crop}**")

# ================= ABOUT =================
with tabs[4]:
    st.markdown("<div class='big-title'>About FASALGPT</div>", unsafe_allow_html=True)
    st.write("Secure AI agriculture advisory platform. No personal data exposed.")
