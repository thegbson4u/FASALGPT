# ================= IMPORTS =================
import streamlit as st
import tensorflow as tf
import numpy as np
import requests
import os
from huggingface_hub import hf_hub_download

# ================= CONFIG =================
st.set_page_config(
    page_title="FasalGPT | AI-Powered Precision Agriculture",
    page_icon="🌾",
    layout="wide",
)

# ================= HIDE STREAMLIT BRANDING =================
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ================= GLOBAL THEME (HTML INSPIRED) =================
st.markdown("""
<style>
:root {
    --primary-green: #2e7d32;
    --accent-glow: #00ff88;
    --dark-bg: #121212;
    --card-bg: #1e1e1e;
    --text-gray: #b0b0b0;
}

html, body {
    background: var(--dark-bg);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

/* NAVBAR */
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px 6%;
    background: rgba(0,0,0,0.85);
    position: sticky;
    top: 0;
    z-index: 999;
}

.logo {
    font-size: 26px;
    font-weight: 900;
    color: var(--accent-glow);
}

.nav-links span {
    margin-left: 22px;
    font-size: 14px;
    cursor: pointer;
    color: white;
}

/* HERO */
.hero {
    height: 80vh;
    background:
        linear-gradient(rgba(0,0,0,0.65), rgba(0,0,0,0.65)),
        url("assets/banner.jpg") center/cover no-repeat;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
}

.hero h1 {
    font-size: 3.8rem;
    margin-bottom: 10px;
}

.hero p {
    color: var(--text-gray);
    font-size: 1.2rem;
    max-width: 650px;
    margin: auto;
}

.btn {
    background: var(--primary-green);
    color: white;
    padding: 12px 32px;
    border-radius: 6px;
    margin: 10px;
    display: inline-block;
    font-weight: 600;
}

.btn-outline {
    background: transparent;
    border: 1px solid white;
}

/* FEATURES */
.features {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px,1fr));
    gap: 30px;
    padding: 90px 10%;
}

.card {
    background: var(--card-bg);
    padding: 32px;
    border-radius: 16px;
    text-align: center;
    transition: 0.3s;
    border-bottom: 3px solid transparent;
}

.card:hover {
    transform: translateY(-10px);
    border-bottom: 3px solid var(--accent-glow);
}

.card h3 {
    color: var(--accent-glow);
}

/* SECTION TITLE */
.section-title {
    font-size: 46px;
    font-weight: 900;
    margin-bottom: 30px;
}
</style>
""", unsafe_allow_html=True)

# ================= NAVBAR =================
st.markdown("""
<div class="navbar">
    <div class="logo">FasalGPT</div>
    <div class="nav-links">
        <span>Home</span>
        <span>Weather</span>
        <span>Disease</span>
        <span>Crop</span>
        <span>About</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ================= HERO =================
st.markdown("""
<section class="hero">
    <div>
        <h1>FasalGPT</h1>
        <p>Harnessing AI to bring sustainable, profitable, and precision agriculture to every Indian farmer.</p>
        <div>
            <span class="btn">Start Analysis</span>
            <span class="btn btn-outline">Learn More</span>
        </div>
    </div>
</section>
""", unsafe_allow_html=True)

# ================= FEATURES =================
st.markdown("""
<section class="features">
    <div class="card">
        <h3>Plant Health</h3>
        <p>Detect diseases early and prevent crop loss.</p>
    </div>
    <div class="card">
        <h3>AI Powered</h3>
        <p>Deep learning models trained on 38 crop diseases.</p>
    </div>
    <div class="card">
        <h3>Increase Yield</h3>
        <p>Smart recommendations for higher productivity.</p>
    </div>
</section>
""", unsafe_allow_html=True)

# ================= LOAD DISEASE MODEL =================
@st.cache_resource
def load_disease_model():
    model_path = hf_hub_download(
        repo_id="THEGBSON/fasalgpt-disease-model",
        filename="trained_model.h5"
    )
    return tf.keras.models.load_model(model_path, compile=False)

disease_model = load_disease_model()

# ================= DISEASE LABELS =================
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

# ================= DISEASE DETECTION =================
st.markdown("<div class='section-title'>🦠 Disease Detection</div>", unsafe_allow_html=True)
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
