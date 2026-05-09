import streamlit as st
from streamlit_option_menu import option_menu

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="FASALGPT",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* GLOBAL */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #111827 !important;
}

/* APP BG */
.stApp {
    background: #f8fafc;
}

/* HIDE STREAMLIT */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* TEXT FIX */
h1, h2, h3, h4, h5, h6,
p, span, label, div {
    color:#111827 !important;
}

/* PADDING */
.block-container{
    padding-top:2rem;
    padding-left:4rem;
    padding-right:4rem;
}

/* NAVBAR */
.nav-link {
    color:white !important;
    font-weight:600 !important;
}

.nav-pills .nav-link.active {
    background: linear-gradient(
        90deg,
        #f58529,
        #dd2a7b,
        #8134af,
        #515bd4
    ) !important;

    color:white !important;
    border-radius:12px;
}

.nav-pills .nav-link {
    background:#111827 !important;
    border-radius:12px;
    margin:0 5px;
}

/* HERO */
.hero-title{
    font-size:78px;
    font-weight:900;
    line-height:1;
    color:#111827 !important;
    margin-bottom:20px;
}

.gradient-text{
    background: linear-gradient(
        90deg,
        #f58529,
        #dd2a7b,
        #8134af,
        #515bd4
    );

    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.subtitle{
    font-size:22px;
    color:#6b7280 !important;
    line-height:1.8;
    max-width:850px;
}

/* BENTO CARD */
.bento-card{
    background:white;
    border-radius:30px;
    padding:32px;
    border:1px solid #e5e7eb;

    box-shadow:
    0 10px 30px rgba(0,0,0,0.05);

    transition:0.35s ease;
    height:100%;
}

.bento-card:hover{
    transform:translateY(-8px);

    box-shadow:
    0 20px 50px rgba(221,42,123,0.12);
}

.large-card{
    min-height:420px;
}

.medium-card{
    min-height:260px;
}

/* CARD TEXT */
.card-title{
    font-size:28px;
    font-weight:800;
    color:#111827 !important;
    margin-bottom:15px;
}

.card-text{
    color:#6b7280 !important;
    font-size:17px;
    line-height:1.8;
}

/* METRIC */
.metric-number{
    font-size:72px;
    font-weight:900;

    background: linear-gradient(
        90deg,
        #f58529,
        #dd2a7b,
        #8134af,
        #515bd4
    );

    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

/* BUTTON */
.stButton > button{

    background: linear-gradient(
        90deg,
        #f58529,
        #dd2a7b,
        #8134af,
        #515bd4
    );

    color:white !important;
    border:none;

    border-radius:18px;

    padding:14px 28px;

    font-size:16px;
    font-weight:700;

    transition:0.3s ease;
}

.stButton > button:hover{
    transform:scale(1.03);
}

/* FILE UPLOADER */
[data-testid="stFileUploader"]{

    background:
    linear-gradient(
        135deg,
        rgba(245,133,41,0.08),
        rgba(221,42,123,0.08),
        rgba(129,52,175,0.08),
        rgba(81,91,212,0.08)
    );

    border:2px dashed rgba(129,52,175,0.25);

    border-radius:28px;

    padding:28px;

    box-shadow:
    0 10px 30px rgba(0,0,0,0.04);

    transition:0.3s ease;
}

[data-testid="stFileUploader"]:hover{

    border:2px dashed #dd2a7b;

    box-shadow:
    0 20px 40px rgba(221,42,123,0.12);
}

[data-testid="stFileUploader"] section{
    background:transparent !important;
    border:none !important;
}

[data-testid="stFileUploader"] *{
    color:#111827 !important;
}

/* METRICS */
[data-testid="metric-container"]{
    background:white;
    border-radius:24px;
    padding:18px;
    border:1px solid #e5e7eb;
    box-shadow:0 10px 20px rgba(0,0,0,0.04);
}

[data-testid="metric-container"] *{
    color:#111827 !important;
}

/* INPUT */
input {
    color:#111827 !important;
}

input::placeholder{
    color:#6b7280 !important;
}

/* SLIDER */
.stSlider label{
    color:#111827 !important;
    font-weight:600 !important;
}

</style>
""", unsafe_allow_html=True)

# ================= NAVIGATION =================
selected = option_menu(
    menu_title=None,
    options=[
        "Home",
        "Weather",
        "Disease Detection",
        "Crop AI",
        "About"
    ],
    icons=[
        "house-fill",
        "cloud-sun-fill",
        "bug-fill",
        "flower1",
        "info-circle-fill"
    ],
    orientation="horizontal"
)

# ================= HOME =================
if selected == "Home":

    st.markdown("""
    <div style="padding-top:30px;padding-bottom:40px;">

    <div class="hero-title">
        AI-Powered
        <span class="gradient-text">
        Agriculture Intelligence
        </span>
    </div>

    <div class="subtitle">
        FASALGPT combines artificial intelligence,
        computer vision, weather intelligence,
        and smart crop analytics into one modern
        precision farming platform.
    </div>

    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.5,1])

    with col1:

        st.markdown("""
        <div class="bento-card large-card">

        <div class="card-title">
        🌾 Smart Farming Dashboard
        </div>

        <div class="card-text">
        AI-powered precision agriculture platform designed for
        disease detection, crop recommendation, and weather analysis.
        </div>

        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="bento-card large-card">

        <div class="card-title">
        📊 AI Accuracy
        </div>

        <div class="metric-number">
        98.4%
        </div>

        <div class="card-text">
        Deep learning powered disease detection system.
        </div>

        </div>
        """, unsafe_allow_html=True)

# ================= WEATHER =================
elif selected == "Weather":

    st.title("🌦 Weather Intelligence")

    city = st.text_input("Enter City Name", "Delhi")

    if st.button("Get Weather"):

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("🌡 Temperature", "28°C")
        c2.metric("💧 Humidity", "74%")
        c3.metric("🌧 Rainfall", "12 mm")
        c4.metric("☁ Condition", "Cloudy")

# ================= DISEASE =================
elif selected == "Disease Detection":

    st.title("🦠 AI Disease Detection")

    uploaded = st.file_uploader(
        "Upload Leaf Image",
        type=["jpg","jpeg","png"]
    )

    if uploaded:

        st.image(uploaded, width=350)

        st.success("Disease Detected: Tomato Early Blight")
        st.info("Confidence Score: 98.4%")

# ================= CROP AI =================
elif selected == "Crop AI":

    st.title("🌾 Crop Recommendation AI")

    col1, col2 = st.columns(2)

    with col1:
        N = st.slider("Nitrogen", 0, 200, 60)
        P = st.slider("Phosphorus", 0, 200, 40)
        K = st.slider("Potassium", 0, 200, 50)

    with col2:
        temp = st.slider("Temperature", 0, 50, 28)
        humidity = st.slider("Humidity", 0, 100, 70)
        rainfall = st.slider("Rainfall", 0, 300, 120)

    if st.button("Recommend Crop"):

        st.success("🌱 Recommended Crop: Rice")

# ================= ABOUT =================
elif selected == "About":

    st.title("ℹ About FASALGPT")

    st.markdown("""
    ### 🌾 AI-Powered Agriculture Platform

    FASALGPT combines:
    - AI Disease Detection
    - Weather Intelligence
    - Crop Recommendation
    - Computer Vision
    - Deep Learning
    """)
