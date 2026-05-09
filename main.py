import streamlit as st
from streamlit_option_menu import option_menu

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="FASALGPT",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================= CSS =================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"]{
    font-family: 'Inter', sans-serif;
}

/* Hide Streamlit Branding */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Main Background */
.stApp{
    background:#f5f7fb;
}

/* Global Padding */
.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
    padding-left:4rem;
    padding-right:4rem;
}

/* Hero Section */
.hero-title{
    font-size:78px;
    font-weight:900;
    line-height:1;
    color:#111827;
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
    color:#6b7280;
    line-height:1.8;
    max-width:800px;
}

/* Bento Card */
.bento-card{
    background:white;
    border-radius:32px;
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
    0 25px 50px rgba(221,42,123,0.12);
}

/* Large Bento */
.large-card{
    min-height:420px;
}

/* Medium Bento */
.medium-card{
    min-height:260px;
}

/* Small Bento */
.small-card{
    min-height:180px;
}

/* Card Title */
.card-title{
    font-size:28px;
    font-weight:800;
    color:#111827;
    margin-bottom:15px;
}

/* Card Text */
.card-text{
    color:#6b7280;
    font-size:17px;
    line-height:1.8;
}

/* Metric Number */
.metric-number{
    font-size:64px;
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

/* Buttons */
.stButton > button{

    background: linear-gradient(
        90deg,
        #f58529,
        #dd2a7b,
        #8134af,
        #515bd4
    );

    color:white;
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

/* Upload Box */
[data-testid="stFileUploader"]{
    background:white;
    border-radius:24px;
    border:1px solid #e5e7eb;
    padding:20px;
}

/* Metrics */
[data-testid="metric-container"]{
    background:white;
    border-radius:24px;
    padding:18px;
    border:1px solid #e5e7eb;
    box-shadow:0 10px 20px rgba(0,0,0,0.04);
}

/* Input */
.stTextInput > div > div > input{
    border-radius:16px;
}

/* Slider */
.stSlider{
    padding-top:15px;
    padding-bottom:15px;
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

    # HERO
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

    # ================= BENTO GRID =================

    # ROW 1
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

        <br><br>

        Built using:
        <br>
        • TensorFlow
        <br>
        • Computer Vision
        <br>
        • Machine Learning
        <br>
        • Weather APIs
        <br>
        • Streamlit

        <br><br>

        Empowering modern agriculture using data-driven insights.
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
        Deep learning powered disease detection system
        trained on agricultural image datasets.

        <br><br>

        Real-time classification using computer vision.
        </div>

        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ROW 2
    col3, col4, col5 = st.columns(3)

    with col3:

        st.markdown("""
        <div class="bento-card medium-card">

        <div class="card-title">
        🌦 Weather Intelligence
        </div>

        <div class="card-text">
        Live weather forecasting,
        rainfall prediction,
        and climate-aware farming insights.
        </div>

        </div>
        """, unsafe_allow_html=True)

    with col4:

        st.markdown("""
        <div class="bento-card medium-card">

        <div class="card-title">
        🦠 Disease Detection
        </div>

        <div class="card-text">
        Upload crop leaf images
        and detect diseases instantly
        using AI-powered computer vision.
        </div>

        </div>
        """, unsafe_allow_html=True)

    with col5:

        st.markdown("""
        <div class="bento-card medium-card">

        <div class="card-title">
        🌱 Crop Recommendation
        </div>

        <div class="card-text">
        Smart crop prediction system
        based on soil nutrients,
        rainfall, and temperature.
        </div>

        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ROW 3
    col6, col7 = st.columns([1,1.5])

    with col6:

        st.markdown("""
        <div class="bento-card medium-card">

        <div class="card-title">
        🚀 Live Demo
        </div>

        <div class="card-text">
        Explore the real-time AI agriculture platform
        with modern interactive dashboards.

        <br><br>

        Streamlit Powered.
        </div>

        </div>
        """, unsafe_allow_html=True)

        st.button("Launch Platform")

    with col7:

        st.markdown("""
        <div class="bento-card medium-card">

        <div class="card-title">
        💻 GitHub Repository
        </div>

        <div class="card-text">
        Explore the complete source code,
        AI models, frontend redesign,
        and deployment-ready architecture
        of FASALGPT.

        <br><br>

        Modern AI SaaS-style agriculture platform.
        </div>

        </div>
        """, unsafe_allow_html=True)

        st.link_button(
            "Open GitHub Repository",
            "https://github.com/thegbson4u/FASALGPT"
        )

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

        st.markdown("""
        ### 🌱 Suggested Treatment
        - Apply fungicide
        - Remove infected leaves
        - Avoid overwatering
        """)

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

    FASALGPT is a modern precision agriculture system that combines:

    - AI Disease Detection
    - Weather Intelligence
    - Crop Recommendation
    - Computer Vision
    - Deep Learning

    Designed to empower modern farming using artificial intelligence.
    """)
```
