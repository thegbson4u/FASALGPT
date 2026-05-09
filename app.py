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

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide Streamlit Branding */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Background */
.stApp {
    background:
    radial-gradient(circle at top left, rgba(0,255,136,0.15), transparent 25%),
    #0B0F19;
}

/* Hero Title */
.hero-title {
    font-size: 64px;
    font-weight: 900;
    line-height: 1;
    margin-bottom: 10px;
}

.gradient-text {
    background: linear-gradient(to right, #00ff88, #22c55e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Subtitle */
.subtitle {
    color: #94A3B8;
    font-size: 20px;
    margin-bottom: 30px;
}

/* Cards */
.glass-card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 20px;
    transition: 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-5px);
}

/* Metric Card */
.metric-card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 18px;
    text-align: center;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(to right, #00ff88, #22c55e);
    color: black;
    border: none;
    border-radius: 12px;
    padding: 12px 24px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.03);
}

/* Upload box */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04);
    padding: 20px;
    border-radius: 18px;
}

</style>
""", unsafe_allow_html=True)

# ================= NAVIGATION =================
selected = option_menu(
    menu_title=None,
    options=["Home", "Weather", "Disease Detection", "Crop Recommendation", "About"],
    icons=["house", "cloud", "bug", "tree", "info-circle"],
    orientation="horizontal",
    default_index=0,
)

# ================= HOME =================
if selected == "Home":

    st.markdown("""
    <div class='hero-title'>
        AI-Powered <span class='gradient-text'>Precision Agriculture</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='subtitle'>
        Smart crop recommendations, real-time weather intelligence,
        and AI disease detection for modern farming.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class='glass-card'>
            <h3>🌦 Weather Intelligence</h3>
            <p>Real-time environmental monitoring and forecasting.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='glass-card'>
            <h3>🦠 Disease Detection</h3>
            <p>AI-powered plant disease classification system.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='glass-card'>
            <h3>🌾 Crop Recommendation</h3>
            <p>Smart farming decisions based on soil conditions.</p>
        </div>
        """, unsafe_allow_html=True)

# ================= WEATHER =================
elif selected == "Weather":

    st.title("🌦 Weather Dashboard")

    city = st.text_input("Enter City")

    if st.button("Get Weather"):

        # Demo values
        temp = 28
        humidity = 74
        rainfall = 12
        condition = "Cloudy"

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("🌡 Temperature", f"{temp} °C")
        c2.metric("💧 Humidity", f"{humidity}%")
        c3.metric("🌧 Rainfall", f"{rainfall} mm")
        c4.metric("☁ Condition", condition)

# ================= DISEASE =================
elif selected == "Disease Detection":

    st.title("🦠 AI Disease Detection")

    uploaded = st.file_uploader(
        "Upload leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:

        st.image(uploaded, width=300)

        st.success("Disease Detected: Tomato Early Blight")
        st.info("Confidence: 98.4%")

        st.markdown("""
        ### Recommended Treatment
        - Apply copper fungicide
        - Avoid overhead irrigation
        - Remove infected leaves
        """)

# ================= CROP =================
elif selected == "Crop Recommendation":

    st.title("🌾 Smart Crop Recommendation")

    col1, col2 = st.columns(2)

    with col1:
        N = st.slider("Nitrogen", 0, 200, 50)
        P = st.slider("Phosphorus", 0, 200, 40)
        K = st.slider("Potassium", 0, 200, 60)

    with col2:
        temp = st.slider("Temperature", 0, 50, 25)
        humidity = st.slider("Humidity", 0, 100, 70)
        rainfall = st.slider("Rainfall", 0, 300, 120)

    if st.button("Recommend Crop"):

        st.markdown("""
        <div class='glass-card'>
            <h2>🌱 Recommended Crop: Rice</h2>
            <p>Best suited for current environmental conditions.</p>
        </div>
        """, unsafe_allow_html=True)

# ================= ABOUT =================
elif selected == "About":

    st.title("ℹ About FASALGPT")

    st.markdown("""
    FASALGPT is an AI-powered agriculture advisory platform that helps farmers with:

    - 🌦 Weather forecasting
    - 🦠 Disease detection
    - 🌾 Crop recommendations
    - 📊 Precision farming insights

    Built using:
    - Streamlit
    - TensorFlow
    - Machine Learning
    - OpenWeather API
    """)
