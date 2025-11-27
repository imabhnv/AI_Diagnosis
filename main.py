# app.py - FULL PROFESSIONAL OPTIMIZED VERSION
# ZERO RANDOM LOADING | ZERO FREEZE | HEAVY ML UI STRUCTURE

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import google.generativeai as genai
import pytesseract
import PyPDF2

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="AI-Assisted Pneumonia Diagnosis System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM PROFESSIONAL UI STYLING
# ============================================================
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
}
.main-title {
    font-size: 42px;
    font-weight: 700;
    color: #00e5ff;
}
.sub-title {
    font-size: 18px;
    color: #ffffffb3;
}
.section-card {
    background: #ffffff10;
    padding: 25px;
    border-radius: 18px;
    margin-bottom: 25px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# ============================================================
# MODEL LOADING FROM GOOGLE DRIVE (USING st.secrets)
# ============================================================
import os
import gdown

MODEL_PATH = "our_model.h5"
DRIVE_FILE_ID = st.secrets.get("MODEL_FILE_ID", None)

@st.cache_resource
def load_pneumonia_model():
    if not os.path.exists(MODEL_PATH):
        if not DRIVE_FILE_ID:
            st.error("MODEL_FILE_ID not found in st.secrets")
            st.stop()
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        with st.spinner("Downloading AI model from Google Drive..."):
            gdown.download(url, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

model = load_pneumonia_model()

# ============================================================
# GEMINI CONFIGURATION
# ============================================================
gemini_api_key = st.secrets.get("GEMINI_API_KEY", None)

if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")
else:
    gemini_model = None


def call_gemini(prompt):
    if not gemini_model:
        return "Gemini API key not configured in Streamlit secrets."
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ AI unavailable: {str(e)}"

# ============================================================
# IMAGE PREDICTION FUNCTION
# ============================================================

def predict_pneumonia(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = img.resize((224, 224))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    confidence = round(float(prediction) * 100, 2)

    if prediction > 0.5:
        return "PNEUMONIA", confidence
    else:
        return "NORMAL", 100 - confidence

# ============================================================
# PAGE HEADER
# ============================================================
st.markdown('<div class="main-title">🧠 AI Pneumonia Detection & Medical Intelligence System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">End-to-End Deep Learning Diagnostic Framework</div>', unsafe_allow_html=True)

# ============================================================
# APPLICATION TABS
# ============================================================
tab1, tab2 = st.tabs([
    "🫁 X-Ray Pneumonia Detection",
    "🤖 AI Medical Assistant & Report Analyzer"
])

# ============================================================
# TAB 1 : X-RAY ANALYSIS
# ============================================================
with tab1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📤 Upload Chest X-Ray")

    uploaded_file = st.file_uploader("Upload Chest X-Ray Image", type=['jpg','jpeg','png'])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded X-Ray", use_container_width=True)

        if st.button("🔍 Run AI Diagnosis"):
            with st.spinner("Analyzing X-Ray using Deep Learning model..."):
                result, confidence = predict_pneumonia(img)

                st.success(f"Diagnosis Result: {result}")
                st.info(f"Confidence Level: {confidence}%")

                prompt = f"""
You are an expert medical AI.
Diagnosis Result: {result}
Confidence: {confidence}%
Generate a structured medical report including:
- Clinical Explanation
- Possible Causes
- Patient Precautions
- Next Steps
- When to Consult Doctor
"""

                ai_report = call_gemini(prompt)

                st.markdown("### 🩺 AI Generated Medical Report")
                st.write(ai_report)

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# TAB 2 : CHATBOT + REPORT ANALYZER + RISK ENGINE
# ============================================================
with tab2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("💬 AI Medical Assistant")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    with st.form("chat_form"):
        user_input = st.text_input("Describe symptoms or ask medical question")
        send_btn = st.form_submit_button("Send")

    if send_btn and user_input.strip() != "":
        st.session_state.chat_history.append(f"USER: {user_input}")

        with st.spinner("AI generating response..."):
            reply = call_gemini("\n".join(st.session_state.chat_history))

        st.session_state.chat_history.append(f"AI: {reply}")

    for msg in st.session_state.chat_history:
        st.markdown(msg)

    st.markdown("---")
    st.subheader("📄 Medical Report / Prescription Analyzer")

    report_file = st.file_uploader("Upload Medical Report (PDF/Image)", type=['jpg','jpeg','png','pdf'])

    def extract_text_from_pdf(file):
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    def extract_text_from_image(file):
        img = Image.open(file)
        return pytesseract.image_to_string(img)

    if report_file:
        with st.spinner("Extracting and analyzing report..."):
            if report_file.type == "application/pdf":
                content = extract_text_from_pdf(report_file)
            else:
                content = extract_text_from_image(report_file)

            prompt = f"""
You are an advanced medical AI.
Below is extracted medical text:
{content}
Analyze and provide:
- Key Findings
- Abnormal Values
- Simplified Explanation
- Recommended Actions
- Urgency Level
"""
            analysis = call_gemini(prompt)
            st.markdown("### 🧾 AI Report Analysis")
            st.write(analysis)

    st.markdown("---")
    st.subheader("❤️ Health Risk Analyzer")

    if 'risk_stage' not in st.session_state:
        st.session_state.risk_stage = 0

    if st.session_state.risk_stage == 0:
        with st.form("risk_init"):
            disease = st.text_input("Enter disease name")
            gender = st.selectbox("Biological Sex", ["Male", "Female", "Other"])
            start_btn = st.form_submit_button("Start Risk Assessment")
        if start_btn and disease:
            question_prompt = f"Generate 10 yes/no diagnostic questions for {disease} for {gender}"
            questions = call_gemini(question_prompt).split("\n")[:10]
            st.session_state.risk_questions = questions
            st.session_state.risk_answers = {}
            st.session_state.risk_stage = 1

    elif st.session_state.risk_stage == 1:
        with st.form("risk_questions_form"):
            for i, q in enumerate(st.session_state.risk_questions):
                st.session_state.risk_answers[i] = st.radio(q, ["Yes", "No"], horizontal=True, key=f"risk_{i}")
            submit_risk = st.form_submit_button("Calculate Risk")
        if submit_risk:
            risk_prompt = f"Patient Answers: {st.session_state.risk_answers}. Calculate health risk percentage and conclusion."
            st.session_state.risk_result = call_gemini(risk_prompt)
            st.session_state.risk_stage = 2

    elif st.session_state.risk_stage == 2:
        st.success("📊 Health Risk Result")
        st.write(st.session_state.risk_result)
        if st.button("Restart Analysis"):
            st.session_state.risk_stage = 0

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("✅ Powered by Deep Learning & AI - Professional Medical System")
