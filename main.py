# app.py - FULL CODE WITH GOOGLE DRIVE MODEL LOADING (gdown + st.secrets)

import streamlit as st
import numpy as np
import os
import gdown
from tensorflow.keras.models import load_model
from PIL import Image
import google.generativeai as genai
import pytesseract
import PyPDF2

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="AI Pneumonia Diagnosis System",
    page_icon="🫁",
    layout="wide"
)

# ---------------------- CUSTOM CSS ----------------------
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
}
.title {
    font-size: 40px;
    font-weight: bold;
    color: #00e5ff;
}
.sub {
    font-size: 18px;
    color: #ffffff;
}
.card {
    background: #ffffff15;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------- MODEL LOADING FROM GOOGLE DRIVE ----------------------
MODEL_PATH = "our_model.h5"
DRIVE_FILE_ID = st.secrets.get("MODEL_FILE_ID", None)

@st.cache_resource
def load_pneumonia_model():
    if not os.path.exists(MODEL_PATH):
        if not DRIVE_FILE_ID:
            st.error("MODEL_FILE_ID missing in st.secrets")
            st.stop()
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        with st.spinner("Downloading AI model from Google Drive..."):
            gdown.download(url, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

model = load_pneumonia_model()

# ---------------------- IMAGE PREDICTION ----------------------
def predict_pneumonia(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = img.resize((224, 224))
    img_array = np.array(img).astype('float32') / 255.0
    if img_array.ndim == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    confidence = round(float(prediction) * 100, 2)

    if prediction > 0.5:
        return "PNEUMONIA", confidence
    else:
        return "NORMAL", 100 - confidence

# ---------------------- GEMINI SDK CONFIG ----------------------
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
    except Exception:
        return "⚠️ Gemini is currently busy or unavailable. Please try again shortly."

# ---------------------- AI DIAGNOSIS GENERATOR ----------------------
def ai_diagnosis(result, confidence):
    prompt = f"""
You are a professional medical AI assistant.

Chest X-ray Result:
Diagnosis: {result}
Confidence: {confidence}%

Provide:
1. Clinical explanation
2. Possible causes
3. Precautions
4. Next steps
5. When to consult a doctor

Keep it medically accurate and easy to understand.
"""
    return call_gemini(prompt)

# ---------------------- CHATBOT ENGINE ----------------------

def chatbot_response(user_input):
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    st.session_state.chat_history.append({"role": "user", "content": user_input})

    conversation = "You are a helpful, professional medical AI assistant.\n"
    for msg in st.session_state.chat_history:
        conversation += f"{msg['role'].upper()}: {msg['content']}\n"

    reply = call_gemini(conversation)
    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    return reply

# ========================== UI ==========================

st.markdown('<div class="title">🧠 AI-Assisted Diagnosis of Pneumonia From Chest Radiograph.</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Deep Learning Medical AI</div>', unsafe_allow_html=True)

xtab1, xtab2 = st.tabs([
    "🫁 X-Ray Pneumonia Detection",
    "🤖 AI Medical Assistant"
])

# ---------------- TAB 1 : X-RAY DETECTION ----------------
with xtab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Upload Chest X-Ray for AI Diagnosis")

    if 'ai_report' not in st.session_state:
        st.session_state.ai_report = None

    uploaded_file = st.file_uploader("Select Chest X-Ray Image", type=['jpg','jpeg','png'])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded X-Ray", use_container_width=True)

        if st.button("🔍 Analyze X-Ray"):
            result, confidence = predict_pneumonia(img)
            st.success(f"Diagnosis Result: {result}")
            st.info(f"Confidence Level: {confidence}%")
            st.session_state.ai_report = ai_diagnosis(result, confidence)

    if st.session_state.ai_report:
        st.markdown("### 🩺 AI Medical Diagnosis Report")
        st.write(st.session_state.ai_report)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- TAB 2 : AI CHATBOT ----------------
with xtab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("AI Medical Assistant")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    user_prompt = st.text_input(
        "Ask about symptoms, treatment or health concerns:",
        key="chat_input"
    )

    if st.button("Send", key="chat_send") and user_prompt:
        response = chatbot_response(user_prompt)
        st.session_state.messages.append((user_prompt, response))

    for user_msg, bot_msg in st.session_state.messages:
        st.markdown(f"**You:** {user_msg}")
        st.markdown(f"**AI:** {bot_msg}")

    st.markdown("---")

    st.subheader("📄 Medical Report / Prescription Analyzer")

    report_file = st.file_uploader(
        "Upload Medical Report or Prescription (PDF/Image)",
        type=['jpg','jpeg','png','pdf'],
        key="report_upload"
    )

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
        st.info("Analyzing report using Gemini AI...")
        if report_file.type == "application/pdf":
            extracted_text = extract_text_from_pdf(report_file)
        else:
            extracted_text = extract_text_from_image(report_file)

        if extracted_text and len(extracted_text.strip()) > 50:
            report_prompt = f"""
Analyze the following medical report text and provide:
1. Key Findings
2. Abnormal Values
3. Simplified Explanation
4. Recommended Next Steps
5. Urgency Level

{extracted_text}
"""
            analysis_output = call_gemini(report_prompt)
            st.markdown("### 🧠 AI Analysis Result")
            st.write(analysis_output)
        else:
            st.warning("Unable to extract readable text from report. Upload clearer file.")

        st.markdown("---")

    # ================= HEALTH RISK ANALYZER =================
    st.subheader("❤️ Health Risk Analyzer (0-100)")
    st.caption("Doctor-style risk assessment with 10 precise follow-up questions")

    if 'risk_stage' not in st.session_state:
        st.session_state.risk_stage = 0
    if 'risk_disease' not in st.session_state:
        st.session_state.risk_disease = ""
    if 'risk_questions' not in st.session_state:
        st.session_state.risk_questions = []
    if 'risk_answers' not in st.session_state:
        st.session_state.risk_answers = {}
    if 'risk_gender' not in st.session_state:
        st.session_state.risk_gender = None

    # STEP 0 : GENDER
    if st.session_state.risk_stage == 0 and st.session_state.risk_gender is None:
        gender = st.radio("Select Biological Sex", ["Male", "Female", "Other"], horizontal=True)
        if st.button("Confirm Gender"):
            st.session_state.risk_gender = gender

    # STEP 1 : DISEASE INPUT
    if st.session_state.risk_stage == 0 and st.session_state.risk_gender is not None:
        disease = st.text_input("Enter disease name for risk analysis")
        if st.button("Start Risk Analysis") and disease:
            st.session_state.risk_disease = disease
            st.session_state.risk_stage = 1
            st.session_state.risk_answers = {}

            q_prompt = f"""
You are a clinical AI doctor.
For disease: {disease}
Patient Biological Sex: {st.session_state.risk_gender}
Generate EXACTLY 10 yes/no diagnostic questions.
Return only the questions without numbering.
"""
            raw_questions = call_gemini(q_prompt)

            cleaned = []
            for line in (raw_questions or "").splitlines():
                line = line.strip().lstrip('-•0123456789.) ').strip()
                if line:
                    cleaned.append(line)
                if len(cleaned) == 10:
                    break

            if len(cleaned) < 10:
                cleaned += [f"Follow-up question {i+1}" for i in range(10 - len(cleaned))]

            st.session_state.risk_questions = cleaned

    # STEP 2 : QUESTIONS
    elif st.session_state.risk_stage == 1:
        total_q = len(st.session_state.risk_questions)
        st.progress(len(st.session_state.risk_answers) / total_q if total_q else 0)

        for idx, question in enumerate(st.session_state.risk_questions):
            st.markdown(f"**Q{idx+1}. {question}**")
            st.session_state.risk_answers[idx] = st.radio(
                label="",
                options=["Yes", "No"],
                horizontal=True,
                key=f"risk_q_{idx}"
            )

        if st.button("Calculate Health Risk"):
            analysis_prompt = f"""
Patient Sex: {st.session_state.risk_gender}
Disease: {st.session_state.risk_disease}
Answers: {st.session_state.risk_answers}
Calculate risk percentage (0-100) and short clinical conclusion.
"""
            st.session_state.risk_result = call_gemini(analysis_prompt)
            st.session_state.risk_stage = 2

    # STEP 3 : RESULT
    elif st.session_state.risk_stage == 2:
        st.subheader("📊 Health Risk Score")
        st.write(st.session_state.risk_result)

        if st.button("Restart Risk Analysis"):
            st.session_state.risk_stage = 0
            st.session_state.risk_disease = ""
            st.session_state.risk_questions = []
            st.session_state.risk_answers = {}
            st.session_state.risk_gender = None
            st.session_state.risk_result = None

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('---')
st.markdown('✅ Powered by Deep Learning & AI')
