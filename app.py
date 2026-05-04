import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="MediPredict AI", page_icon="🧠", layout="centered")
st.markdown("""
<style>
body {
    background: #0f172a;
    color: white;
}

.title {
    text-align: center;
    font-size: 40px;
    font-weight: 700;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 25px;
}

.card {
    background: #1e293b;
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
}
.stButton>button {
    background: #2563eb;
    color: white;
    border-radius: 10px;
    height: 45px;
    width: 100%;
    border: none;
}
.stButton>button:hover {
    background: #1d4ed8;
}
</style>
""", unsafe_allow_html=True)
@st.cache_resource
def load_model():
    return joblib.load("medi_predict_model.joblib")

try:
    saved_data = load_model()
    model = saved_data["model"]
    le = saved_data["le"]
    symptoms = saved_data["symptoms"]
except:
    st.error("Model file not found!")
    st.stop()
specialist_map = {
    "Heart attack": ["Cardiologist"],
      "Hypertension": ["Cardiologist"],
    "Asthma": ["Pulmonologist"],
      "Pneumonia": ["Pulmonologist"],
    "Diabetes": ["Endocrinologist"],
      "Migraine": ["Neurologist"],
    "Stroke": ["Neurologist"],
      "Gastritis": ["Gastroenterologist"],
    "Flu": ["General Physician"],
      "Cold": ["General Physician"]
}


st.markdown('<div class="title">🧠 MediPredict AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart Disease Prediction System</div>', unsafe_allow_html=True)

st.markdown("### 🧾 Patient Info")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 25)

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])

gender_value = 1 if gender == "Male" else 0

st.markdown("### 🔍 Select Symptoms")

symptom_columns = [col for col in symptoms if col not in ["age", "gender"]]

selected_symptoms = []
cols = st.columns(3)

for i, symptom in enumerate(symptom_columns):
    clean_name = symptom.replace("_", " ").title()
    if cols[i % 3].checkbox(clean_name):
        selected_symptoms.append(symptom)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚀 Analyze Now"):

    if not selected_symptoms:
        st.warning("⚠ Please select at least one symptom")

    else:
        input_data = [age, gender_value] + [
            1 if col in selected_symptoms else 0 for col in symptom_columns
        ]

        input_df = pd.DataFrame([input_data], columns=symptoms)

        probs = model.predict_proba(input_df)[0]
        top_idx = probs.argmax()
        confidence = probs[top_idx]

        main_disease = le.inverse_transform([top_idx])[0]
        specialists = specialist_map.get(main_disease, ["General Physician"])

        st.markdown(f"""
        <div class="card">
            <h2>🦠 {main_disease}</h2>
            <p><b>Confidence:</b> {round(confidence*100,2)}%</p>
            <p><b>👨‍⚕️ Specialist:</b> {", ".join(specialists)}</p>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Check Important.... This is not final diagnosis")