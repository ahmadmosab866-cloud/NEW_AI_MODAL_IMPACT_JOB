import streamlit as st
import pickle
import numpy as np
from streamlit_mic_recorder import speech_to_text

# 1. Models aur Data Load karein
@st.cache_resource
def load_resources():
    with open('job_risk_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('job_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    with open('job_titles.pkl', 'rb') as f:
        job_titles = sorted(pickle.load(f))  # List ko alphabetically sort kar diya
    with open('job_stats.pkl', 'rb') as f:
        job_stats = pickle.load(f)
    return model, scaler, le, job_titles, job_stats

try:
    model, scaler, le, job_titles, job_stats = load_resources()
except FileNotFoundError:
    st.error("❌ Model files nahi milin! Pehle 'train_model.py' run karein.")
    st.stop()

# App UI
st.title("🤖 AI Job Impact Predictor 2030")
st.write("Apni job ka naam type karein ya niche diye gaye Mic button se bolein:")

# 2. Voice Input Section (Voice Command)
st.write("### 🎙️ Voice Command")
voice_text = speech_to_text(
    start_prompt="🔴 Click to Speak (Job ka naam bolein)",
    stop_prompt="⏹️ Stop Recording (Bolna band karein)",
    language='en',  # Kyunki excel file mein job titles English mein hain
    key='voice_search'
)

# Agar user ne kuch bola hai toh usay match karne ka logic
default_index = 0
if voice_text:
    clean_voice_text = voice_text.strip().lower()
    st.info(f"🎤 Aap ne bola: *\"{voice_text}\"*")
    
    # Check karna ke bola hua lafz kis job title ke andar aata hai
    for idx, job in enumerate([""] + job_titles):
        if clean_voice_text in job.lower() and job != "":
            default_index = idx
            break

# 3. Search Box with Live Auto-complete List
# Agar voice se koi match mila hoga toh 'index=default_index' ki wajah se woh automatic select ho jayega
selected_job = st.selectbox(
    "Job Title Select Ya Type Karein:",
    options=[""] + job_titles,
    index=default_index,
    format_func=lambda x: "🔍 Type to search job..." if x == "" else x
)

# 4. Prediction Section
if selected_job != "":
    st.info(f"Selected Job: **{selected_job}**")
    
    if st.button("Predict AI Impact"):
        try:
            # Job ke features background se nikalna
            job_data = job_stats[selected_job]
            
            # Features array taiyar karna
            features = np.array([[
                job_data['Average_Salary'],
                job_data['Years_Experience'],
                job_data['AI_Exposure_Index'],
                job_data['Tech_Growth_Factor'],
                job_data['Automation_Probability_2030']
            ]])
            
            # Scale input
            scaled_features = scaler.transform(features)
            
            # Predict Risk
            prediction_encoded = model.predict(scaled_features)[0]
            risk_status = le.inverse_transform([prediction_encoded])[0]
            
            # Result Display
            st.markdown("---")
            st.subheader(f"Analysis for: **{selected_job}**")
            
            # Visual alerts based on Risk
            if risk_status == 'Low':
                st.success(f"✅ **AI Risk Category: LOW**\n\nYeh job 2030 mein safe lag rahi hai. AI isme madadgaar sabit hoga, khatra nahi.")
            elif risk_status == 'Medium':
                st.warning(f"🔄 **AI Risk Category: MEDIUM**\n\nIs job mein kafi tabdeeli ayegi. AI tools seekhna aapke liye zaroori hoga.")
            else: # High
                st.error(f"🚨 **AI Risk Category: HIGH**\n\nIs job par automation ka bohot zyada asar hone ka khadsha hai. Skills upgrade karne ki zaroorat hai!")
                
            # Extra Details
            st.write(f"📉 **Automation Probability:** {job_data['Automation_Probability_2030']*100:.1f}%")
            st.write(f"⚡ **AI Exposure Index:** {job_data['AI_Exposure_Index']:.2f}")
            
        except KeyError:
            st.error("❌ Is job ka data record mein nahi mila. Please list se koi doosri job select karein.")
else:
    st.write("💡 *Upar mic button dabayein ya box mein apni job ka naam likhna shuru karein...*")
