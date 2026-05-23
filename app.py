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
        job_titles = pickle.load(f)
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
st.write("Apni job ka naam type karein ya Mic button daba kar bolein:")

# 2. Voice Input Section
st.write("### 🎙️ Voice Search (Optional)")
# Yeh button mic active karega aur English mein aawaz ko text banayega
voice_text = speech_to_text(
    start_prompt="🔴 Click to Speak (Bolna shuru karein)",
    stop_prompt="⏹️ Stop Recording",
    language='en',  # Kyunki job titles English mein hain (e.g., "Software Engineer")
    key='voice_search'
)

# Agar voice se koi text mila hai toh use default value bana dein
default_index = 0
if voice_text:
    st.info(f"🎤 Aap ne bola: *\"{voice_text}\"*")
    # Check karein ke kya bola hua lafz kisi job title se match karta hai
    for idx, job in enumerate([""] + job_titles):
        if voice_text.lower() in job.lower() and job != "":
            default_index = idx
            break

# 3. Search Box (Isme voice input automatic select ho jayega agar match hua)
selected_job = st.selectbox(
    "Job Title Type, Select ya Voice se dhoondein:",
    options=[""] + job_titles,
    index=default_index,
    format_func=lambda x: "🔍 Yahan type karein ya upar mic use karein..." if x == "" else x
)

# 4. Prediction Section
if selected_job != "":
    st.markdown(f"### Selected Job: **{selected_job}**")
    
    if st.button("Predict AI Impact"):
        job_data = job_stats[selected_job]
        
        # Features array banana
        features = np.array([[
            job_data['Average_Salary'],
            job_data['Years_Experience'],
            job_data['AI_Exposure_Index'],
            job_data['Tech_Growth_Factor'],
            job_data['Automation_Probability_2030']
        ]])
        
        # Scale aur Predict karna
        scaled_features = scaler.transform(features)
        prediction_encoded = model.predict(scaled_features)[0]
        risk_status = le.inverse_transform([prediction_encoded])[0]
        
        # Result Output
        st.markdown("---")
        st.subheader(f"Analysis for: **{selected_job}**")
        
        if risk_status == 'Low':
            st.success(f"✅ **AI Risk Category: LOW**\n\nYeh job 2030 mein safe lag rahi hai. AI isme madadgaar sabit hoga.")
        elif risk_status == 'Medium':
            st.warning(f"🔄 **AI Risk Category: MEDIUM**\n\nIs job mein kafi tabdeeli ayegi. AI tools seekhna zaroori hai.")
        else:
            st.error(f"🚨 **AI Risk Category: HIGH**\n\nIs job par automation ka bohot zyada asar hone ka khadsha hai!")
            
        # Extra stats show karna
        st.write(f"📉 **Automation Probability:** {job_data['Automation_Probability_2030']*100:.1f}%")
        st.write(f"⚡ **AI Exposure Index:** {job_data['AI_Exposure_Index']:.2f}")
else:
    st.info("💡 *Upar diye gaye Mic button par click karke job ka naam bolein (jaise: 'Software Engineer' ya 'Data'), system khud usay list mein select kar lega.*")
