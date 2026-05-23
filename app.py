import streamlit as st
import pickle
import numpy as np
import difflib
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
    st.error("❌ Model files nahi mili! Pehle 'train_model.py' run karein.")
    st.stop()

# App UI
st.title("🤖 AI Job Impact Predictor 2030")
st.write("Apni job ka naam likhein ya Mic use karein aur check karein ke 2030 tak AI ka us par kya asar hoga.")

# 2. Voice Input Section (Voice Command)
st.write("### 🎙️ Voice Command (Optional)")
voice_text = speech_to_text(
    start_prompt="🔴 Click to Speak (Job ka naam bolein)",
    stop_prompt="⏹️ Stop Recording (Bolna band karein)",
    language='en',  # Dataset English mein hai isliye language 'en' rakhi hai
    key='voice_search'
)

# Agar voice se kuch bola gaya hai toh text box ki default value wohi ban jaye
input_value = ""
if voice_text:
    input_value = voice_text.strip()
    st.info(f"🎤 Aap ne bola: *\"{input_value}\"*")

# 3. User Input (Text Box)
# Agar voice text majood hai toh text_input mein automatic fill ho jayega
user_job = st.text_input(
    "Enter Job Title (e.g., Data Scientist, Construction Worker):", 
    value=input_value
).strip()

if user_job:
    # 4. Error Handling & Similarity Check (Fuzzy Matching)
    closest_matches = difflib.get_close_matches(user_job, job_titles, n=3, cutoff=0.4)
    
    selected_job = None
    
    # Check agar exact match mil gaya
    if user_job.lower() in [job.lower() for job in job_titles]:
        selected_job = [job for job in job_titles if job.lower() == user_job.lower()][0]
    elif closest_matches:
        st.warning(f"⚠️ Aapki likhi/boli hui job '{user_job}' database mein nahi mili.")
        # User ko options dikhane ke liye dropdown
        selected_job = st.selectbox("Kya aapka matlab in mein se koi job tha?", closest_matches)
    else:
        st.error("❌ Afsos! Is naam se milti julti koi job database mein nahi mili. Baraye meherbani koi doosra naam try karein.")
    
    # 5. Prediction Section
    if selected_job:
        st.info(f"Selected: **{selected_job}**")
        if st.button("Predict AI Impact"):
            try:
                # Job ke features dataset se nikalna
                job_data = job_stats[selected_job]
                
                # Features ko array format mein convert karna
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
                
                # Visual layout based on Risk
                if risk_status == 'Low':
                    st.success(f"✅ **AI Risk Category: LOW**\n\nYeh job 2030 mein safe lag rahi hai. AI isme madadgaar sabit hoga, khatra nahi.")
                elif risk_status == 'Medium':
                    st.warning(f"🔄 **AI Risk Category: MEDIUM**\n\nIs job mein tabdeeli ayegi. AI tools seekhna aapke liye zaroori hoga.")
                else: # High
                    st.error(f"🚨 **AI Risk Category: HIGH**\n\nIs job par automation ka bohot zyada asar hone ka khadsha hai. Skills upgrade karne ki zaroorat hai!")
                    
                # Additional Details
                st.write(f"📉 **Automation Probability:** {job_data['Automation_Probability_2030']*100:.1f}%")
                st.write(f"⚡ **AI Exposure Index:** {job_data['AI_Exposure_Index']:.2f}")
                
            except KeyError:
                st.error("❌ Is job ka statistical data record mein nahi mila.")
