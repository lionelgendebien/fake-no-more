import streamlit as st
from audiorecorder import audiorecorder
import time
import random

# Set page config
st.set_page_config(
    page_title="Fake No More - Voice Analysis",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- App Header ---
st.markdown(
    """
    <style>
    body {
        background-color: #07182A;
    }
    .title {
        font-size: 58px;
        font-weight: bold;
        text-align: center;
        color: #408BDE;
        margin-bottom: 20px;
        animation: fade-in 2s;
    }
    .subtitle {
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 20px;
        color: #AAD7FA;
    }
    .subtitle2 {
        text-align: left;
        font-size: 22px;
        font-weight: bold;
        margin-bottom: 20px;
        color: #00aaff;
    }
    </style>
    <div class="title">🎙️ Fake No More 🎙️</div>
    <div class="subtitle">Drawing the line between real and artificial voices</div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# --- Voice Recording Section ---
st.markdown("<div class='subtitle2'>Step 1 - Record your voice:</div>", unsafe_allow_html=True)

# Use audiorecorder to record voice
audio = audiorecorder("Click to record", "Click to stop recording")

if len(audio) > 0:
    # To play the recorded audio in the frontend:
    st.audio(audio.export().read())

    # Save the audio to a file
    audio.export("recorded_audio.wav", format="wav")

    # --- Analyze Recorded Audio ---
    st.markdown("---")
    st.markdown("<div class='subtitle2'>Step 2 - Predict the results for the recorded voice:</div>", unsafe_allow_html=True)

    if st.button("Analyze Recorded Voice"):
        with st.spinner("Analyzing..."):
            time.sleep(2)  # Simulate API response time
            prediction = random.choice(["This voice is real.", "This voice is AI."])  # Placeholder prediction
            st.markdown(
                f"""
                <div style='text-align: center; padding: 20px; font-size: 20px; 
                            font-weight: bold; color: {"#2ecc71" if prediction == "This voice is real." else "#e74c3c"}; 
                            border: 2px solid {"#2ecc71" if prediction == "This voice is real." else "#e74c3c"}; 
                            border-radius: 10px; margin-top: 20px;'>
                    {prediction}
                </div>
                """,
                unsafe_allow_html=True,
            )
