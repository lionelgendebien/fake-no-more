import streamlit as st
import pandas as pd
import numpy as np

import time
import random
from audiorecorder import audiorecorder
import requests 

# --- Page Config ---
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
    /* Set background color for the entire app */
    body {
        background-color: #07182A; /* Light blue background */
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
    .next-button {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background-color: #408BDE;
        color: white;
        font-size: 18px;
        padding: 10px 20px;
        border-radius: 8px;
        cursor: pointer;
    }
    
    </style>
    <div class="title">🎙️ Fake No More 🎙️</div>
    <div class="subtitle">Drawing the line between real and artificial voices</div>
    """,
    unsafe_allow_html=True,
)

#st.write("### Upload Your Audio File"; margin-bottom: 20px)
st.markdown("---")

st.markdown(
    """
     <style>
    .subtitle2 {
        text-align: left;
        font-size: 22px;
        font-weight: bold;
        margin-bottom: 20px
        color: #00aaff;
    }
    
    </style>
    <div class="subtitle2">Step 1 - Upload a file: </div>
    """,
    unsafe_allow_html=True,
)

# --- File Upload ---
uploaded_file = st.file_uploader(' ', type=["wav"])


# --- TO CHECK !!! ---
if uploaded_file:
    st.markdown("---")
    st.markdown(
        """
        </style>
        <div class="subtitle2">Step 2 - Play and analyze the audio</div>
        """,
        unsafe_allow_html=True,
)
    st.write("\n\n\n")
    # --- Audio Playback ---
    st.audio(uploaded_file, format="audio/wav")
    
    # Send the file to the FastAPI endpoint for upload
    upload_url = "http://127.0.0.1:8080/upload-audio/"
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "audio/wav")}
    
    try:
        response = requests.post(upload_url, files=files)
        if response.status_code == 200:
            st.success(f"File {uploaded_file.name} uploaded successfully.")

    # --- Placeholder Prediction ---
            st.markdown("---")
            st.markdown(
            """
            </style>
            <div class="subtitle2">Step 3 - Predict the results</div>
            """,
            unsafe_allow_html=True,
        )
            st.write("\n\n\n")
            if st.button("Analyze Voice"):
                with st.spinner("Analyzing..."):
                    # Now send the file to the /predict-deepfake/ endpoint to get the prediction
                    predict_url = "http://127.0.0.1:8080/predict-deepfake/"
                    predict_response = requests.post(predict_url, files=files)
                    if predict_response.status_code == 200:
                        prediction = predict_response.json().get("prediction")
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
                    else:
                        st.error(f"Error in prediction: {predict_response.json().get('error', 'Unknown error')}")
        else:
            st.error(f"Error in file upload: {response.json().get('error', 'Unknown error')}")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to the API: {e}")

    # Button to go to the next page (second page)
    st.markdown(
    '<button class="next-button" onclick="window.location.href=\'/pages/page_01.py\'">Next</button>',
    unsafe_allow_html=True,
    )