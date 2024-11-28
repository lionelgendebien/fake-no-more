from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import librosa  # Example for audio processing
import numpy as np
from io import BytesIO
import joblib  # For loading your machine learning model
from fake_no_more.data_extraction_demo import extract_features

# Load your pre-trained machine learning model (make sure it's saved as a .pkl file)
# model = joblib.load("your_model.pkl")

app = FastAPI()

# Define the root `/` endpoint to check if the server is running
@app.get("/")
def read_root():
    return {"message": "Welcome to the Audio File Upload API!"}

# Endpoint to handle file uploads and process them
@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    try:
        # Read the uploaded file content as a binary stream
        audio_data = await file.read()

        # Convert binary audio data to a numpy array
        audio_file = BytesIO(audio_data)

        # Use librosa to load the audio (you can use other methods based on your needs)
        # Ensure to read the audio correctly with librosa (e.g., mono or stereo)
        audio, sr = librosa.load(audio_file, sr=None)  # sr=None preserves original sampling rate

        # You can add your preprocessing steps here (e.g., feature extraction)
        # Example: Convert audio to features (e.g., MFCC)
        features = librosa.feature.mfcc(y=audio, sr=sr)
        features = np.mean(features, axis=1)  # Mean aggregation of features

        # Pass the features to the model (uncomment the next line if you have a model)
        # prediction = model.predict(features.reshape(1, -1))

        # For this example, let's pretend we're predicting a label (classification)
        # Assume the model output is a class label
        prediction = "spoof"  # Replace this with your model prediction logic

        return JSONResponse(content={"prediction": prediction})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
