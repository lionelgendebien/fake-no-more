from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import librosa
from io import BytesIO
import os

app = FastAPI()

UPLOAD_DIRECTORY = "uploaded_audio_files"

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

# Step 1: Endpoint to upload an audio file
@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    try:
        # Read the uploaded file content as a binary stream
        audio_data = await file.read()

        # Store the uploaded file temporarily
        file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
        with open(file_path, "wb") as f:
            f.write(audio_data)

        # Return a success message with the filename
        return JSONResponse(content={"message": "File uploaded successfully", "filename": file.filename})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# Step 2: Predict the length (duration) of the uploaded audio file
@app.post("/predict-duration/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded audio file
        audio_data = await file.read()

        # Convert binary audio data to a BytesIO object for librosa to read
        audio_file = BytesIO(audio_data)

        # Load the audio file using librosa
        audio, sr = librosa.load(audio_file, sr=None)  # sr=None to preserve original sample rate

        # Calculate the duration (in seconds) of the audio file
        duration = librosa.get_duration(y=audio, sr=sr)

        # Return the duration in the response
        return JSONResponse(content={"message": "Duration calculated successfully", "duration_seconds": duration})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# Define the root `/` endpoint to check if the server is running
@app.get("/")
def read_root():
    return {"message": "Welcome to the Audio File Upload API!"}
