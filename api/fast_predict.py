from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import pickle
from fake_no_more.prediction_demo import prepare_test_data, load_model, predict_X
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Define paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd()))
MODEL_PATH = os.path.join(PROJECT_ROOT, "fake_no_more", "model_specifications", "lstm.pkl")
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "api", "uploaded_audio_files")

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Load the model once when the app starts
try:
    app.state.model = load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model at {MODEL_PATH}: {e}")


@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Return a success message with the filename
        return JSONResponse(content={"message": "File uploaded successfully", "filename": file.filename})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

#last_prediction = {"message": "No prediction made yet.", "prediction": None}

@app.post("/predict-deepfake")
async def predict(file: UploadFile = File(...)):
    #global last_prediction
    print(datetime.now())
    try:
        # Save the uploaded file temporarily
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        print("open file", datetime.now())
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Preprocess the audio file for prediction using the existing function
        try:
            print("prep start", datetime.now())
            X_test_scaled = prepare_test_data(file_path)
            print("prep end", datetime.now())
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"Failed to process audio: {e}"})
        
        # Predict using the loaded model
        try:
            print("pred start", datetime.now())
            prediction = predict_X(app.state.model, X_test_scaled)
            print("pred end", datetime.now())
            last_prediction = {"message": "Prediction successful", "prediction": prediction}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Prediction failed: {e}"})

        # Return the prediction result
        return JSONResponse(content=last_prediction)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/")
def read_root():
    return {"message": "Welcome to the Audio File Upload API!"}

#@app.get("/predict-deepfake")
#def get_last_prediction():
#    return JSONResponse(content=last_prediction)
