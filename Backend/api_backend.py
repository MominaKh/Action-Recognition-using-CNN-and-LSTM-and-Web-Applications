"""
FastAPI Backend for Action Recognition API
Handles image uploads and returns predictions with annotations
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import json

# -----------------------------
# Initialize FastAPI app
# -----------------------------
app = FastAPI(
    title="Action Recognition API",
    description="CNN-LSTM based action recognition with image annotation",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Global variables
# -----------------------------
model = None
metadata = None
IMG_SIZE = (128, 128)  # Must match your trained model
SEQ_LEN = 10            # For sequence-based models (if needed)

# -----------------------------
# Pydantic models
# -----------------------------
class TopPrediction(BaseModel):
    action: str
    confidence: float

class PredictionResponse(BaseModel):
    action: str
    confidence: float
    annotation: str
    top_5_predictions: List[TopPrediction]

# -----------------------------
# Helper functions
# -----------------------------
def load_model_and_metadata():
    """Load trained model and metadata at startup"""
    global model, metadata
    try:
        model = keras.models.load_model('final_action_model.h5')
        print("✓ Model loaded successfully")
        with open('dataset_metadata.json', 'r') as f:
            metadata = json.load(f)
        print("✓ Metadata loaded successfully")
        print(f"✓ Ready to predict {metadata['num_classes']} actions")
    except Exception as e:
        print(f"Error loading model or metadata: {e}")
        raise

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess uploaded image for model inference"""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize(IMG_SIZE)
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def generate_annotation(action_name: str) -> str:
    """Generate natural language annotation"""
    if action_name in metadata.get('action_descriptions', {}):
        return metadata['action_descriptions'][action_name]
    else:
        formatted_action = action_name.replace('_', ' ').title()
        return f"A person is {formatted_action.lower()}"

# -----------------------------
# Startup event
# -----------------------------
@app.on_event("startup")
async def startup_event():
    load_model_and_metadata()

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/")
async def root():
    return {"message": "Action Recognition API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    if model is None or metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True, "num_classes": metadata['num_classes']}

@app.get("/actions")
async def get_actions():
    if metadata is None:
        raise HTTPException(status_code=503, detail="Metadata not loaded")
    return {"actions": metadata['classes'], "total": len(metadata['classes'])}

@app.post("/predict", response_model=PredictionResponse)
async def predict_action(file: UploadFile = File(...)):
    if model is None or metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        processed_image = preprocess_image(image_bytes)

        predictions = model.predict(processed_image, verbose=0)[0]

        predicted_idx = int(np.argmax(predictions))
        confidence = float(predictions[predicted_idx])
        action_name = metadata['classes'][predicted_idx]
        annotation = generate_annotation(action_name)

        top_5_indices = np.argsort(predictions)[-5:][::-1]
        top_5_predictions = [
            TopPrediction(action=metadata['classes'][idx], confidence=float(predictions[idx]))
            for idx in top_5_indices
        ]

        return PredictionResponse(
            action=action_name,
            confidence=confidence,
            annotation=annotation,
            top_5_predictions=top_5_predictions
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch_predict")
async def batch_predict(files: List[UploadFile] = File(...)):
    if model is None or metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images allowed")

    results = []
    for file in files:
        try:
            image_bytes = await file.read()
            processed_image = preprocess_image(image_bytes)
            predictions = model.predict(processed_image, verbose=0)[0]
            predicted_idx = int(np.argmax(predictions))
            confidence = float(predictions[predicted_idx])
            action_name = metadata['classes'][predicted_idx]
            annotation = generate_annotation(action_name)

            results.append({
                "filename": file.filename,
                "action": action_name,
                "confidence": confidence,
                "annotation": annotation
            })

        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})

    return {"results": results}

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
