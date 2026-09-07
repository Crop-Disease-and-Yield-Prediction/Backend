import io
from pathlib import Path

import torch
import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models

from cure_service import get_cure_advice

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# 1. ENABLE CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. THE MODEL'S 42 CLASS MAPPING INDEX
# Keep this order aligned with the class order used during model training.
CLASS_NAMES = [
    'American Bollworm on Cotton', 'Anthracnose on Cotton', 'Army worm',
    'Becterial Blight in Rice', 'Brownspot', 'Common_Rust', 'Cotton Aphid',
    'Flag Smut', 'Gray_Leaf_Spot', 'Healthy Maize', 'Healthy Wheat',
    'Healthy cotton', 'Leaf Curl', 'Leaf smut', 'Mosaic sugarcane',
    'RedRot sugarcane', 'RedRust sugarcane', 'Rice Blast', 'Sugarcane Healthy',
    'Tungro', 'Wheat Brown leaf Rust', 'Wheat Stem fly', 'Wheat aphid',
    'Wheat black rust', 'Wheat leaf blight', 'Wheat mite', 'Wheat powdery mildew',
    'Wheat scab', 'Wheat___Yellow_Rust', 'Wilt', 'Yellow Rust Sugarcane',
    'bacterial_blight in Cotton', 'bollrot on Cotton', 'bollworm on Cotton',
    'cotton mealy bug', 'cotton whitefly', 'maize ear rot', 'maize fall armyworm',
    'maize stem borer', 'pink bollworm in cotton', 'red cotton bug',
    'thirps on  cotton'
]

# 3. LOAD YOUR MODELS AT STARTUP
disease_model = models.efficientnet_b0(weights=None)
disease_model.classifier[1] = torch.nn.Linear(disease_model.classifier[1].in_features, 42)
disease_model.load_state_dict(torch.load(
    MODELS_DIR / "efficientnet_realworld_robust.pth",
    map_location=torch.device("cpu"),
    weights_only=True,
))
disease_model.eval()

yield_model = joblib.load(MODELS_DIR / "crop_yield_fixed_ensemble.joblib")

# 4. FIX PREPROCESSING: Added ImageNet normalization parameters
# Without normalization, real-world images will result in highly inaccurate predictions!
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

class YieldInput(BaseModel):
    Crop: str
    Crop_Year: int = Field(ge=1997, le=2020)
    Season: str
    State: str
    Area: float = Field(gt=0)
    Production: float = Field(gt=0)
    Annual_Rainfall: float = Field(ge=0)
    Fertilizer: float = Field(ge=0)
    Pesticide: float = Field(ge=0)


class CureInput(BaseModel):
    disease: str = Field(min_length=2, max_length=200)

@app.get("/")
def home():
    return {"message": "Crop Project Backend is Running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Read the incoming image file
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # 2. Preprocess image for the disease model
    tensor_image = transform(image).unsqueeze(0) 
    
    # 3. Run Disease Prediction
    with torch.no_grad():
        disease_outputs = disease_model(tensor_image)
        probabilities = torch.nn.functional.softmax(disease_outputs, dim=1)
        confidence, predicted_class_idx = torch.max(probabilities, 1)
        
        disease_idx = int(predicted_class_idx.item()) 
        
        # Translate the numerical index into a clean, human-readable name string
        disease_name = CLASS_NAMES[disease_idx]
        display_name = disease_name.replace('___', ' - ').replace('_', ' ')
        confidence_pct = round(confidence.item() * 100, 2)
    
    # 4. Run Yield Prediction Placeholder
    yield_result = "Calculated Yield Output Here" 
    
    # 5. Return JSON straight back to frontend
    return {
        "status": "success",
        "disease_detected_index": disease_idx,
        "disease_detected_name": display_name,
        "confidence_level": f"{confidence_pct}%",
        "predicted_yield": yield_result
    }

@app.post("/predict-yield")
def predict_yield(payload: YieldInput):
    input_data = pd.DataFrame([payload.model_dump()])
    prediction = yield_model.predict(input_data)[0]
    return {
        "status": "success",
        "predicted_yield": round(float(prediction), 4),
        "yield_formula_reference": "Production / Area",
    }


@app.post("/disease-cure")
def disease_cure(payload: CureInput):
    try:
        advice = get_cure_advice(payload.disease.strip())
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return {
        "status": "success",
        "disease": payload.disease.strip(),
        "summary": advice.summary,
        "treatment_options": advice.treatment_options,
        "prevention_steps": advice.prevention_steps,
        "disclaimer": "This is general information, not a confirmed diagnosis. Follow product labels and local agricultural guidance.",
    }