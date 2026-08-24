import io
from pathlib import Path

import torch
import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
FRONTEND_DIR = BASE_DIR / "test_frontend"

# 1. ENABLE CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. THE OFFICIAL 38 CLASS MAPPING INDEX
# This index acts as your translation dictionary between the index number and the disease text name.
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy',
    'Corn___Common_rust', 'Corn___Gray_leaf_spot', 'Corn___Northern_Leaf_Blight', 'Corn___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites_Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# 3. LOAD YOUR MODELS AT STARTUP
disease_model = models.efficientnet_b0(weights=None)
disease_model.classifier[1] = torch.nn.Linear(disease_model.classifier[1].in_features, 38)
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

@app.get("/")
def home():
    return {"message": "Crop Project Backend is Running!"}

app.mount("/test", StaticFiles(directory=FRONTEND_DIR, html=True), name="test-frontend")

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