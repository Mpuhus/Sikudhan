# Temporary change to force Git commit
import os
import torch
import io
from PIL import Image
import torchvision.transforms as transforms
from fastapi import FastAPI, UploadFile, File

# Initialize FastAPI app
app = FastAPI()

# Load YOLOv7 model
model_path = "best.pt"  # Ensure 'best.pt' is in the project directory
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} not found. Ensure it is uploaded.")

model = torch.hub.load('WongKinYiu/yolov7', 'custom', path=model_path, source='github')

# Image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

@app.get("/")
def home():
    return {"message": "YOLOv7 API is running!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Transform image
    img_tensor = transform(image).unsqueeze(0)

    # Perform inference
    results = model(img_tensor)
    
    # Process results
    predictions = results.pandas().xyxy[0].to_dict(orient="records")

    return {"detections": predictions}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Get PORT from environment variable
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
