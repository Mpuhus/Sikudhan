import os
import torch
import io
from PIL import Image
import torchvision.transforms as transforms
from fastapi import FastAPI, UploadFile, File

# Initialize FastAPI app
app = FastAPI()

# Load YOLOv7 model correctly
model_path = "best.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} not found. Ensure it is uploaded.")

# Load YOLOv7 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    from models.experimental import attempt_load
    from utils.general import non_max_suppression

    model = attempt_load(model_path, map_location=device)  # Correct model loading
    model.eval()

except Exception as e:
    raise RuntimeError(f"Error loading YOLOv7 model: {e}")

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
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        results = model(img_tensor)
    
    # Process results
    detections = non_max_suppression(results)[0].cpu().numpy()

    return {"detections": detections.tolist()}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Get PORT from environment variable
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
