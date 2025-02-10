import subprocess
from fastapi import FastAPI, UploadFile, File
import shutil
import os

app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    
    # Save uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run YOLOv7 detection
    yolo_command = f"python D:/yolov7/yolov7/detect.py --weights D:/yolov7/yolov7/runs/train/best.pt --source {file_path} --device cpu"
    
    process = subprocess.run(yolo_command, shell=True, capture_output=True, text=True)
    
    if process.returncode == 0:
        return {"message": "Detection completed!", "output": "Check runs/detect folder for results"}
    else:
        return {"error": process.stderr}

@app.get("/")
def home():
    return {"message": "YOLOv7 API is running!"}
