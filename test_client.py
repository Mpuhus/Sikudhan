import requests

url = "http://127.0.0.1:8000/predict/"
file_path = "D:/yolov7/image_data/images/train/IMG_20230401_122337.jpg"

with open(file_path, "rb") as file:
    response = requests.post(url, files={"file": file})

print(response.json())  # Print the API response
