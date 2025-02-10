from kivy.app import App
from kivy.uix.image import Image
import requests

class YOLOApp(App):
    def build(self):
        url = "https://your-api-url.com/predict/"
        image_path = "test.jpg"  # Change this to the image you want to upload

        with open(image_path, "rb") as file:
            response = requests.post(url, files={"file": file})
        
        with open("detected.jpg", "wb") as f:
            f.write(response.content)

        return Image(source="detected.jpg")

if __name__ == "__main__":
    YOLOApp().run()
