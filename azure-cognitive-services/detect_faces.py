import requests
import json
from PIL import Image, ImageDraw
from io import BytesIO
from dotenv import load_dotenv
import os
import sys

load_dotenv()

subscription_key = os.getenv("AZURE_COGNITIVE_SERVICES_SUBSCRIPTION_KEY")
endpoint = os.getenv("AZURE_COGNITIVE_SERVICES_ENDPOINT")

detect_url = endpoint + "/face/v1.0/detect"

headers = {
    "Content-Type": "application/json",
    "Ocp-Apim-Subscription-Key": subscription_key
}

image_url = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/Face/images/identification1.jpg"

body = {
    "url": image_url
}

params = {
    "detectionModel": "detection_03",
    "recognitionModel": "recognition_04",
    "returnFaceLandmarks": "true",
    "returnFaceAttributes": "blur,exposure,glasses,headpose,mask,occlusion,qualityforrecognition"
}

response = requests.post(detect_url, headers=headers, json=body, params=params)
faces_json = response.json()
print(json.dumps(faces_json, indent=2))


response_image = requests.get(image_url)
image = Image.open(BytesIO(response_image.content))
draw = ImageDraw.Draw(image)

for face in faces_json:
    rect = face['faceRectangle']
    left = rect['left']
    top = rect['top']
    right = left + rect['width']
    bottom = top + rect['height']
    draw.rectangle([(left, top), (right, bottom)], outline='red', width=2)

    landmarks = face['faceLandmarks']
    for landmark, point in landmarks.items():
        x, y = point['x'], point['y']
        draw.ellipse([(x - 2, y - 2), (x + 2, y + 2)], outline='blue', width=2)

# image.save('output.png')
image.save('output_landmarks.png')


