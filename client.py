from __future__ import print_function

import base64
import io
import os
import json

import requests
from PIL import Image


def convert_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        image = Image.open(img_file)
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")

    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def send_image_to_server(image_path, server_url):
    base64_image = convert_to_base64(image_path)
    payload = {"image": base64_image, "image_name": os.path.basename(image_path)}
    response = requests.post(server_url, json=payload)
    return response.json()


if __name__ == "__main__":
    image_path = os.path.join("images", "img3.jpg")
    server_url = "http://127.0.0.1:5000/process_image"
    detections = send_image_to_server(image_path, server_url)
    print(detections)

    # To save json: json_object
    #with open("results.json", "w") as outfile:
    #    json.dump(detections, outfile)
