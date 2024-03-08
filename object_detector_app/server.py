import base64
import io
import os

from flask import Flask, jsonify, request
from PIL import Image

from detector import ObjectDetector

app = Flask(__name__)


def process_image(image_base64, image_name):
    decoded_image = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(decoded_image))

    detector = ObjectDetector(os.path.join("models", "yolov8n.pt"))

    return detector.detect(image, image_name)


@app.route("/process_image", methods=["POST"])
def process_image_route():
    data = request.get_json()
    image_base64 = data["image"]
    detections = process_image(image_base64, data["image_name"])
    return jsonify(detections)


if __name__ == "__main__":
    app.run(debug=True)
