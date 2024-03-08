import os

from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, model_path=None, confidence=0.5, image_size=640):
        if model_path is None:
            model_path = os.path.join("models", "yolov8n.pt")
        else:
            assert os.path.exists(model_path)

        self.model = YOLO(model_path, verbose=False)
        self.confidence = confidence
        self.image_size = image_size
        self.fixed_classes = [0, 2]  # Person and car

    def detect(self, input_image, image_name="no-name", save_image=False):
        res = self.model.predict(
            input_image,
            conf=self.confidence,
            imgsz=self.image_size,
            classes=self.fixed_classes,
            verbose=False,
        )[0]
        if save_image:
            res.save(f"result_{os.path.splitext(image_name)[0]}.jpg")

        # Filter result to result only person and car classes.
        detections = []
        for i, id in enumerate(list(res.boxes.cls.numpy())):
            boxes = [int(value) for value in res.boxes.xyxy.numpy()[i]]
            conf = float(res.boxes.conf.numpy()[i])
            detections.append(
                {
                    "class": "person" if id == 0 else "car",
                    "confidence": conf,
                    "bbox": boxes,
                }
            )
        return {
            "image_name": image_name,
            "num_objects": len(detections),
            "objects": detections,
        }
