import cv2
import numpy as np

class YOLODetector:
    def __init__(self, config_path="yolofiles/yolov3.cfg", weights_path="yolofiles/yolov3.weights", class_path="yolofiles/coco.names"):
        self.net = cv2.dnn.readNet(weights_path, config_path)
        with open(class_path, "r") as file:
            self.classes = file.read().strip().split("\n")

    def detect_people(self, frame):
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_names = self.net.getUnconnectedOutLayersNames()
        detections = self.net.forward(layer_names)

        detected_objects = []

        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5 and self.classes[class_id] == "person":
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)

                    x1 = int(center_x - w / 2)
                    y1 = int(center_y - h / 2)
                    x2 = int(center_x + w / 2)
                    y2 = int(center_y + h / 2)

                    # Append the valid bounding box to the detected_objects list
                    detected_objects.append([x1, y1, x2, y2, confidence, class_id])

        return detected_objects
