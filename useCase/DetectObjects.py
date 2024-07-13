from ultralytics import YOLO
from model.DetectedObject import DetectedObject
import math

class DetectObjects:

    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.model.export(format="ncnn")
        self.ncnn_model = YOLO("yolov8n_ncnn_model")


    def execute(self, frame):

        detectedObjects = []
        results = self.model(
            source=frame,
            imgsz=320
        )
        for r in results:
            boxes = r.boxes
            for box in boxes:
                confidence = math.ceil((box.conf[0]*100))/100
                objectClass = int(box.cls[0])
                detectedObjects.append({"object": self.model.names[objectClass], "confidence": confidence})
            '''
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                confidence = math.ceil((box.conf[0]*100))/100
                print("confidence: ", confidence)

                objectClass = int(box.cls[0])
                print("class: ", confidence)

                origin = [x1, y1]
                font = cv2.FONT_HERSHEY_PLAIN
                fontScale = 2
                color = (0, 255, 255)
                thickness = 2

                cv2.putText(frame, self.model.names[objectClass], origin, font, fontScale, color, thickness)
            '''
            
        return detectedObjects