from ultralytics import YOLO
from model.DetectedObject import DetectedObject
import math

class DetectObjects:

    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.model.export(format="ncnn")

    def execute(self, frame):

        detectedObjects = []
    
        trackedObjects = self.model.track(
            source = frame,
            conf = 0.5,
            vid_stride = 2,
            imgsz = 320
        )
        for r in trackedObjects:
            boxes = r.boxes
            for box in boxes:
                confidence = math.ceil((box.conf[0]*100))/100
                objectClass = int(box.cls[0])
                detectedObject = DetectedObject(box.id, self.model.names[objectClass], confidence)
                #detectedObjects.append({"id": box.id.cpu().numpy().astype(int), "class": self.model.names[objectClass], "confidence": confidence, "location": box.xyxy})
                detectedObjects.append(detectedObject)

        return detectedObjects