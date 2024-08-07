from ultralytics import YOLO
from model.DetectedObject import DetectedObject
import math

class DetectObjects:

    def __init__(self, use_ncnn=True):
        self.model = YOLO("yolov8n.pt")
        self.use_ncnn = use_ncnn
        if use_ncnn:
            self.model.export(format="ncnn")
            self.ncnn_model = YOLO("./yolov8n_ncnn_model")

    def execute(self, frame):

        detectedObjects = []
        
        trackedObjects = None
        if self.use_ncnn:
            trackedObjects = self.ncnn_model.track(
                source = frame,
                conf = 0.75,
                vid_stride = 2,
                classes = [0],
            )
        else:
            trackedObjects = self.model.track(
                source = frame,
                conf = 0.75,
                vid_stride = 2,
                imgsz = 320,
                classes = [0],
            )

        for r in trackedObjects:
            boxes = r.boxes
            for box in boxes:
                confidence = math.ceil((box.conf[0]*100))/100
                objectClass = int(box.cls[0])
                detectedObject = DetectedObject(box.id, self.model.names[objectClass], confidence)
                detectedObjects.append(detectedObject)

        return detectedObjects