from ultralytics import YOLO
from model.DetectedObject import DetectedObject
import math

class DetectObjects:
    '''
        Runs YOLOv8 to detect objects in the frame
    '''
    def __init__(self, use_ncnn=True):
        self.model = YOLO("yolov8n.pt")
        self.use_ncnn = use_ncnn
        if use_ncnn:
            self.model.export(format="ncnn")
            self.ncnn_model = YOLO("./yolov8n_ncnn_model")

    def execute(self, frame):

        detectedObjects = []
        
        results = None

        # It can either use the ncnn or the regular yolov8n model. NCNN is optimized for devices such as a raspberry pi and is the default option 
        if self.use_ncnn:
            results = self.ncnn_model(
                source = frame,
                conf = 0.75, # Sets a confidence limit
                vid_stride = 2, # The stride is two pixels, this is done to increase performance.
                classes = [0], # Class 0 is person, all other classes are ignored in the detection
            )
        else:
            results = self.model(
                source = frame,
                conf = 0.75,
                vid_stride = 2,
                imgsz = 320,
                classes = [0],
            )

        # Iterates over the tracked objects and it's boxes. Creates a detected object instance and add it to the list
        for r in results:
            boxes = r.boxes
            for box in boxes:
                confidence = math.ceil((box.conf[0]*100))/100
                objectClass = int(box.cls[0])
                detectedObject = DetectedObject(box.id, self.model.names[objectClass], confidence)
                detectedObjects.append(detectedObject)

        # Returns the detected objects in the frame
        return detectedObjects