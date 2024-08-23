import torch
import torchvision
from model.DetectedObject import DetectedObject
import math
import torch.nn.utils.prune as prune


class DetectObjectsSSDLite:
    '''
        Runs torchVision SSD300 model to detect objects in the frame
    '''
    def __init__(self):
        self.model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
            pretrained=True, 
            task='detect',
        )
        self.model.eval().to("cpu")  # Set the model to evaluation mode
        self.coco_labels = [
            "__background__", "person"
        ]
        self.quantized_model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Conv2d}, dtype=torch.qint8)
        for module in self.quantized_model.modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=0.2)


    def execute(self, frame):

        # Ensure the input frame is a PyTorch tensor and normalized as required by the SSD model
        frame_tensor = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        detectedObjects = []
        
        # Perform object detection using the SSD model
        with torch.no_grad():
            predictions = self.quantized_model(frame_tensor)
        
         # SSD model output: list of dictionaries with 'boxes', 'labels', and 'scores'
        for prediction in predictions:
            boxes = prediction['boxes']
            labels = prediction['labels']
            scores = prediction['scores']

            for i in range(len(boxes)):
                if scores[i] >= 0.75:  # Apply confidence threshold
                    box = boxes[i]
                    label = labels[i].item()
                    confidence = math.ceil((scores[i].item() * 100)) / 100
                    detectedObject = DetectedObject(None, self.coco_labels[label], confidence)
                    detectedObjects.append(detectedObject)
                    print(f"Detected {self.coco_labels[label]} with confidence {confidence}")
                    print(f"Bounding box: {box.tolist()}")

        # Returns the detected objects in the frame
        return detectedObjects