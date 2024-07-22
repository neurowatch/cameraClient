class DetectedObjectsStore:

    def __init__(self):
        self.detectedObjects = []
    
    def store(self, detectedObjectsInFrame):
        for detectedObjectInFrame in detectedObjectsInFrame:
            detectedObject = next((detectedObject for  detectedObject in self.detectedObjects if detectedObject.id == detectedObjectInFrame.id), None)
            if detectedObject:
                confidenceInFrame = detectedObjectInFrame.confidence
                if confidenceInFrame > detectedObject.confidence:
                    detectedObject.confidence = confidenceInFrame
            else:
                self.detectedObjects.append(detectedObjectInFrame)

    def pop(self):
        objectsToReturn = self.detectedObjects.copy()
        self.detectedObjects.clear()
        return objectsToReturn