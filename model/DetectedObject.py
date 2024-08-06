class DetectedObject:
    def __init__(self, id, name, confidence):
        self.id = id
        self.name = name
        self.confidence = confidence
        self.detected_in_frame = None

    def __repr__(self):  
        return f'DetectedObject: {self.id} - name: {self.name}, confidence: {self.confidence}, frameNumber: {self.detected_in_frame}'