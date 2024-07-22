class DetectedObject:
    def __init__(self, id, name, confidence):
        self.id = id
        self.name = name
        self.confidence = confidence

    def __repr__(self):  
        return f'DetectedObject: {self.id} - name: {self.name}, confidence: {self.confidence}'