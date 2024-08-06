class DetectedObjectsStore:

    def __init__(self):
        self.detected_objects = []
    
    def store(self, detected_objects_in_frame, current_clip_frame):
        for detected_object_in_frame in detected_objects_in_frame:
            detected_object = next((detected_object for  detected_object in self.detected_objects if detected_object.id == detected_object_in_frame.id), None)
            if detected_object:
                confidence_in_frame = detected_object_in_frame.confidence
                if confidence_in_frame > detected_object.confidence:
                    detected_object.confidence = confidence_in_frame
                if detected_object.detected_in_frame == None:
                    detected_object.detected_in_frame = current_clip_frame
            else:
                self.detected_objects.append(detected_object_in_frame)

    def pop(self):
        objects_to_return = self.detected_objects.copy()
        self.detected_objects.clear()
        return objects_to_return