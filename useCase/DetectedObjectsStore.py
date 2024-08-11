class DetectedObjectsStore:
    '''
        Stores detected objects while the clip creation is in process and pop them when it is ready.
    '''


    def __init__(self):
        # List to store detected objects
        self.detected_objects = []

        # Keeps the count of the latest frame where an object was detected. YOLO has no way of knowing if an object that is reentering the frame is the same
        # as the one that got out of it, for now they will be treated as separate objects.
        self.last_clip_with_objects = -1
    
    def store(self, detected_objects_in_frame, current_clip_frame):

        # Iterates obver the objects array
        for detected_object_in_frame in detected_objects_in_frame:

            # Finds the next stored object that has the same id as the object form the array
            detected_object = next((detected_object for  detected_object in self.detected_objects if detected_object.id == detected_object_in_frame.id), None)

            # Obtains the difference between the current clip and the last one that has objects
            clip_detection_diff = current_clip_frame - self.last_clip_with_objects
            self.last_clip_with_objects = current_clip_frame

            # If a detected object exists and the diff is less than 10 frames, it is the same object. The diff is an arbitrary value but it can solve the problem of
            # YOLO not knowing if an object reentering the frame is the same as the one that exited it, but it may lead to other issues such as mixing two objects that
            # enter and exit in close succession.
            if detected_object and clip_detection_diff < 10:
                confidence_in_frame = detected_object_in_frame.confidence
                # Sets the confindence vlaue to the largest.
                if confidence_in_frame > detected_object.confidence:
                    detected_object.confidence = confidence_in_frame
            else:
                # If no object exists in the stored objects array, store it.
                detected_object_in_frame.detected_in_frame = current_clip_frame
                self.detected_objects.append(detected_object_in_frame)

    def pop(self):
        # Returns the objects and clears the list.
        objects_to_return = self.detected_objects.copy()
        self.detected_objects.clear()
        return objects_to_return