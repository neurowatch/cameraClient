class SaveClip:
    
    def __init__(self, neurowatchService):
        self.service = neurowatchService

    def execute(self, file_path, detected_objects):
        with open(file_path, 'rb') as video_file:
            files = {'video': video_file}
            data = {'detected_objects': []}
            for detected_object in detected_objects:
                data['detected_objects'].append(
                    {
                        "object_name": detected_object.name, 
                        "detection_confidence": detected_object.confidence
                    }
                )
            self.service.uploadVideo(files=files, data=data)