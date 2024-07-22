import requests
import settings

class NeurowatchService:

    def __init__(self):
        self.base_url = settings.API_URL

    def uploadVideo(self, files, data):
        video_url = "videos/"
        url = self.base_url + video_url
        response = requests.post(url, files=files, data=data)

        return response