import requests

class NeurowatchService:

    def __init__(self):
        self.base_url = "http://127.0.0.1:8000/"

    def uploadVideo(self, video_path):
        video_url = "videos/"

        url = self.base_url + video_url

        with open(video_path, 'rb') as video_file:
            files = {'video': video_file}
            response = requests.post(url, files=files)

        return response