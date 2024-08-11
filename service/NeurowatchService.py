import requests
import settings

class NeurowatchService:

    def __init__(self, api_url = settings.API_URL, token = settings.API_KEY):
        self.base_url = api_url
        self.headers = {
            'Authorization': f'Token {token}'
        }

    def upload_video(self, files, data):
        video_url = "videos/"
        url = self.base_url + video_url
        print(self.headers)
        response = requests.post(url, headers=self.headers, files=files, data=data)
        return response
    
    def perform_ping(self):
        ping_url = "ping/"
        url = self.base_url + ping_url
        response = requests.post(url, headers=self.headers)
        return response