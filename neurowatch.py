import os
import configparser
from CameraController import CameraController
from service import NeurowatchService

CONFIG_FILE = 'config.ini'

def initialize_config():
    message = '''
        Welcome to Neurowatch Client Setup.

        An api token is required to upload video clips to the Neurowatch server and recieve notifications.

        The api token can be generated in the web settings page.
    '''
    print(message)
    token = input('Api token: ')
    return token

def test_config(token):
    print('Testing configuration:')
    service = NeurowatchService(token=token)
    result = service.perform_ping()
    if result:
        print('Success!')
        return True
    else:
        print('Failure! Please ensure the api token is correct')
        return False
    
def save_config(token):
    config = configparser.ConfigParser()
    config['DEFAULT'] = {'token': token}
    with open(CONFIG_FILE, 'w') as config_file:
        config.write(config_file)

def main():
    if not os.path.exists(CONFIG_FILE):
        token = initialize_config()
        while(test_config(token) == False):
            token = initialize_config()
        save_config(token)
        print('Starting client...')
        cameraController = CameraController()
        cameraController.caputre_video()
    else:
        cameraController = CameraController()
        cameraController.caputre_video()

if __name__ == '__main__':
    main()
