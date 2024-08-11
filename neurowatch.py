import os
import configparser
from CameraController import CameraController
from service import NeurowatchService

CONFIG_FILE = 'config.ini'

def initialize_config(CONFIG_FILE):
    message = '''
        Welcome to Neurowatch Client Setup.

        An api token is required to upload video clips to the Neurowatch server and recieve notifications.

        The api token can be generated in the web settings page.
    '''
    print(message)

    configParser = configparser.ConfigParser()
    configParser.read(CONFIG_FILE)
    config = configParser['DEFAULT']

    token = input('Api token: ')

    if 'api_url' not in config:
        api_url = input(f'Base url:')
    else:
        default_api_url = config['api_url']
        api_url = input(f'Base url (Default: {default_api_url}):') or default_api_url        

    return {
        'token': token, 
        'api_url': api_url
    }

def test_config(config):
    try:
        print(config)
        print('Testing configuration:')
        service = NeurowatchService(api_url=config['api_url'], token=config['token'])
        result = service.perform_ping()
        if result:
            print('Success!')
            return True
        else:
            print('Failure! Please ensure the api token is correct')
            return False
    except:
        return False
    
def save_config(config):
    configParser = configparser.ConfigParser()
    configParser['DEFAULT'] = config
    with open(CONFIG_FILE, 'w') as config_file:
        configParser.write(config_file)

def setup_config(CONFIG_FILE):
    config = initialize_config(CONFIG_FILE)
    while(test_config(config) == False):
        config = initialize_config(CONFIG_FILE)
    save_config(config)
    print('Starting client...')

def get_current_config(CONFIG_FILE):
    configParser = configparser.ConfigParser()
    configParser.read(CONFIG_FILE)
    config = configParser['DEFAULT']
    return config

def main():
    if not os.path.exists(CONFIG_FILE):
        setup_config(CONFIG_FILE)
        cameraController = CameraController()
        cameraController.caputre_video()
    else:
        config = get_current_config(CONFIG_FILE)
        print(config)
        if not test_config(config):
            setup_config(CONFIG_FILE)

        cameraController = CameraController()
        cameraController.caputre_video()

if __name__ == '__main__':
    main()
