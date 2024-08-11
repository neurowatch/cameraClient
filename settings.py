import configparser

CONFIG_FILE = 'config.ini'

configParser = configparser.ConfigParser()
configParser.read(CONFIG_FILE)

config = configParser['DEFAULT']

# TODO: Move this to config.ini
RESIZE_WIDTH = 320
RESIZE_HEIGHT = 240
BLUR_KERNEL = (5, 5)
SOURCE=0
HISTORY=10
UPDATE_INTERVAL=100

if 'api_url' not in config:
    raise ValueError('api_url not defined')
else:
    API_URL= config['api_url']

if 'token' not in config:
    API_KEY= '' 
else:
    API_KEY= config['token']