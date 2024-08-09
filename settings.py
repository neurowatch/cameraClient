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

API_URL="http://127.0.0.1:8000/api/"

if 'token' not in config:
    API_KEY= '' 
else:
    API_KEY= config['token']