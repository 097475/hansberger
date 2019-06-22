from channels.generic.websocket import WebsocketConsumer
import json


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class StatusHolder(metaclass=Singleton):

    def __init__(self):
        self.reset()

    def set_status(self, value):
        self.status = value

    def set_limit(self, value):
        self.limit = value

    def get_status(self):
        return self.status

    def get_limit(self):
        return self.limit

    def reset(self):
        self.status = 0
        self.limit = 0
        self.kill = False

    def set_kill(self):
        self.kill = True

    def get_kill(self):
        return self.kill


class AnalysisConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()
        self.status_logger = StatusHolder()

    def disconnect(self, close_code):
        pass

    def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message = text_data_json['signal']
        if message == 'status':
            self.send(text_data=json.dumps({
                'status': self.status_logger.get_status(),
                'limit': self.status_logger.get_limit()
            }))
        elif message == 'kill':
            self.status_logger.set_kill()
