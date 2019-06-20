from channels.generic.websocket import WebsocketConsumer
import json


class StatusHolder():
    status = 0
    @staticmethod
    def init():
        StatusHolder.reset_status()

    @staticmethod
    def set_status(value):
        StatusHolder.status = value

    @staticmethod
    def get_status():
        return StatusHolder.status

    @staticmethod
    def reset_status():
        StatusHolder.status = 0


class AnalysisConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()
        self.send(text_data=json.dumps({
            'status': "test"
        }))

    def disconnect(self, close_code):
        pass

    def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message = text_data_json['signal']
        if message == 'status':
            self.send(text_data=json.dumps({
                'status': StatusHolder.get_status()
            }))
