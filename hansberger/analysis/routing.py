from django.conf.urls import url
from . import consumers

websocket_urlpatterns = [
    # (http->django views is added by default)
    url(r'^ws/analysis/', consumers.AnalysisConsumer),
]
