from django.apps import AppConfig


class DatasetsConfig(AppConfig):
    name = 'datasets'

    def ready(self):
        import datasets.signals # noqa
