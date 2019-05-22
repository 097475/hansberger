from django.contrib.postgres.fields import JSONField
from django.db import models
from .analysis import FiltrationAnalysis, MapperAnalysis


class Window(models.Model):
    data = JSONField()
    plot = models.ImageField(max_length=700)

    class Meta:
        abstract = True


class FiltrationWindow(Window):
    analysis = models.ForeignKey(
        FiltrationAnalysis,
        on_delete=models.CASCADE,
        related_name='windows',
        related_query_name='window'
    )


class MapperWindow(Window):
    analysis = models.ForeignKey(
        MapperAnalysis,
        on_delete=models.CASCADE,
        related_name='windows',
        related_query_name='window'
    )
