from django.contrib.postgres.fields import JSONField
from django.db import models
from ..analysis.analysis import Analysis


class Window(models.Model):
    analysis = models.ForeignKey(
        Analysis,
        on_delete=models.CASCADE,
        related_name='windows',
        related_query_name='window',
    )
    start_point = models.IntegerField()
    end_point = models.IntegerField()
    plot = models.ImageField(max_length=300)
    data = JSONField()

    def __str__(self):
        return f"Window {self.start_point} - {self.end_point}: {self.analysis.name}"
