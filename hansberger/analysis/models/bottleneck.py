from django.db import models


class BottleneckManager(models.Manager):
    def create_bottleneck(self):
        pass


class DiagramManager(models.Manager):
    def create_diagram(self):
        pass


class Bottleneck(models.Model):
    BOTTLENECK_TYPES = [('consecutive', 'consecutive'), ('one_to_all', 'one_to_all'), ('all_to_all', 'all_to_all')]
    analysis = models.ForeignKey(
        'analysis.FiltrationAnalysis',
        on_delete=models.CASCADE,
        null=True
    )
    window = models.ForeignKey(
        'analysis.Window',
        on_delete=models.CASCADE,
        null=True
    )
    homology = models.PositiveIntegerField()
    kind = models.CharField(choices=BOTTLENECK_TYPES)


class Diagram(models.Model):
    bottleneck = models.ForeignKey(
        Bottleneck,
        on_delete=models.CASCADE
    )
    window = models.ForeignKey(
        'analysis.Window',
        on_delete=models.CASCADE
    )
    image = models.TextField()
    bottleneck_distance = models.FloatField()
