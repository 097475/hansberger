from django.dispatch import receiver
from django.db.models import signals
from .models import Dataset, FiltrationAnalysis


@receiver(signals.post_delete, sender=Dataset)
def submission_delete(sender, instance, **kwargs):
    instance.source.delete(False)


@receiver(signals.post_delete, sender=FiltrationAnalysis)
def analysis_plot_delete(sender, instance, **kwargs):
    instance.result_plot.delete(False)
