from django.dispatch import receiver
from django.models import signals
from .models import Dataset


@receiver(signals.post_delete, sender=Dataset)
def submission_delete(sender, instance, **kwargs):
    instance.source.delete(False)
