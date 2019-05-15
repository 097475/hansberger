from django.dispatch import receiver
from django.models import signals
from .models import TextDataset


@receiver(signals.post_delete, sender=TextDataset)
def submission_delete(sender, instance, **kwargs):
    instance.source.delete(False)
