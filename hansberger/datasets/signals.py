from django.dispatch import receiver
from django.db.models import signals
from .models import TextDataset


@receiver(signals.post_delete, sender=TextDataset)
def submission_delete(sender, instance, **kwargs):
    instance.source.delete(False)
