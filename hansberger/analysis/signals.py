from django.dispatch import receiver
from django.db.models import signals
from .models import FiltrationWindow


@receiver(signals.post_delete, sender=FiltrationWindow)
def delete_window(sender, instance, **kwargs):
    instance.result_plot.delete(False)
