from os.path import basename
from django.db import models
from django.dispatch import receiver
from django.db.models import signals
from ..models import Dataset


class EdfSource(models.Model):
    file = models.FileField(upload_to='research/datasets/sources/edf')
    dataset = models.OneToOneField(
        Dataset,
        on_delete=models.CASCADE,
        related_name='edfsource_of'
    )

    class Meta:
        verbose_name = "EDF source"
        verbose_name_plural = "EDF sources"

    def __str__(self):
        return basename(self.file.name)


class TextSource(models.Model):
    file = models.FileField(upload_to='research/datasets/sources/text')
    dataset = models.OneToOneField(
        Dataset,
        on_delete=models.CASCADE,
        related_name='textsource_of'
    )
    values_separator_character = models.CharField(
        max_length=5,
        verbose_name="delimiter character of the values in the file"
    )
    identity_column_index = models.IntegerField(
        verbose_name="column number that identifies the progressive number of rows in the file"
    )
    header_row_index = models.IntegerField(
        verbose_name="row number that identifies the column in the file"
    )

    def __str__(self):
        super().__str__()
        return basename(self.file.name)


@receiver(signals.post_delete, sender=EdfSource)
@receiver(signals.post_delete, sender=TextSource)
def submission_delete(sender, instance, **kwargs):
    instance.file.delete(False)
