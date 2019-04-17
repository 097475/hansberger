from django.contrib.postgres.fields import JSONField
from django.db import models
from django.utils.text import slugify
from django.db.models.signals import post_delete, post_save
from django.dispatch import receiver
from enum import Enum
from .research import Research


from .utils import edfmodule, csvmodule
from os.path import join
from django.conf import settings
from django.db.models import signals
import json


class FileType(Enum):
    TEXT = "Text file"
    EDF = "EDF file"

    @classmethod
    def all(self):
        return [FileType.TEXT, FileType.EDF]


class Dataset(models.Model):
    name = models.CharField(max_length=150)
    slug = models.SlugField(
        db_index=True,
        max_length=150,
        blank=True,
        null=True,
    )
    description = models.TextField(max_length=500, blank=True, null=True)
    creation_date = models.DateField(auto_now_add=True)
    research = models.ForeignKey(
        Research,
        on_delete=models.CASCADE,
        related_name='datasets',
        related_query_name='dataset',
    )
    file = models.FileField(upload_to="research/datasets/")
    file_type = models.CharField(
        max_length=10,
        choices=[(type.value, type.name) for type in FileType.all()]
    )
    values_delimiter_character = models.CharField(
        max_length=5,
        null=True,
        default=',',
        verbose_name='delimiter character of the values in the file'
    )
    row_id_column_index = models.IntegerField(
        null=True,
        default=0,
        verbose_name='column number that identifies the progressive number of rows in the file'
    )
    header_row_index = models.IntegerField(
        null=True,
        default=0,
        verbose_name='row number that identifies the column in the file'
    )
    data = JSONField(blank=True, null=True)
    data_image = models.ImageField(upload_to="research/datasets/images", blank=True, null=True)

    class Meta:
        ordering = ['-creation_date']
        unique_together = (("slug", "research"))
        verbose_name = 'dataset'
        verbose_name_plural = 'datasets'

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        if not self.id:
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)

    @models.permalink
    def get_absolute_url(self):
        return ("research:dataset-detail", (), {"dataset_slug": self.slug, "research_slug": self.research.slug})


@receiver(post_delete, sender=Dataset)
def submission_delete(sender, instance, **kwargs):
    instance.file.delete(False)


@receiver(post_save, sender=Dataset)
def my_handler(sender, instance, **kwargs):
    path = join(settings.MEDIA_ROOT, 'research', 'datasets', 'images', instance.slug+'_image.svg')
    if instance.file_type == FileType.TEXT.value:
        df = csvmodule.getDataFrameFromText(instance.file.path, index=instance.row_id_column_index,
                                            delim=instance.values_delimiter_character,
                                            header_index=instance.header_row_index)
        csvmodule.plotDF(df, path)
        instance.data = json.dumps(csvmodule.getMatrixFromDataFrame(df))
    elif instance.file_type == FileType.EDF.value:
        df = edfmodule.readEDF(instance.file.path)
        edfmodule.plotEDF(df, path)
        instance.data = json.dumps(edfmodule.edfToMatrix(df))
    instance.data_image = join('research', 'datasets', 'images', instance.slug+'_image.svg')
    signals.post_save.disconnect(my_handler, sender=Dataset)
    instance.save()
    signals.post_save.connect(my_handler, sender=Dataset)
