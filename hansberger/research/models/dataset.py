import os.path
from django.contrib.postgres.fields import JSONField
from django.db import models
from django.utils.text import slugify
from django.dispatch import receiver
from django.db.models import signals
from django.conf import settings
import matplotlib.pyplot as plt
import pandas as pd
from .research import Research


class EDFDatasetManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset().filter(
            file_type=Dataset.EDF
        )


class TextDatasetManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset().filter(
            file_type=Dataset.TEXT
        )


def dataset_directory_path(instance, filename):
    return f'research/{instance.research.slug}/datasets/{instance.slug}/{filename}'


class Dataset(models.Model):
    EDF = 'EDF'
    TEXT = 'TXT'
    FILE_TYPE_CHOICES = (
        (EDF, 'EDF file'),
        (TEXT, 'Text file'),
    )
    name = models.CharField(max_length=150)
    slug = models.SlugField(db_index=True, max_length=150)
    description = models.TextField(max_length=500, blank=True, null=True)
    creation_date = models.DateField(auto_now_add=True)
    research = models.ForeignKey(
        Research,
        on_delete=models.CASCADE,
        related_name='datasets',
        related_query_name='dataset',
    )
    file = models.FileField(upload_to=dataset_directory_path)
    file_type = models.CharField(
        max_length=3,
        choices=FILE_TYPE_CHOICES
    )
    plot = models.ImageField(max_length=300, blank=True, null=True)
    matrix = JSONField(blank=True, null=True)

    class Meta:
        ordering = ['-creation_date']
        unique_together = (('slug', 'research'))
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
        return ('research:dataset-detail', (), {'dataset_slug': self.slug, 'research_slug': self.research.slug})


@receiver(signals.post_delete, sender=Dataset)
def submission_delete(sender, instance, **kwargs):
    instance.file.delete(False)
    instance.plot.delete(False)


class TextDataset(Dataset):
    objects = TextDatasetManager()

    class Meta:
        proxy = True

    def process_file(self, values_separator, header_row_index, identity_column_index):
        dataframe = self.__get_dataframe_from_text(identity_column_index, values_separator, header_row_index)
        self.__save_dataframe_plot(dataframe)
        self.__save_dataframe_matrix(dataframe)
        self.save()

    def __get_dataframe_from_text(self, identity_column_index, values_separator, header_row_index):
        return pd.read_csv(
            self.file.path,
            index_col=identity_column_index,
            sep=values_separator,
            header=header_row_index,
        )

    def __save_dataframe_plot(self, dataframe):
        dataframe.plot()
        plot_filename = self.slug + '_plot.svg'
        relative_plot_dir = os.path.join('research', self.research.slug, 'datasets', self.slug)
        absolute_plot_dir = os.path.join(settings.MEDIA_ROOT, relative_plot_dir)
        if not os.path.exists(absolute_plot_dir):
            os.makedirs(absolute_plot_dir)
        plt.savefig(os.path.join(absolute_plot_dir, plot_filename))
        self.plot = os.path.join(relative_plot_dir, plot_filename)

    def __save_dataframe_matrix(self, dataframe):
        self.matrix = dataframe.values.tolist()


class EDFDataset(Dataset):
    objects = EDFDatasetManager()

    class Meta:
        proxy = True

    def process_file(self):
        raise(NotImplementedError)
