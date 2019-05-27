import os.path
from django.contrib.postgres.fields import JSONField
from django.db import models
from django.utils.text import slugify
from django.conf import settings
import pandas as pd
import matplotlib.pyplot as plt
import scipy.spatial.distance as distance
import numpy
from .research import Research


class TextDatasetManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset().filter(
            source_type=Dataset.TEXT
        )


def dataset_directory_path(instance, filename):
    return f'research/{instance.research.slug}/datasets/{instance.slug}/{filename}'


class Dataset(models.Model):
    TEXT = 'TXT'
    SOURCE_TYPE_CHOICES = (
        (TEXT, 'Text'),
    )
    name = models.CharField(max_length=150)
    slug = models.SlugField(db_index=True, max_length=150)
    description = models.TextField(max_length=500, blank=True, null=True)
    creation_date = models.DateField(auto_now_add=True)
    source = models.FileField(upload_to=dataset_directory_path)
    source_type = models.CharField(
        max_length=3,
        choices=SOURCE_TYPE_CHOICES
    )
    storage_path = models.CharField(max_length=500)
    research = models.ForeignKey(
        Research,
        on_delete=models.CASCADE,
        related_name='text_datasets',
        related_query_name='text_dataset',
    )
    data = JSONField(null=True, blank=True)
    plot = models.ImageField(max_length=500, null=True, blank=True)

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
        if not self.storage_path:
            self.storage_path = os.path.join(
                self.research.storage_path,
                'datasets',
                self.slug,
            )
        super().save(*args, **kwargs)

    @models.permalink
    def get_absolute_url(self):
        return ('research:dataset-detail', (), {'dataset_slug': self.slug, 'research_slug': self.research.slug})

    @property
    def absolute_storage_path(self):
        return os.path.join(settings.MEDIA_ROOT, self.storage_path)

    def get_distance_matrix(self, metric):
        return distance_matrix(self.data, metric)

    def get_correlation_matrix(self):
        return correlation_matrix(self.data)

    def split_matrix(self, window, overlap):  # returns a generator
        matrix = numpy.array(self.data).transpose()
        # matrix = self.data
        cols = len(matrix[0])
        step = window - overlap
        windows = 1 + (cols - window) // step

        for i in range(windows):
            tmp = matrix[:, window*i - overlap*i: window*(i+1) - overlap*i]
            yield tmp


class TextDataset(Dataset):
    objects = TextDatasetManager()

    class Meta:
        proxy = True

    def process_source_and_save_information(self, values_separator, identity_column_index, header_row_index):
        dataframe = self.get_dataframe(values_separator, identity_column_index, header_row_index)
        self.data = dataframe.values.tolist()
        self.__make_plot(dataframe)
        self.save()

    def get_dataframe(self, values_separator, identity_column_index, header_row_index):
        return pd.read_csv(
            self.source.path,
            index_col=identity_column_index,
            sep=values_separator,
            header=header_row_index,
        )

    def __make_plot(self, dataframe):
        dataframe.plot()
        plot_filename = self.slug + '_plot.svg'
        if not os.path.exists(self.absolute_storage_path):
            os.makedirs(self.absolute_storage_path)
        plt.savefig(os.path.join(self.absolute_storage_path, plot_filename))
        plt.clf()
        self.plot = os.path.join(self.storage_path, plot_filename)


def distance_matrix(matrix, metric):
    return distance.squareform(distance.pdist(
                numpy.array(matrix).transpose(),
                metric=metric
            ))


def correlation_matrix(matrix):
    return numpy.corrcoef(numpy.array(matrix))
