from django.db import models
from django.utils.text import slugify
from django.urls import reverse_lazy
import pandas as pd
import matplotlib.pyplot as plt
import mpld3
from research.models import Research


class Dataset(models.Model):
    name = models.CharField(max_length=150)
    slug = models.SlugField(db_index=True, max_length=150)
    description = models.TextField(max_length=500, blank=True, null=True)
    creation_date = models.DateField(auto_now_add=True)
    research = models.ForeignKey(
        Research,
        on_delete=models.CASCADE,
        related_name='datasets',
        related_query_name='dataset'
    )

    class Meta:
        unique_together = (('slug', 'research'))
        verbose_name = "dataset"
        verbose_name_plural = "datasets"

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        if not self.id:
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)

    def get_absolute_url(self):
        return reverse_lazy(
            'datasets:dataset-detail',
            kwargs={
                'research_slug': self.research.slug,
                'dataset_slug': self.slug,
            }
        )


class TextDataset(Dataset):
    source = models.FileField()
    values_separator_character = models.CharField(max_length=5)
    identity_column_index = models.IntegerField()
    header_row_index = models.IntegerField()

    @property
    def dataframe(self):
        return pd.read_csv(
            self.source.path,
            index_col=self.identity_column_index,
            sep=self.values_separator_character,
            header=self.header_row_index,
        )

    @property
    def plot(self):
        self.dataframe.plot()
        figure = plt.gcf()
        html_figure = mpld3.fig_to_html(figure, template_type='general')
        plt.clf()
        return html_figure
