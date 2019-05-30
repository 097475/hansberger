from enum import Enum
from django.db import models
from django.utils.text import slugify
from django.urls import reverse_lazy
from research.models import Research


class DatasetKindChoice(Enum):
    TEXT = "Text"
    EDF = "EDF"


class Dataset(models.Model):
    name = models.CharField(max_length=150)
    slug = models.SlugField(db_index=True, max_length=150)
    description = models.TextField(max_length=500, blank=True, null=True)
    kind = models.CharField(
        max_length=10,
        choices=[(kind.name, kind.value) for kind in DatasetKindChoice]
    )
    source = models.FileField()
    creation_date = models.DateField(auto_now_add=True)
    research = models.ForeignKey(
        Research,
        on_delete=models.CASCADE,
        related_name='datasets',
        related_query_name='dataset'
    )

    class Meta:
        ordering = ['-creation_date']
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
