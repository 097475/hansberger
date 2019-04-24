from django.contrib.postgres.fields import JSONField
from django.db import models
from django.utils.text import slugify
from django.dispatch import receiver
from django.db.models import signals
from ..models import Research


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
    file = models.FileField(upload_to='research/datasets/')
    file_type = models.CharField(
        max_length=3,
        choices=FILE_TYPE_CHOICES
    )
    PLOTS_DIR = 'research/datasets/plots/'
    plot = models.ImageField(upload_to=PLOTS_DIR, blank=True, null=True)
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
