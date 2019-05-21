import os.path
from django.conf import settings
from django.db import models
from django.utils.text import slugify


class Research(models.Model):
    name = models.CharField(max_length=150)
    slug = models.SlugField(db_index=True, unique=True, max_length=150)
    description = models.TextField(max_length=500, blank=True, null=True)
    creation_date = models.DateField(auto_now_add=True)
    storage_path = models.CharField(max_length=300)

    class Meta:
        ordering = ['-creation_date']
        verbose_name = 'research'
        verbose_name_plural = 'research'

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        if not self.id:
            self.slug = slugify(self.name)
        if not self.storage_path:
            self.storage_path = os.path.join(
                'research',
                self.slug,
            )
        super(Research, self).save(*args, **kwargs)

    @models.permalink
    def get_absolute_url(self):
        return ('research:research-detail', (), {'research_slug': self.slug})

    @property
    def absolute_storage_path(self):
        return os.path.join(settings.MEDIA_ROOT, self.storage_path)
