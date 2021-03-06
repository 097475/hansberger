from django.db import models
from django.utils.text import slugify
from django.urls import reverse_lazy


class Research(models.Model):
    name = models.CharField(max_length=150)
    slug = models.SlugField(db_index=True, unique=True, max_length=150)
    description = models.TextField(max_length=500, blank=True, null=True)
    creation_date = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = 'research'
        ordering = ['-creation_date']
        verbose_name = 'research'
        verbose_name_plural = 'research'

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        if not self.id:
            self.slug = slugify(self.name)
        super(Research, self).save(*args, **kwargs)

    def get_absolute_url(self):
        return reverse_lazy(
            'research:research-detail',
            kwargs={'research_slug': self.slug}
        )
