from django.contrib.postgres.fields import JSONField
from django.db import models
from django.utils.text import slugify
from enum import Enum
import math
from .research import Research


class Analysis(models.Model):
    name = models.CharField(max_length=100)
    slug = models.SlugField(
        db_index=True,
        max_length=110,
        blank=True,
        null=True,
    )
    description = models.TextField(max_length=500, blank=True, null=True)
    creation_date = models.DateField(auto_now_add=True)
    research = models.ForeignKey(
        Research,
        on_delete=models.CASCADE,
    )

    class Meta:
        abstract = True
        unique_together = (("slug", "research"))

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        if not self.id:
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)


class FiltrationAnalysis(Analysis):
    class FiltrationChoice(Enum):
        VIETORIS_RIPS = 'Vietoris Rips Filtration'
        CLIQUE_WEIGHTED_RANK = 'Clique Weighted Rank Filtration'

    type = models.CharField(
        max_length=50,
        choices=[(type.name, type.value) for type in FiltrationChoice]
    )
    max_homology_dimension = models.IntegerField(default=1)
    max_distances_considered = models.FloatField(default=math.inf)
    coeff = models.IntegerField(default=2)
    do_cocycles = models.BooleanField(default=False)
    n_perm = models.IntegerField(default=None)

    result = JSONField()

    class Meta(Analysis.Meta):
        verbose_name = f"{type} analysis"
        verbose_name_plural = f"{type}s analysis"

    def execute(self, matrix, start_point=None, end_point=None):
        # TODO: Implementare
        raise NotImplementedError()


class MapperAnalysis(Analysis):
    # TODO: Inserire parametri

    class Meta(Analysis.Meta):
        verbose_name = "mapper algorithm analysis"
        verbose_name_plural = "mapper algoritms analysis"

    def execute(self, distances_matrix):
        # TODO: Implementare
        raise NotImplementedError()
