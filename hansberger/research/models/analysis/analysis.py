import ripser
import matplotlib.pyplot as plt
import json
import numpy
import math
import scipy.spatial.distance as dist
import os.path
from django.conf import settings
from django.contrib.postgres.fields import JSONField
from django.db import models
from django.utils.text import slugify
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.db.models import signals
from ..research import Research
from ..dataset.dataset import Dataset


class Analysis(models.Model):
    name = models.CharField(max_length=100)
    slug = models.SlugField(db_index=True, max_length=110)
    description = models.TextField(max_length=500, blank=True, null=True)
    creation_date = models.DateField(auto_now_add=True)
    research = models.ForeignKey(
        Research,
        on_delete=models.CASCADE,
        related_name='analysis_set',
        related_query_name='analysis',
    )
    dataset = models.ForeignKey(
        Dataset,
        on_delete=models.CASCADE,
        related_name='analysis_set',
        related_query_name='analysis',
    )

    class Meta:
        abstract = True
        unique_together = (('slug', 'research'))

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        if not self.id:
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)


class FiltrationAnalysis(Analysis):
    VIETORIS_RIPS_FILTRATION = 'VRF'
    CLIQUE_WEIGHTED_RANK_FILTRATION = 'CWRF'

    FILTRATION_TYPE_CHOICES = (
        (VIETORIS_RIPS_FILTRATION, 'Vietoris Rips Filtration'),
        (CLIQUE_WEIGHTED_RANK_FILTRATION, 'Clique Weighted Rank Filtration'),
    )

    METRIC_CHOICES = (
        ('braycurtis', 'Braycurtis'),
        ('canberra', 'Canberra'),
        ('chebyshev', 'Chebyshev'),
        ('cityblock', 'City block'),
        ('correlation', 'Correlation'),
        ('cosine', 'Cosine'),
        ('dice', 'Dice'),
        ('euclidean', 'Euclidean'),
        ('hamming', 'Hamming'),
        ('jaccard', 'Jaccard'),
        ('jensenshannon', 'Jensen Shannon'),
        ('kulsinski', 'Kulsinski'),
        ('mahalanobis', 'Mahalonobis'),
        ('matching', 'Matching'),
        ('minkowski', 'Minkowski'),
        ('rogerstanimoto', 'Rogers-Tanimoto'),
        ('russellrao', 'Russel Rao'),
        ('seuclidean', 'Seuclidean'),
        ('sokalmichener', 'Sojal-Michener'),
        ('sokalsneath', 'Sokal-Sneath'),
        ('sqeuclidean', 'Sqeuclidean'),
        ('yule', 'Yule'),
    )

    filtration_type = models.CharField(
        max_length=50,
        choices=FILTRATION_TYPE_CHOICES,
    )
    distance_matrix_metric = models.CharField(
        max_length=20,
        choices=METRIC_CHOICES
    )
    max_homology_dimension = models.IntegerField(default=1)
    max_distances_considered = models.FloatField(default=None, null=True, blank=True)  # None/Null means infinity
    coeff = models.IntegerField(default=2)
    do_cocycles = models.BooleanField(default=False)
    n_perm = models.IntegerField(default=None, null=True, blank=True)

    result_matrix = JSONField(blank=True, null=True)
    result_plot = models.ImageField(max_length=300, blank=True, null=True)

    @models.permalink
    def get_absolute_url(self):
        return ('research:filtrationanalysis-detail', (), {'filtrationanalysis_slug': self.slug,
                'research_slug': self.research.slug})

    def execute(self, input_matrix):
        _thresh = math.inf if self.max_distances_considered is None else self.max_distances_considered
        rips = ripser.Rips(maxdim=self.max_homology_dimension, thresh=_thresh, coeff=self.coeff,
                           do_cocycles=self.do_cocycles, n_perm=self.n_perm)
        analysis_result_matrix = rips.fit_transform(input_matrix, distance_matrix=True)
        self.__save_plot(rips)
        self.__save_matrix_json([l.tolist() for l in analysis_result_matrix])

    def __save_plot(self, rips):
        plot_filename = self.slug + '_plot.svg'
        relative_plot_dir = os.path.join('research', self.research.slug, 'analysis', self.slug)
        absolute_plot_dir = os.path.join(settings.MEDIA_ROOT, relative_plot_dir)
        if not os.path.exists(absolute_plot_dir):
            os.makedirs(absolute_plot_dir)
        rips.plot()
        plt.savefig(os.path.join(absolute_plot_dir, plot_filename))
        self.result_plot = os.path.join(relative_plot_dir, plot_filename)

    def __save_matrix_json(self, matrix):
        self.result_matrix = json.dumps(matrix)


@receiver(post_save, sender=FiltrationAnalysis)
def run_ripser(sender, instance, **kwargs):
    input_matrix = numpy.array(instance.dataset.matrix)
    if instance.filtration_type == FiltrationAnalysis.VIETORIS_RIPS_FILTRATION:
        ripser_input_matrix = dist.squareform(dist.pdist(input_matrix.transpose(),
                                              metric=instance.distance_matrix_metric))
    elif instance.filtration_type == FiltrationAnalysis.CLIQUE_WEIGHTED_RANK_FILTRATION:
        ripser_input_matrix = numpy.corrcoef(input_matrix)
    instance.execute(ripser_input_matrix)
    signals.post_save.disconnect(run_ripser, sender=FiltrationAnalysis)
    instance.save()
    signals.post_save.connect(run_ripser, sender=FiltrationAnalysis)
