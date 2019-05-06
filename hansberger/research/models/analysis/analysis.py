import ripser
import matplotlib.pyplot as plt
import json
from os.path import join
from django.conf import settings
from django.contrib.postgres.fields import JSONField
from django.db import models
from django.utils.text import slugify
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
        on_delete=models.PROTECT,
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

    '''
    source_dataset = models.ForeignKey(
        Dataset,
        on_delete=models.PROTECT,
        related_name='filtration_analysis_set',
        related_query_name='filtration_analysis'
    )
    '''

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
    result_plot = models.ImageField(blank=True, null=True)

    @models.permalink
    def get_absolute_url(self):
        return ('research:filtrationanalysis-detail', (), {'filtrationanalysis_slug': self.slug,
                'research_slug': self.research.slug})

    def execute(self, input_matrix):
        rips = ripser.Rips(maxdim=self.max_homology_dimension, thresh=self.max_distances_considered, coeff=self.coeff,
                           do_cocycles=self.do_cocycles, n_perm=self.n_perm)
        analysis_result_matrix = rips.fit_transform(input_matrix, distance_matrix=True)
        self.__save_matrix_plot(rips, analysis_result_matrix)
        self.__save_matrix_json([l.tolist() for l in analysis_result_matrix])
        self.save()

    def __save_plot(self, rips):
        relative_plot_path = join('research', self.research.slug, self.slug, self.slug+'_plot.svg')
        absolute_plot_path = join(settings.MEDIA_ROOT, relative_plot_path)
        rips.plot()
        plt.savefig(absolute_plot_path)
        self.plot = relative_plot_path

    def __save_matrix_json(self, matrix):
        self.result = json.dumps(matrix)
