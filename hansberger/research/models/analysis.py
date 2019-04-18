from django.contrib.postgres.fields import JSONField
from django.db import models
from django.utils.text import slugify
from enum import Enum
import math
from .research import Research

import sklearn.preprocessing
import sklearn.cluster
import sklearn.mixture


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
    n_perm = models.IntegerField(default=None, null=True)

    result = JSONField()

    class Meta(Analysis.Meta):
        verbose_name = f"{type} analysis"
        verbose_name_plural = f"{type}s analysis"

    def execute(self, matrix, start_point=None, end_point=None):
        # TODO: Implementare
        raise NotImplementedError()


class MapperAnalysis(Analysis):
    scalers = {
        'None': None,
        'MinMaxScaler': sklearn.preprocessing.MinMaxScaler(),
        'MaxAbsScaler': sklearn.preprocessing.MaxAbsScaler(),
        'RobustScaler': sklearn.preprocessing.RobustScaler(),
        'StandardScaler': sklearn.preprocessing.StandardScaler()
    }

    clusterers = {
        'K-Means': sklearn.cluster.KMeans(),
        'Affinity propagation': sklearn.cluster.AffinityPropagation(),
        'Mean-shift': sklearn.cluster.MeanShift(),
        'Spectral clustering': sklearn.cluster.SpectralClustering(),
        'Agglomerative clustering': sklearn.cluster.AgglomerativeClustering(),
        'DBSCAN': sklearn.cluster.DBSCAN(min_samples=3),
        'Gaussian mixtures': sklearn.mixture.GaussianMixture(),
        'Birch': sklearn.cluster.Birch()
    }
    # TODO: Inserire parametri

    class ProjectionChoice(Enum):
        SUM = 'sum'
        MEAN = 'mean'
        MEDIAN = 'median'
        MAX = 'max'
        MIN = 'min'
        STD = 'std'
        DIST_MEAN = 'dist_mean'
        L2NORM = 'l2norm'
        KNN_DISTANCE = 'knn_distance_n'  # TODO knn_distance, add scikit classes

    class ScalerChoice(Enum):
        NONE = 'None'
        MINMAXSCALER = 'MinMaxScaler'
        MAXABSSCALER = 'MaxAbsScaler'
        ROBUSTSCALER = 'RobustScaler'
        STANDARDSCALER = 'StandardScaler'

    # fit_transform parameters; not implemented : scaler params, scikit projections
    projection = models.CharField(
                max_length=50,
                choices=[(type.name, type.value) for type in ProjectionChoice]
                )
    scaler = models.CharField(
                max_length=50,
                choices=[(type.name, type.value) for type in ScalerChoice]
    )
    # map parameters; not implemented : clusterer params, cover limits

    class ClustererChoice(Enum):
        KMEANS = 'K-Means'
        AFFINITYPROPAGATION = 'Affinity propagation'
        MEANSHIFT = 'Mean-shift'
        SPECTRALCLUSTERING = 'Spectral clustering'
        WARDHC = 'Ward hierarchical clustering'
        AGGLOMERATIVE = 'Agglomerative clustering'
        DBSCAN = 'DBSCAN'
        GAUSSIANMIXTURES = 'Gaussian mixtures'
        BIRCH = 'Birch'
    use_original_data = models.BooleanField(default=False)
    clusterer = models.CharField(
                max_length=50,
                choices=[(type.name, type.value) for type in ClustererChoice],
                default=ClustererChoice.DBSCAN
                )
    cover_n_cubes = models.IntegerField(default=10)
    cover_perc_overlap = models.FloatField(default=0.5)
    graph_nerve_min_intersection = models.IntegerField(default=1)
    remove_duplicate_nodes = models.BooleanField(default=False)

    class Meta(Analysis.Meta):
        verbose_name = "mapper algorithm analysis"
        verbose_name_plural = "mapper algoritms analysis"

    def execute(self, distances_matrix):
        # TODO: Implementare
        raise NotImplementedError()
