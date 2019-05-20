import ripser
import kmapper
import matplotlib.pyplot as plt
import json
import numpy
import math
import sklearn.cluster
import sklearn.preprocessing
import sklearn.mixture
import os.path
from django.conf import settings
from django.contrib.postgres.fields import JSONField
from django.db import models
from django.utils.text import slugify
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.db.models import signals
from .research import Research
from .dataset import Dataset


class Analysis(models.Model):
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
    name = models.CharField(max_length=100)
    slug = models.SlugField(db_index=True, max_length=110)
    description = models.TextField(max_length=500, blank=True, null=True)
    creation_date = models.DateField(auto_now_add=True)
    research = models.ForeignKey(
        Research,
        on_delete=models.CASCADE,
        related_name='%(class)s_requests_created',
        related_query_name='analysis',
    )
    dataset = models.ForeignKey(
        Dataset,
        on_delete=models.CASCADE,
        related_name='%(class)s_requests_created',
        related_query_name='analysis',
        blank=True,
        null=True
    )
    distance_matrix_metric = models.CharField(
        max_length=20,
        choices=METRIC_CHOICES,
        default='euclidean'
    )
    precomputed_distance_matrix = models.FileField(upload_to='research/precomputed/', default=None, null=True,
                                                   blank=True)  # TODO
    window_size = models.IntegerField(default=None, null=True, blank=True)  # default no window
    window_overlap = models.IntegerField(default=0)

    def get_type(self):
        return self._meta.verbose_name

    class Meta:
        abstract = True
        unique_together = (('slug', 'research'))

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        if not self.id:
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)


class MapperAnalysis(Analysis):
    scalers = {
        'None': None,
        'MinMaxScaler': sklearn.preprocessing.MinMaxScaler(),
        'MaxAbsScaler': sklearn.preprocessing.MaxAbsScaler(),
        'RobustScaler': sklearn.preprocessing.RobustScaler(),
        'StandardScaler': sklearn.preprocessing.StandardScaler()
    }

    clusterers = {
        'k-means': sklearn.cluster.KMeans(),
        'affinity_propagation': sklearn.cluster.AffinityPropagation(),
        'mean-shift': sklearn.cluster.MeanShift(),
        'spectral_clustering': sklearn.cluster.SpectralClustering(),
        'agglomerative_clustering': sklearn.cluster.AgglomerativeClustering(),
        'DBSCAN': sklearn.cluster.DBSCAN(min_samples=1),  # should be 3
        'gaussian_mixtures': sklearn.mixture.GaussianMixture(),
        'birch': sklearn.cluster.Birch()
    }
    # TODO: Inserire parametri

    PROJECTION_CHOICES = (
        ('sum', 'Sum'),
        ('mean', 'Mean'),
        ('median', 'Median'),
        ('max', 'Max'),
        ('min', 'Min'),
        ('std', 'Std'),
        ('dist_mean', 'Dist_mean'),
        ('l2norm', 'L2norm'),
        ('knn_distance_n', 'knn_distance_n')  # TODO knn_distance, add scikit classes
    )

    SCALER_CHOICES = (
        ('None', 'None'),
        ('MinMaxScaler', 'MinMaxScaler'),
        ('MaxAbsScaler', 'MaxAbsScaler'),
        ('RobustScaler', 'RobustScaler'),
        ('StandardScaler', 'StandardScaler'),  # missing parameters
    )

    CLUSTERER_CHOICES = (
        ('k-means', 'K-Means'),
        ('affinity_propagation', 'Affinity propagation'),
        ('mean-shift', 'Mean-shift'),
        ('spectral_clustering', 'Spectral clustering'),
        ('agglomerative_clustering', 'StandardScaler'),
        ('DBSCAN', 'DBSCAN'),
        ('gaussian_mixtures', 'Gaussian mixtures'),
        ('birch', 'Birch')  # missing parameters
    )

    # fit_transform parameters; not implemented : scaler params, scikit projections
    projection = models.CharField(
                max_length=50,
                choices=PROJECTION_CHOICES
                )
    scaler = models.CharField(
                max_length=50,
                choices=SCALER_CHOICES
    )

    # map parameters; not implemented : clusterer params, cover limits
    use_original_data = models.BooleanField(default=False)
    clusterer = models.CharField(
                max_length=50,
                choices=CLUSTERER_CHOICES,
                default='DBSCAN'
                )
    # missing cover limits
    cover_n_cubes = models.IntegerField(default=10)
    cover_perc_overlap = models.FloatField(default=0.5)
    graph_nerve_min_intersection = models.IntegerField(default=1)
    remove_duplicate_nodes = models.BooleanField(default=False)

    graph = models.TextField(blank=True, null=True)

    class Meta(Analysis.Meta):
        verbose_name = "mapper algorithm analysis"
        verbose_name_plural = "mapper algoritm analyses"

    # TODO: check precomputed=False
    def execute(self, distance_matrix, original_matrix):
        mapper = kmapper.KeplerMapper()
        mycover = kmapper.Cover(n_cubes=self.cover_n_cubes, perc_overlap=self.cover_perc_overlap)
        mynerve = kmapper.GraphNerve(min_intersection=self.graph_nerve_min_intersection)
        original_data = original_matrix if self.use_original_data else None
        projected_data = mapper.fit_transform(distance_matrix, projection=self.projection,
                                              scaler=MapperAnalysis.scalers[self.scaler], distance_matrix=False)
        graph = mapper.map(projected_data, X=original_data, clusterer=MapperAnalysis.clusterers[self.clusterer],
                           cover=mycover, nerve=mynerve, precomputed=False,
                           remove_duplicate_nodes=self.remove_duplicate_nodes)
        output_graph = mapper.visualize(graph, save_file=False)
        self.graph = output_graph


class FiltrationAnalysis(Analysis):
    VIETORIS_RIPS_FILTRATION = 'VRF'
    CLIQUE_WEIGHTED_RANK_FILTRATION = 'CWRF'

    FILTRATION_TYPE_CHOICES = (
        (VIETORIS_RIPS_FILTRATION, 'Vietoris Rips Filtration'),
        (CLIQUE_WEIGHTED_RANK_FILTRATION, 'Clique Weighted Rank Filtration'),
    )

    filtration_type = models.CharField(
        max_length=50,
        choices=FILTRATION_TYPE_CHOICES,
    )
    max_homology_dimension = models.IntegerField(default=1)
    max_distances_considered = models.FloatField(default=None, null=True, blank=True)  # None/Null means infinity
    coeff = models.IntegerField(default=2)
    do_cocycles = models.BooleanField(default=False)
    n_perm = models.IntegerField(default=None, null=True, blank=True)

    result_matrix = JSONField(blank=True, null=True)
    result_plot = models.ImageField(max_length=300, blank=True, null=True)
    result_entropy = JSONField(blank=True, null=True)

    class Meta(Analysis.Meta):
        verbose_name = "filtration analysis"
        verbose_name_plural = "filtration analyses"

    @models.permalink
    def get_absolute_url(self):
        return ('research:filtrationanalysis-detail', (), {'filtrationanalysis_slug': self.slug,
                'research_slug': self.research.slug})

    def execute(self, input_matrix):
        _thresh = math.inf if self.max_distances_considered is None else self.max_distances_considered
        result = ripser.ripser(input_matrix, maxdim=self.max_homology_dimension, thresh=_thresh, coeff=self.coeff,
                               distance_matrix=True, do_cocycles=self.do_cocycles, n_perm=self.n_perm)
        self.__save_plot(result['dgms'])
        self.__save_entropy_json(result['dgms'])
        self.__save_matrix_json(result)  # this method modifies permanently the result dict

    def __save_plot(self, diagrams):
        plot_filename = self.slug + '_plot.svg'
        relative_plot_dir = os.path.join('research', self.research.slug, 'analysis', self.slug)
        absolute_plot_dir = os.path.join(settings.MEDIA_ROOT, relative_plot_dir)
        if not os.path.exists(absolute_plot_dir):
            os.makedirs(absolute_plot_dir)
        ripser.Rips().plot(diagrams)
        plt.savefig(os.path.join(absolute_plot_dir, plot_filename))
        plt.clf()
        self.result_plot = os.path.join(relative_plot_dir, plot_filename)

    def __save_matrix_json(self, analysis_result_matrix):
        for k in analysis_result_matrix:
            if isinstance(analysis_result_matrix[k], numpy.ndarray):
                analysis_result_matrix[k] = analysis_result_matrix[k].tolist()
            elif isinstance(analysis_result_matrix[k], list):
                analysis_result_matrix[k] = [l.tolist() for l in analysis_result_matrix[k]
                                             if isinstance(l, numpy.ndarray)]
        self.result_matrix = json.dumps(analysis_result_matrix)

    def __save_entropy_json(self, diagrams):
        entropies = dict()
        i = 0
        for ripser_matrix in diagrams:
            entropies["H"+str(i)] = FiltrationAnalysis.calculate_entropy(ripser_matrix)
            i = i + 1
        self.result_entropy = json.dumps(entropies)

    @staticmethod
    def calculate_entropy(ripser_matrix):
        if ripser_matrix.size == 0:
            return 0
        non_infinity = list(filter((lambda x: x[1] != math.inf), ripser_matrix))
        if non_infinity == []:  # TODO: check this better
            return 0
        max_death = max(map((lambda x: x[1]), non_infinity)) + 1
        li = list(map((lambda x: x[1]-x[0] if x[1] != math.inf else max_death - x[0]), ripser_matrix))
        ltot = sum(li)
        # maybe check if ltot != 0
        return -sum(map((lambda x: x/ltot * math.log10(x/ltot)), li))


def splitMatrix(m, window, overlap):
    '''
    # for correlation matrix
    if window != 0 and window < len(m):
    raise ValueError("window must be >= the number of rows of input matrix")
    '''
    cols = len(m[0])
    step = window - overlap
    windows = 1 + (cols - window) // step

    for i in range(windows):
        tmp = m[:, window*i - overlap*i: window*(i+1) - overlap*i]
        yield tmp


@receiver(post_save, sender=FiltrationAnalysis)
def run_ripser(sender, instance, **kwargs):

    '''
    if instance.window_size is not None:  # add alert
        windows = splitMatrix(input_matrix, instance.window_size, instance.overlap)
    '''
    if instance.precomputed_distance_matrix:
        ripser_input_matrix = numpy.loadtxt(instance.precomputed_distance_matrix.path)  # TODO: add more read types
    else:
        if instance.filtration_type == FiltrationAnalysis.VIETORIS_RIPS_FILTRATION:
            ripser_input_matrix = instance.dataset.get_distance_matrix(instance.distance_matrix_metric)
        elif instance.filtration_type == FiltrationAnalysis.CLIQUE_WEIGHTED_RANK_FILTRATION:
            ripser_input_matrix = instance.dataset.get_correlation_matrix()
    instance.execute(ripser_input_matrix)
    signals.post_save.disconnect(run_ripser, sender=FiltrationAnalysis)
    instance.save()
    signals.post_save.connect(run_ripser, sender=FiltrationAnalysis)


@receiver(post_save, sender=MapperAnalysis)
def run_kmapper(sender, instance, **kwargs):
    if not instance.precomputed_distance_matrix:
        original_matrix = numpy.array(instance.dataset.data)
        distance_matrix = instance.dataset.get_distance_matrix(instance.distance_matrix_metric)
    else:
        original_matrix = None
        distance_matrix = numpy.loadtxt(instance.precomputed_distance_matrix.path)  # TODO: wrong logic
    instance.execute(original_matrix, distance_matrix)
    signals.post_save.disconnect(run_kmapper, sender=MapperAnalysis)
    instance.save()
    signals.post_save.connect(run_kmapper, sender=MapperAnalysis)
