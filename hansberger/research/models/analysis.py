import json
import math
import os.path
from django.conf import settings
from django.contrib.postgres.fields import JSONField
from django.db import models
from django.utils.text import slugify
from django.db.models.signals import post_save
from django.dispatch import receiver
import ripser
import kmapper
import matplotlib.pyplot as plt
import sklearn.cluster
import sklearn.preprocessing
import sklearn.mixture
import numpy
from .research import Research
from .dataset import Dataset, distance_matrix, correlation_matrix
from .window import FiltrationWindow, MapperWindow


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
    name = models.CharField(max_length=100, help_text="Name this analysis")
    slug = models.SlugField(db_index=True, max_length=110)
    description = models.TextField(max_length=500, blank=True, help_text="Write a brief description of the analysis")
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
        null=True,
        help_text="Select the source dataset from the loaded datasets"
    )

    precomputed_distance_matrix = models.FileField(upload_to='research/precomputed/', default=None, null=True,
                                                   blank=True, help_text="""Upload a precomputed distance matrix
                                                   instead of selecting a dataset""")  # TODO
    window_size = models.PositiveIntegerField(default=None, null=True, blank=True,
                                              help_text="Leave window size blank to not use windows.")
    window_overlap = models.PositiveIntegerField(default=0, help_text="""How many columns of overlap to have in
                                                 consequent windows. It must be at most 1 less than window size.""")

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

    distance_matrix_metric = models.CharField(
        max_length=20,
        choices=Analysis.METRIC_CHOICES,
        default='euclidean',
        help_text="If not using a precomputed matrix, choose the distance metric to use on the dataset."
    )
    # fit_transform parameters; not implemented : scaler params, scikit projections
    projection = models.CharField(
                max_length=50,
                choices=PROJECTION_CHOICES,
                help_text="Specify a projection/lens type."
                )
    scaler = models.CharField(
                max_length=50,
                choices=SCALER_CHOICES,
                help_text="Scaler of the data applied after mapping. Use None for no scaling."
    )

    # map parameters; not implemented : clusterer params, cover limits
    use_original_data = models.BooleanField(default=False, help_text="""If ticked, clustering is run on the original data,
                                            else it will be run on the lower dimensional projection.""")
    clusterer = models.CharField(
                max_length=50,
                choices=CLUSTERER_CHOICES,
                default='DBSCAN',
                help_text="Select the clustering algorithm."
                )
    # missing cover limits
    cover_n_cubes = models.IntegerField(default=10, help_text="""Number of hypercubes along each dimension.
                                        Sometimes referred to as resolution.""")
    cover_perc_overlap = models.FloatField(default=0.5, help_text="""Amount of overlap between adjacent cubes calculated
                                           only along 1 dimension.""")
    graph_nerve_min_intersection = models.IntegerField(default=1, help_text="""Minimum intersection considered when
                                                       computing the nerve. An edge will be created only when the
                                                       intersection between two nodes is greater than or equal to
                                                       min_intersection""")
    remove_duplicate_nodes = models.BooleanField(default=False, help_text="""Removes duplicate nodes before edges are
                                                 determined. A node is considered to be duplicate if it has exactly
                                                 the same set of points as another node.""")

    graph = models.TextField(blank=True, null=True)

    class Meta(Analysis.Meta):
        verbose_name = "mapper algorithm analysis"
        verbose_name_plural = "mapper algoritm analyses"

    # TODO: check precomputed=False
    def execute(self, distance_matrix, original_matrix=None, number=0):
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
        window = MapperWindow.objects.create_window(str(number), self)
        window.graph = output_graph
        window.save()


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
        help_text="Choose the type of analysis."
    )
    distance_matrix_metric = models.CharField(
        max_length=20,
        choices=Analysis.METRIC_CHOICES,
        blank=True,
        help_text="""If Vietoris-Rips filtration is selected and not using a precomputed distance matrix, choose the
                  distance metric to use on the selected dataset. This parameter is ignored in all other cases."""
    )
    max_homology_dimension = models.PositiveIntegerField(default=1, help_text="""Maximum homology dimension computed. Will compute all dimensions lower than and equal to this value.
                                                 For 1, H_0 and H_1 will be computed.""")
    max_distances_considered = models.FloatField(default=None, null=True, blank=True, help_text="""Maximum distances considered when constructing filtration.
                                                 If blank, compute the entire filtration.""")
    coeff = models.PositiveIntegerField(default=2, help_text="""Compute homology with coefficients in the prime field Z/pZ for
                                p=coeff.""")
    do_cocycles = models.BooleanField(default=False, help_text="Indicator of whether to compute cocycles.")
    n_perm = models.IntegerField(default=None, null=True, blank=True, help_text="""The number of points to subsample in
                                 a “greedy permutation,” or a furthest point sampling of the points. These points will
                                 be used in lieu of the full point cloud for a faster computation, at the expense of
                                 some accuracy, which can be bounded as a maximum bottleneck distance to all diagrams
                                 on the original point set""")

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

    def execute(self, input_matrix, number=0):
        _thresh = math.inf if self.max_distances_considered is None else self.max_distances_considered
        result = ripser.ripser(input_matrix, maxdim=self.max_homology_dimension, thresh=_thresh, coeff=self.coeff,
                               distance_matrix=True, do_cocycles=self.do_cocycles, n_perm=self.n_perm)
        window = FiltrationWindow.objects.create_window(number, self)
        window.save_plot(result['dgms'])
        window.save_entropy_json(result['dgms'])
        window.save_matrix_json(result)  # this method modifies permanently the result dict
        window.save()

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


def single_run(instance):
    analysis_type = type(instance)
    if instance.precomputed_distance_matrix:
        input_matrix = numpy.loadtxt(instance.precomputed_distance_matrix.path)
        instance.execute(input_matrix)  # TODO: add more read types
    elif analysis_type is FiltrationAnalysis:
        if instance.filtration_type == FiltrationAnalysis.VIETORIS_RIPS_FILTRATION:
            input_matrix = instance.dataset.get_distance_matrix(instance.distance_matrix_metric)
        elif instance.filtration_type == FiltrationAnalysis.CLIQUE_WEIGHTED_RANK_FILTRATION:
            input_matrix = instance.dataset.get_correlation_matrix()
        instance.execute(input_matrix)
    elif analysis_type is MapperAnalysis:
        input_matrix = instance.dataset.get_distance_matrix(instance.distance_matrix_metric)
        original_matrix = numpy.array(instance.dataset.data)
        instance.execute(input_matrix, original_matrix)


def multiple_run(instance, window_generator):
    count = 0
    analysis_type = type(instance)
    if analysis_type is FiltrationAnalysis:
        for window in window_generator:
            if instance.filtration_type == FiltrationAnalysis.VIETORIS_RIPS_FILTRATION:
                input_matrix = distance_matrix(window, instance.distance_matrix_metric)
            elif instance.filtration_type == FiltrationAnalysis.CLIQUE_WEIGHTED_RANK_FILTRATION:
                input_matrix = correlation_matrix(window)
            instance.execute(input_matrix, count)
            count = count + 1
    elif analysis_type is MapperAnalysis:
        for window in window_generator:
            input_matrix = distance_matrix(window, instance.distance_matrix_metric)
            original_matrix = numpy.array(window)
            instance.execute(input_matrix, original_matrix, count)
            count = count + 1


#  traspose before or after splitting?
@receiver(post_save, sender=FiltrationAnalysis)
@receiver(post_save, sender=MapperAnalysis)
def run_ripser(sender, instance, **kwargs):
    #  TODO: alert about wrong overlap and/or window size!
    if instance.window_size is not None:  # add alert
        window_generator = instance.dataset.split_matrix(instance.window_size, instance.window_overlap)
        multiple_run(instance, window_generator)
    else:
        single_run(instance)
