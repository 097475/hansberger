import math
import json
import matplotlib
import matplotlib.pyplot as plt
import mpld3
from django.db import models
from django.utils.text import slugify
from django.contrib.postgres.fields import JSONField
import ripser
import kmapper
import sklearn.cluster
import sklearn.preprocessing
import sklearn.mixture
import numpy
import persim
import base64
from io import BytesIO
from research.models import Research
from datasets.models import Dataset
from datasets.models.dataset import distance_matrix, correlation_matrix
from .window import FiltrationWindow, MapperWindow
matplotlib.use('Agg')


class Analysis(models.Model):

    def precomputed_directory_path(instance, filename):
        # file will be uploaded to MEDIA_ROOT/user_<id>/<filename>
        return 'research/precomputed/'+instance.slug+'/'+filename

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
    creation_date = models.DateTimeField(auto_now_add=True)
    research = models.ForeignKey(
        Research,
        on_delete=models.CASCADE
    )
    dataset = models.ForeignKey(
        Dataset,
        on_delete=models.CASCADE,
        blank=True,
        null=True,
        help_text="Select the source dataset from the loaded datasets"
    )
    '''
    precomputed_distance_matrix = models.FileField(upload_to=precomputed_directory_path,
                                                   default=None, null=True, blank=True,
                                                   help_text="""Upload a precomputed distance matrix
                                                   instead of selecting a dataset""")  # TODO
    '''
    precomputed_distance_matrix_json = JSONField(blank=True, null=True)

    window_size = models.PositiveIntegerField(default=None, null=True, blank=True,
                                              help_text="""Leave window size blank to not use windows. Window parameter
                                              is ignored when dealing with precomputed distance matrix. Always check
                                              the dimensions of the dataset your are operating on and plan your windows
                                              accordingly; eventual data that won't fit into the final window will be
                                              discarded.""")
    window_overlap = models.PositiveIntegerField(default=0, help_text="""How many columns of overlap to have in
                                                 consequent windows, if windows are being used. It must be at most 1
                                                 less than window size.""")

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
        'DBSCAN(default)': sklearn.cluster.DBSCAN(min_samples=3),  # should be 3
        'DBSCAN(min_samples=1)': sklearn.cluster.DBSCAN(min_samples=1),  # should be 3
        'gaussian_mixtures': sklearn.mixture.GaussianMixture(),
        'birch': sklearn.cluster.Birch()
    }

    PROJECTION_CHOICES = (
        ('sum', 'Sum'),
        ('mean', 'Mean'),
        ('median', 'Median'),
        ('max', 'Max'),
        ('min', 'Min'),
        ('std', 'Std'),
        ('dist_mean', 'Dist_mean'),
        ('l2norm', 'L2norm'),
        ('knn_distance_n', 'knn_distance_n')
    )

    SCALER_CHOICES = (
        ('None', 'None'),
        ('MinMaxScaler', 'MinMaxScaler'),
        ('MaxAbsScaler', 'MaxAbsScaler'),
        ('RobustScaler', 'RobustScaler'),
        ('StandardScaler', 'StandardScaler'),
    )

    CLUSTERER_CHOICES = (
        ('k-means', 'K-Means'),
        ('affinity_propagation', 'Affinity propagation'),
        ('mean-shift', 'Mean-shift'),
        ('spectral_clustering', 'Spectral clustering'),
        ('agglomerative_clustering', 'StandardScaler'),
        ('DBSCAN(min_samples=1)', 'DBSCAN(min_samples=1)'),
        ('DBSCAN(default)', 'DBSCAN(default)'),
        ('gaussian_mixtures', 'Gaussian mixtures'),
        ('birch', 'Birch')
    )

    distance_matrix_metric = models.CharField(
        max_length=20,
        choices=Analysis.METRIC_CHOICES,
        default='euclidean',
        help_text="If not using a precomputed matrix, choose the distance metric to use on the dataset."
    )

    projection = models.CharField(
                max_length=50,
                choices=PROJECTION_CHOICES,
                help_text="Specify a projection/lens type.",
                default='sum'
                )
    knn_n_value = models.PositiveIntegerField(
                  help_text="Specify the value of n in knn_distance_n",
                  blank=True,
                  null=True
                )
    scaler = models.CharField(
                max_length=50,
                choices=SCALER_CHOICES,
                help_text="Scaler of the data applied after mapping. Use None for no scaling.",
                default='MinMaxScaler'
    )

    use_original_data = models.BooleanField(default=False, help_text="""If ticked, clustering is run on the original data,
                                            else it will be run on the lower dimensional projection.""")
    clusterer = models.CharField(
                max_length=50,
                choices=CLUSTERER_CHOICES,
                default='DBSCAN',
                help_text="Select the clustering algorithm."
                )

    cover_n_cubes = models.PositiveIntegerField(default=10, help_text="""Number of hypercubes along each dimension.
                                        Sometimes referred to as resolution.""")
    cover_perc_overlap = models.FloatField(default=0.5, help_text="""Amount of overlap between adjacent cubes calculated
                                           only along 1 dimension.""")
    graph_nerve_min_intersection = models.IntegerField(default=1, help_text="""Minimum intersection considered when
                                                       computing the nerve. An edge will be created only when the
                                                       intersection between two nodes is greater than or equal to
                                                       min_intersection""")
    precomputed = models.BooleanField(default=False, help_text="""Tell Mapper whether the data that you are clustering on
                                      is a precomputed distance matrix. If set to True, the assumption is that you are
                                      also telling your clusterer that metric=’precomputed’ (which is an argument for
                                      DBSCAN among others), which will then cause the clusterer to expect a square
                                      distance matrix for each hypercube. precomputed=True will give a square matrix
                                      to the clusterer to fit on for each hypercube.""")
    remove_duplicate_nodes = models.BooleanField(default=False, help_text="""Removes duplicate nodes before edges are
                                                 determined. A node is considered to be duplicate if it has exactly
                                                 the same set of points as another node.""")

    class Meta(Analysis.Meta):
        verbose_name = "mapper algorithm analysis"
        verbose_name_plural = "mapper algoritm analyses"

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        run_analysis(self)

    def execute(self, distance_matrix, original_matrix=None, number=0):
        mapper = kmapper.KeplerMapper()
        mycover = kmapper.Cover(n_cubes=self.cover_n_cubes, perc_overlap=self.cover_perc_overlap)
        mynerve = kmapper.GraphNerve(min_intersection=self.graph_nerve_min_intersection)
        original_data = original_matrix if self.use_original_data else None
        projection = self.projection if self.projection != 'knn_distance_n' else 'knn_distance_' + str(self.knn_n_value)
        projected_data = mapper.fit_transform(distance_matrix, projection=projection,
                                              scaler=MapperAnalysis.scalers[self.scaler], distance_matrix=False)
        graph = mapper.map(projected_data, X=original_data, clusterer=MapperAnalysis.clusterers[self.clusterer],
                           cover=mycover, nerve=mynerve, precomputed=self.precomputed,
                           remove_duplicate_nodes=self.remove_duplicate_nodes)
        output_graph = mapper.visualize(graph, save_file=False)
        window = MapperWindow.objects.create_window(number, self)
        window.graph = output_graph
        window.save_window_info()
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
    bottleneck_distance_consecutive = JSONField(blank=True, null=True)
    bottleneck_distance_consecutive_diags = JSONField(blank=True, null=True)

    class Meta(Analysis.Meta):
        verbose_name = "filtration analysis"
        verbose_name_plural = "filtration analyses"

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        run_analysis(self)

    @models.permalink
    def get_absolute_url(self):
        return ('research:filtrationanalysis-detail', (), {'filtrationanalysis_slug': self.slug,
                'research_slug': self.research.slug})

    def execute(self, input_matrix, number=0):
        _thresh = math.inf if self.max_distances_considered is None else self.max_distances_considered
        result = ripser.ripser(input_matrix, maxdim=self.max_homology_dimension, thresh=_thresh, coeff=self.coeff,
                               distance_matrix=True, do_cocycles=self.do_cocycles, n_perm=self.n_perm)
        window = FiltrationWindow.objects.create_window(number, self)
        window.save_diagrams(result['dgms'])
        window.save_entropy_json(result['dgms'])
        window.save_matrix_json(result)  # this method modifies permanently the result dict
        window.save_window_info()
        window.save()

    @property
    def plot_entropy(self):
        entropies = self.get_entropy_data()
        for key in entropies:
            plt.plot(entropies[key], 'o')
        plt.legend([key for key in entropies])
        figure = plt.gcf()
        html_figure = mpld3.fig_to_html(figure, template_type='general')
        plt.clf()
        return html_figure

    def get_entropy_data(self):
        windows = FiltrationWindow.objects.filter(analysis=self).order_by('name')
        entropies = {"H"+str(i): [] for i in range(self.max_homology_dimension + 1)}  # initialize result dict
        entropy_dicts = map(lambda window: json.loads(window.result_entropy), windows)
        for entropy_dict in entropy_dicts:
            for key, value in entropy_dict.items():
                entropies[key].append(value)
        return entropies

    def show_entropy_data(self):
        return json.dumps(self.get_entropy_data())

    def bottleneck_calculation_consecutive(self):
        if self.bottleneck_distance_consecutive or self.bottleneck_distance_consecutive_diags:
            return
        windows = FiltrationWindow.objects.filter(analysis=self).order_by('name')
        distances = {}
        diags = {}
        for i, window1 in enumerate(windows.exclude(name=windows.count()-1)):
            window2 = windows.get(name=i+1)
            print(str(window1.name)+" "+str(window2.name))
            (d, (matching, D)) = persim.bottleneck(json.loads(window1.diagrams)[0], json.loads(window2.diagrams)[0], True) # noqa
            distances[window1.name] = d
            diags[window1.name] = (matching, D.tolist())
        self.bottleneck_distance_consecutive = json.dumps(distances)
        self.bottleneck_distance_consecutive_diags = json.dumps(diags)
        super().save()

    def plot_bottleneck_consecutive(self):
        windows = FiltrationWindow.objects.filter(analysis=self).order_by('name')
        bottleneck_data = json.loads(self.bottleneck_distance_consecutive_diags)
        output_diag = ""
        for i, window1 in enumerate(windows.exclude(name=windows.count()-1)):
            window2 = windows.get(name=i+1)
            current_data = bottleneck_data[str(window1.name)]
            matchidx = current_data[0]
            D = numpy.array(current_data[1])
            persim.bottleneck_matching(window1.get_diagram(0), window2.get_diagram(0), matchidx, D,
                                       labels=["window_"+str(window1.name), "window_"+str(window2.name)])
            buf = BytesIO()
            plt.savefig(buf, format="png")
            # Embed the result in the html output.
            data = base64.b64encode(buf.getbuffer()).decode("ascii")
            plt.clf()
            output_diag = output_diag + f"<img src='data:image/png;base64,{data}'/>"
        return output_diag

    def bottleneck_calculation_alltoall(self):
        windows = FiltrationWindow.objects.filter(analysis=self).order_by('name')
        for window in windows:
            window.bottleneck_calculation()

    def plot_bottleneck_alltoall(self):
        windows = FiltrationWindow.objects.filter(analysis=self).order_by('name')
        bottleneck_data = {}
        for window in windows:
            bottleneck_data[window.name] = json.loads(window.bottleneck_distance_versus_all_diags)
        output_diag = ""
        for i, window1 in enumerate(windows):
            for j in range(i, windows.count()):
                window2 = windows.get(name=j)
                current_data = bottleneck_data[window1.name][str(j)]
                matchidx = current_data[0]
                D = numpy.array(current_data[1])
                persim.bottleneck_matching(window1.get_diagram(0), window2.get_diagram(0), matchidx, D,
                                           labels=["window_"+str(window1.name), "window_"+str(window2.name)])
                buf = BytesIO()
                plt.savefig(buf, format="png")
                # Embed the result in the html output.
                data = base64.b64encode(buf.getbuffer()).decode("ascii")
                plt.clf()
                output_diag = output_diag + f"<img src='data:image/png;base64,{data}'/>"
        return output_diag

    def get_bottleneck_matrix(self):
        windows = FiltrationWindow.objects.filter(analysis=self).order_by('name')
        matrix = []
        for window in windows:
            bottleneck_dict = json.loads(window.bottleneck_distance_versus_all)
            data = [bottleneck_dict[str(i)] for i in range(windows.count())]
            matrix.append(data)
        return matrix


#  multithreading decorator -> add connection.close() at end of function

'''
def start_new_thread(function):
    def decorator(*args, **kwargs):
        t = Thread(target=function, args=args, kwargs=kwargs)
        t.daemon = True
        t.start()
    return decorator
'''


def single_run(instance):
    analysis_type = type(instance)
    print(instance.precomputed_distance_matrix_json)
    if json.loads(instance.precomputed_distance_matrix_json) != []:
        input_matrix = json.loads(instance.precomputed_distance_matrix_json)
        instance.execute(input_matrix)
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


def multiple_run_precomputed(instance, precomputed_matrixes):
    count = 0
    analysis_type = type(instance)
    if analysis_type is FiltrationAnalysis:
        for matrix in precomputed_matrixes:
            instance.execute(numpy.array(matrix), count)
            count = count + 1
    elif analysis_type is MapperAnalysis:
        for matrix in precomputed_matrixes:
            original_matrix = numpy.array(matrix)
            instance.execute(numpy.array(matrix), original_matrix, count)
            count = count + 1


def run_analysis(instance):
    precomputed_distance_matrixes = json.loads(instance.precomputed_distance_matrix_json)
    if instance.window_size is not None and precomputed_distance_matrixes == []:
        window_generator = instance.dataset.split_matrix(instance.window_size, instance.window_overlap)
        multiple_run(instance, window_generator)
    elif precomputed_distance_matrixes != [] and instance.window_size is None:
        multiple_run_precomputed(instance, precomputed_distance_matrixes)
    else:
        single_run(instance)
