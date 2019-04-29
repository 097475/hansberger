from os.path import join
from django.conf import settings
from django.db import models
import scipy.spatial.distance as dist
import ripser
import matplotlib.pyplot as plt
import json
from . import FiltrationAnalysis


class VietorisFiltrationManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset().filter(
            filtration_type=FiltrationAnalysis.VIETORIS_RIPS_FILTRATION
        )


class VietorisFiltrationAnalysis(FiltrationAnalysis):
    objects = VietorisFiltrationManager()

    class Meta:
        proxy = True

    def execute(self, matrix, start_point=None, end_point=None):
        distance_matrix = self.__get_distance_matrix(matrix, metric=self.distance_matrix_metric)
        rips = ripser.Rips(maxdim=self.max_homology_dimension, thresh=self.max_distances_considered, coeff=self.coeff,
                           do_cocycles=self.do_cocycles, n_perm=self.n_perm)
        analysis_result_matrix = rips.fit_transform(distance_matrix, distance_matrix=True)
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

    # matrix must be numpy array
    @staticmethod
    def __get_distance_matrix(matrix, metric='euclidean'):
        _metric = metric
        return dist.squareform(dist.pdist(matrix.transpose(), metric=_metric))
