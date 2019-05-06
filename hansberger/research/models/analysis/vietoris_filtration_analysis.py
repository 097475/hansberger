from django.db import models
import scipy.spatial.distance as dist
from .analysis import FiltrationAnalysis


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
        super(VietorisFiltrationAnalysis, self).execute(distance_matrix)

    # matrix must be numpy array
    @staticmethod
    def __get_distance_matrix(matrix, metric='euclidean'):
        _metric = metric
        return dist.squareform(dist.pdist(matrix.transpose(), metric=_metric))
