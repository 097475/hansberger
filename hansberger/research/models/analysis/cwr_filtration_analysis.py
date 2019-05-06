from django.db import models
import numpy
from .analysis import FiltrationAnalysis


class CWRFManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset().filter(
            filtration_type=FiltrationAnalysis.CLIQUE_WEIGHTED_RANK_FILTRATION
        )


class CliqueWeightedRankFiltrationAnalysis(FiltrationAnalysis):
    objects = CWRFManager()

    class Meta:
        proxy = True

    def execute(self, matrix, start_point=None, end_point=None):
        correlation_matrix = numpy.corrcoef(matrix)
        super(CliqueWeightedRankFiltrationAnalysis, self).execute(correlation_matrix)
