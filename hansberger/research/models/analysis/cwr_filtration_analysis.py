from os.path import join
from django.conf import settings
from django.db import models
import ripser
import matplotlib.pyplot as plt
import json
import numpy
from ..models import FiltrationAnalysis


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
        rips = ripser.Rips(maxdim=self.max_homology_dimension, thresh=self.max_distances_considered, coeff=self.coeff,
                           do_cocycles=self.do_cocycles, n_perm=self.n_perm)
        analysis_result_matrix = rips.fit_transform(correlation_matrix, distance_matrix=True)
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
