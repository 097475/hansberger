import json
import matplotlib
import matplotlib.pyplot as plt
import ripser
import numpy
import math
import base64
from io import BytesIO
from django.contrib.postgres.fields import JSONField
from django.db import models
from django.utils.text import slugify
from .bottleneck import Bottleneck
matplotlib.use('Agg')


def window_batch_generator(analysis):
    window_count = FiltrationWindow.objects.filter(analysis=analysis).order_by('name').count()
    batch_size = window_count // 4
    for window_batch in range(window_count // batch_size + 1):
        windows = FiltrationWindow.objects.filter(analysis=analysis).order_by(
                  'name')[batch_size*window_batch:batch_size*window_batch+batch_size]
        yield windows


class WindowManager(models.Manager):
    def create_window(self, name, analysis):
        window = self.create(name=name, analysis=analysis)
        return window


class Window(models.Model):
    name = models.PositiveIntegerField()
    slug = models.SlugField(db_index=True, max_length=150)
    creation_date = models.DateTimeField(auto_now_add=True)
    start = models.PositiveIntegerField(null=True, blank=True)
    end = models.PositiveIntegerField(null=True, blank=True)

    class Meta:
        abstract = True

    def save_window_info(self):
        analysis = self.analysis
        if json.loads(analysis.precomputed_distance_matrix_json) != []:  # no windows and no datasets are being used
            self.start = None
            self.end = None
        else:
            if analysis.window_size is None:  # no windows are used
                self.start = 0
                self.end = analysis.dataset.cols
            else:
                self.start = 0 if self.name == 0 else self.name * analysis.window_size - analysis.window_overlap
                self.end = self.start + analysis.window_size

    def save(self, *args, **kwargs):
        if not self.id:
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)


class FiltrationWindow(Window):
    analysis = models.ForeignKey(
        'analysis.FiltrationAnalysis',
        on_delete=models.CASCADE
    )
    result_matrix = JSONField(blank=True, null=True)
    diagrams = JSONField(blank=True, null=True)
    result_entropy_normalized = JSONField(blank=True, null=True)
    result_entropy_unnormalized = JSONField(blank=True, null=True)
    objects = WindowManager()

    def save_data(self, result):
        self.save_diagrams(result['dgms'])
        self.save_entropy_json(result['dgms'])
        self.save_matrix_json(result)  # this method modifies permanently the result dict
        self.save_window_info()
        self.save()

    def save_diagrams(self, diagrams):
        self.diagrams = json.dumps([d.tolist() for d in diagrams])

    def get_diagram(self, homology):
        diagrams = json.loads(self.diagrams)
        return numpy.array(diagrams[homology])

    def save_matrix_json(self, analysis_result_matrix):
        for k in analysis_result_matrix:
            if isinstance(analysis_result_matrix[k], numpy.ndarray):
                analysis_result_matrix[k] = analysis_result_matrix[k].tolist()
            elif isinstance(analysis_result_matrix[k], list):
                analysis_result_matrix[k] = [l.tolist() for l in analysis_result_matrix[k]
                                             if isinstance(l, numpy.ndarray)]
        self.result_matrix = json.dumps(analysis_result_matrix)

    def save_entropy_json(self, diagrams):
        entropies_normalized = dict()
        entropies_unnormalized = dict()
        i = 0
        for ripser_matrix in diagrams:
            entropies_normalized["H"+str(i)] = FiltrationWindow.calculate_entropy(ripser_matrix, True)
            entropies_unnormalized["H"+str(i)] = FiltrationWindow.calculate_entropy(ripser_matrix, False)
            i = i + 1
        self.result_entropy_normalized = json.dumps(entropies_normalized)
        self.result_entropy_unnormalized = json.dumps(entropies_unnormalized)

    @staticmethod
    def calculate_entropy(ripser_matrix, normalize=False):
        if ripser_matrix.size == 0:
            return 0
        non_infinity = list(filter((lambda x: x[1] != math.inf), ripser_matrix))
        if non_infinity == []:  # single infinity element
            return 0
        max_death = max(map((lambda x: x[1]), non_infinity)) + 1
        li = list(map((lambda x: x[1]-x[0] if x[1] != math.inf else max_death - x[0]), ripser_matrix))
        ltot = sum(li)
        if normalize:
            norm_value = (1 / numpy.log10(len(ripser_matrix))) if len(ripser_matrix) != 1 else 1
            return norm_value * -sum(map((lambda x: x/ltot * numpy.log10(x/ltot)), li))
        else:
            return -sum(map((lambda x: x/ltot * math.log10(x/ltot)), li))

    @property
    def plot(self):
        diagrams = []
        for diagram in json.loads(self.diagrams):
            if diagram == []:
                diagrams.append(numpy.empty(shape=(0, 2)))
            else:
                diagrams.append(numpy.array(diagram))
        ripser.Rips().plot(diagrams)
        # Save it to a temporary buffer.
        buf = BytesIO()
        plt.savefig(buf, format="png")
        # Embed the result in the html output.
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        plt.close()
        return f"<img src='data:image/png;base64,{data}'/>"

    def bottleneck_calculation_onetoall(self, homology):
        if Bottleneck.objects.filter(window=self, kind=Bottleneck.ONE, homology=homology).count() == 1:
            return
        windows = FiltrationWindow.objects.filter(analysis=self.analysis).order_by('name')
        bottleneck = Bottleneck.objects.create_bottleneck(self, Bottleneck.ONE, homology)
        bottleneck.run_bottleneck(windows)
        bottleneck.save()

    def get_bottleneck(self, homology):
        return Bottleneck.objects.get(window=self, kind=Bottleneck.ONE, homology=homology)


class MapperWindow(Window):
    analysis = models.ForeignKey(
        'analysis.MapperAnalysis',
        on_delete=models.CASCADE,
        related_name='windows',
        related_query_name='window'
    )
    graph = models.TextField(blank=True, null=True)
    objects = WindowManager()

    def save_data(self, output_graph):
        self.graph = output_graph
        self.save_window_info()
        self.save()
