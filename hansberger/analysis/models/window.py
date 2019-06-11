import json
import matplotlib
import matplotlib.pyplot as plt
import ripser
import numpy
import math
import base64
import persim
from io import BytesIO
from django.contrib.postgres.fields import JSONField
from django.db import models
from django.utils.text import slugify
matplotlib.use('Agg')


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
        on_delete=models.CASCADE,
        related_name='windows',
        related_query_name='window'
    )
    result_matrix = JSONField(blank=True, null=True)
    diagrams = JSONField(blank=True, null=True)
    result_entropy = JSONField(blank=True, null=True)
    bottleneck_distance_versus_all = JSONField(blank=True, null=True)
    bottleneck_distance_versus_all_diags = JSONField(blank=True, null=True)
    objects = WindowManager()

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
        entropies = dict()
        i = 0
        for ripser_matrix in diagrams:
            entropies["H"+str(i)] = FiltrationWindow.calculate_entropy(ripser_matrix)
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
        plt.clf()
        return f"<img src='data:image/png;base64,{data}'/>"

    def bottleneck_calculation(self):
        windows = FiltrationWindow.objects.filter(analysis=self.analysis).order_by('name')
        distances = {}
        diags = {}
        for window in windows:
            print(window.name)
            (d, (matching, D)) = persim.bottleneck(json.loads(window.diagrams)[0], json.loads(window.diagrams)[0], True)
            distances[window.name] = d
            diags[window.name] = (matching, D.tolist())
        self.bottleneck_distance_versus_all = json.dumps(distances)
        self.bottleneck_distance_versus_all_diags = json.dumps(diags)
        self.save()

    def plot_bottleneck(self):
        windows = FiltrationWindow.objects.filter(analysis=self.analysis).order_by('name')
        bottleneck_data = json.loads(self.bottleneck_distance_versus_all_diags)
        output_diag = ""
        for window in windows:
            current_data = bottleneck_data[str(window.name)]
            matchidx = current_data[0]
            D = numpy.array(current_data[1])
            persim.bottleneck_matching(self.get_diagram(0), window.get_diagram(0), matchidx, D)
            buf = BytesIO()
            plt.savefig(buf, format="png")
            # Embed the result in the html output.
            data = base64.b64encode(buf.getbuffer()).decode("ascii")
            plt.clf()
            output_diag = output_diag + f"<img src='data:image/png;base64,{data}'/>"
        return output_diag


class MapperWindow(Window):
    analysis = models.ForeignKey(
        'analysis.MapperAnalysis',
        on_delete=models.CASCADE,
        related_name='windows',
        related_query_name='window'
    )
    graph = models.TextField(blank=True, null=True)
    objects = WindowManager()
