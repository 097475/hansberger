import matplotlib
import matplotlib.pyplot as plt
import persim
import ripser
import base64
import json
import numpy
import pandas
from io import BytesIO
from django.db import models
from django.core.exceptions import ObjectDoesNotExist
matplotlib.use('Agg')


class BottleneckManager(models.Manager):
    def create_bottleneck(self, owner, kind, homology):
        if kind == Bottleneck.CONS or kind == Bottleneck.ALL:
            bottleneck = self.create(analysis=owner, kind=kind, homology=homology)
        elif kind == Bottleneck.ONE:
            bottleneck = self.create(window=owner, kind=kind, homology=homology)
        return bottleneck


class DiagramManager(models.Manager):
    def create_diagram(self, bottleneck, window1, window2, value, image):
        return self.create(bottleneck=bottleneck, window1=window1, window2=window2, bottleneck_distance=value,
                           image=image)


class Bottleneck(models.Model):
    CONS = 'consecutive'
    ONE = 'one_to_all'
    ALL = 'all_to_all'
    BOTTLENECK_TYPES = [(CONS, 'consecutive'), (ONE, 'one_to_all'), (ALL, 'all_to_all')]
    analysis = models.ForeignKey(
        'analysis.FiltrationAnalysis',
        on_delete=models.CASCADE,
        null=True
    )
    window = models.ForeignKey(
        'analysis.FiltrationWindow',
        on_delete=models.CASCADE,
        null=True
    )
    homology = models.PositiveIntegerField()
    kind = models.CharField(choices=BOTTLENECK_TYPES, max_length=20)
    objects = BottleneckManager()

    def manage_persim_crash(self, window):
        diagram = json.loads(window.diagrams)[self.homology]
        if diagram == []:
            diagram = numpy.empty(shape=(0, 2))
        else:
            diagram = numpy.array(diagram)
        ripser.Rips().plot(diagram, labels='window_'+str(window.name))
        # Save it to a temporary buffer.
        buf = BytesIO()
        plt.savefig(buf, format="png")
        # Embed the result in the html output.
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        plt.clf()
        return (0, f"<img src='data:image/png;base64,{data}'/>")

    def bottleneck_calculation_CONS(self, windows):
        for i, window1 in enumerate(windows.exclude(name=windows.count()-1)):
            window2 = windows.get(name=i+1)
            print(str(window1.name)+" "+str(window2.name))
            diag1 = json.loads(window1.diagrams)[self.homology]
            diag2 = json.loads(window2.diagrams)[self.homology]
            if diag1 == [] or diag2 == []:
                continue
            (d, (matching, D)) = persim.bottleneck(diag1, diag2, True) # noqa
            image = self.plot_bottleneck(window1, window2, matching, D)
            diagram = Diagram.objects.create_diagram(self, window1, window2, d, image)
            diagram.save()

    def bottleneck_calculation_ONE(self, windows):
        reference_window = self.window
        for window in windows:
            print(window.name)
            diag1 = json.loads(reference_window.diagrams)[self.homology]
            diag2 = json.loads(window.diagrams)[self.homology]
            if diag1 == [] or diag2 == []:
                continue
            if reference_window == window and len(diag1) == 1:
                (d, image) = self.manage_persim_crash(window)
            else:
                (d, (matching, D)) = persim.bottleneck(diag1, diag2, True)
                image = self.plot_bottleneck(reference_window, window, matching, D)
            diagram = Diagram.objects.create_diagram(self, reference_window, window, d, image)
            diagram.save()

    def bottleneck_calculation_ALL(self, windows):
        for reference_window in windows:
            for window in windows:
                if window.name < reference_window.name:
                    continue
                print(window.name)
                diag1 = json.loads(reference_window.diagrams)[self.homology]
                diag2 = json.loads(window.diagrams)[self.homology]
                if diag1 == [] or diag2 == []:
                    continue
                if reference_window == window and len(diag1) == 1:
                    (d, image) = self.manage_persim_crash(window)
                else:
                    (d, (matching, D)) = persim.bottleneck(diag1, diag2, True)
                    image = self.plot_bottleneck(reference_window, window, matching, D)
                diagram = Diagram.objects.create_diagram(self, reference_window, window, d, image)
                diagram.save()

    def plot_bottleneck(self, window1, window2, matchidx, D):
        persim.bottleneck_matching(window1.get_diagram(self.homology), window2.get_diagram(self.homology), matchidx, D,
                                   labels=["window_"+str(window1.name), "window_"+str(window2.name)])
        buf = BytesIO()
        plt.savefig(buf, format="png")
        # Embed the result in the html output.
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        plt.clf()
        return f"<img src='data:image/png;base64,{data}'/>"

    def run_bottleneck(self, windows):
        if self.kind == self.ONE:
            self.bottleneck_calculation_ONE(windows)
        elif self.kind == self.CONS:
            self.bottleneck_calculation_CONS(windows)
        elif self.kind == self.ALL:
            self.bottleneck_calculation_ALL(windows)

    def get_bottleneck_matrix(self):
        diagrams = Diagram.objects.filter(bottleneck=self).order_by('window1__name', 'window2__name')
        if self.kind == self.ONE:
            data = []
            labels = []
            reference_window = diagrams.first().window1.name
            for diagram in diagrams:
                data.append(diagram.bottleneck_distance)
                labels.append(str(diagram.window2.name))
            df = pandas.DataFrame([data], index=[str(reference_window)], columns=labels)
            return df.to_csv(index=True, header=True)
        elif self.kind == self.CONS:
            data = []
            labels = []
            for diagram in diagrams:
                data.append(diagram.bottleneck_distance)
                labels.append(str(diagram.window1.name))
            df = pandas.DataFrame([data], index=['n+1'], columns=labels)
            return df.to_csv(index=True, header=True)
        elif self.kind == self.ALL:
            row = diagrams.filter(window1__name=0)
            n_cols = row.count()  # actual number of cols
            expected_cols = self.analysis.get_window_number()
            labels = [i for i in range(expected_cols)]
            matrix = []
            i = 0
            while(n_cols != 0):
                current_row = []
                for j in range(expected_cols):
                    if j < i:
                        current_row.append(0)
                    else:
                        try:
                            current_row.append(diagrams.get(window1__name=i, window2__name=j).bottleneck_distance)
                        except ObjectDoesNotExist:
                            current_row.append(float('NaN'))
                matrix.append(current_row)
                i = i + 1
                row = diagrams.filter(window1__name=i)
                n_cols = row.count()
            if matrix == []:
                out = matrix
            else:
                matrix = numpy.array(matrix)
                out = matrix.T + matrix
                numpy.fill_diagonal(out, numpy.diag(matrix))
            df = pandas.DataFrame(out, index=labels, columns=labels)
            return df.to_csv(index=True, header=True)

    def get_diagrams(self):
        return Diagram.objects.filter(bottleneck=self).order_by('window1__name', 'window2__name')


class Diagram(models.Model):
    bottleneck = models.ForeignKey(
        Bottleneck,
        on_delete=models.CASCADE
    )
    window1 = models.ForeignKey(
        'analysis.FiltrationWindow',
        on_delete=models.CASCADE,
        related_name='window1'
    )
    window2 = models.ForeignKey(
        'analysis.FiltrationWindow',
        on_delete=models.CASCADE,
        related_name='window2'
    )
    image = models.TextField()
    bottleneck_distance = models.FloatField()
    objects = DiagramManager()
