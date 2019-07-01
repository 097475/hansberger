import matplotlib
import matplotlib.pyplot as plt
import persim
import ripser
import base64
import numpy
import pandas
import gc
from io import BytesIO
from django.db import models
from django.core.exceptions import ObjectDoesNotExist
from ..consumers import StatusHolder, bottleneck_logger_decorator
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

    def manage_persim_crash(self, window, other_window_name):
        diagram = window.get_diagram(self.homology)
        if diagram == []:
            diagram = numpy.empty(shape=(0, 2))
        else:
            diagram = numpy.array(diagram)
        ripser.Rips().plot(diagram, labels='window_'+str(window.name)+', window_'+str(other_window_name))
        # Save it to a temporary buffer.
        buf = BytesIO()
        plt.savefig(buf, format="png")
        # Embed the result in the html output.
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        plt.close()
        return (0, f"<img src='data:image/png;base64,{data}'/>")

    def __bottleneck(self, reference_window, window):
        diag1 = reference_window.get_diagram(self.homology)
        diag2 = window.get_diagram(self.homology)
        if diag1.size == 0 or diag2.size == 0:
            return
        if reference_window == window:
            (d, image) = self.manage_persim_crash(reference_window, window.name)
        else:
            (d, (matching, D)) = persim.bottleneck(diag1, diag2, True)
            image = self.plot_bottleneck(reference_window, window, matching, D)
        diagram = Diagram.objects.create_diagram(self, reference_window, window, d, image)
        diagram.save()

    @bottleneck_logger_decorator
    def bottleneck_calculation_CONS(self, windows):
        last = None
        for window_batch in windows:
            batch = list(window_batch)
            if last is not None and batch != []:
                self.__bottleneck(last, batch[0])
            for i, window1 in enumerate(batch[:-1]):
                window2 = batch[i+1]
                self.__bottleneck(window1, window2)
                if StatusHolder().get_kill():
                    return
                StatusHolder().set_status(window1.name)
            last = batch[-1]

    @bottleneck_logger_decorator
    def bottleneck_calculation_ONE(self, windows):
        reference_window = self.window
        for window_batch in windows:
            for window in window_batch:
                self.__bottleneck(reference_window, window)
                if StatusHolder().get_kill():
                    return
                StatusHolder().set_status(window.name)
            gc.collect()

    @bottleneck_logger_decorator
    def bottleneck_calculation_ALL(self, batch_1, batch_2):
        batch_2 = list(batch_2)
        for window_batch_1 in batch_1:
            for reference_window in window_batch_1:
                StatusHolder().set_status(reference_window.name)
                for window_batch_2 in batch_2:
                    for window in window_batch_2:
                        if window.name < reference_window.name:
                            continue
                        self.__bottleneck(reference_window, window)
                    if StatusHolder().get_kill():
                        return
                    gc.collect()
            gc.collect()

    def plot_bottleneck(self, window1, window2, matchidx, D):
        persim.bottleneck_matching(window1.get_diagram(self.homology), window2.get_diagram(self.homology), matchidx, D,
                                   labels=["window_"+str(window1.name), "window_"+str(window2.name)])
        buf = BytesIO()
        plt.savefig(buf, format="png")
        # Embed the result in the html output.
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        plt.close()
        return f"<img src='data:image/png;base64,{data}'/>"

    def run_bottleneck(self, *args):
        if self.kind == self.ONE:
            self.bottleneck_calculation_ONE(*args)
        elif self.kind == self.CONS:
            self.bottleneck_calculation_CONS(*args)
        elif self.kind == self.ALL:
            self.bottleneck_calculation_ALL(*args)

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
            if matrix == [] or len(matrix) < expected_cols:
                for i in range(expected_cols - len(matrix)):
                    matrix.append([float('NaN')]*expected_cols)
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
