import numpy
import json
from itertools import chain
from django.shortcuts import redirect
from django.http import HttpResponse
from django_downloadview import VirtualDownloadView
from django.core.files.base import ContentFile
from django.shortcuts import get_object_or_404, render
from django.urls import reverse_lazy
from django.db.transaction import non_atomic_requests
from django.utils.decorators import method_decorator
from django.views.generic import (
    View,
    CreateView,
    DeleteView,
    DetailView,
    ListView,
)
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from .models import (
    Analysis,
    FiltrationAnalysis,
    MapperAnalysis,
    Window,
    FiltrationWindow,
    MapperWindow,
    Bottleneck,
)
from research.models import Research
from .forms import (
    SourceChoiceForm,
    FiltrationAnalysisCreationForm_Dataset,
    FiltrationAnalysisCreationForm_Precomputed,
    MapperAnalysisCreationForm_Dataset,
    MapperAnalysisCreationForm_Precomputed,
    AnalysisBottleneckCreationForm,
    WindowBottleneckCreationForm
)

form_dict = {
    'filtration_analysis': {
        'precomputed': FiltrationAnalysisCreationForm_Precomputed,
        'dataset': FiltrationAnalysisCreationForm_Dataset
    },
    'mapper_analysis': {
        'precomputed': MapperAnalysisCreationForm_Precomputed,
        'dataset': MapperAnalysisCreationForm_Dataset
    }
}


def SourceChoice(request, research_slug):
    research = get_object_or_404(Research, slug=research_slug)
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = SourceChoiceForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            cleaned_data = form.cleaned_data
            analysis = cleaned_data.get("analysis")
            source = cleaned_data.get("source")
            if analysis == 'filtration_analysis':
                return redirect('analysis:filtrationanalysis-create', form=source, research_slug=research.slug)
            elif analysis == 'mapper_analysis':
                return redirect('analysis:mapperanalysis-create', form=source, research_slug=research.slug)

    # if a GET (or any other method) we'll create a blank form
    else:
        form = SourceChoiceForm()

    return render(request, 'analysis/analysis_source_choice.html', {'form': form, 'research': research})


class AnalysisDetailView(View):
    def get(self, request, *args, **kwargs):
        my_analysis = (FiltrationAnalysis.objects.filter(
            research__slug=self.kwargs['research_slug'],
            slug=self.kwargs['analysis_slug']
            ).first()
            or
            MapperAnalysis.objects.filter(
                research__slug=self.kwargs['research_slug'],
                slug=self.kwargs['analysis_slug']
            ).first())
        if isinstance(my_analysis, FiltrationAnalysis):
            return render(request, 'analysis/filtrationanalysis_detail.html', context={'analysis': my_analysis,
                          'homology': range(my_analysis.max_homology_dimension + 1)})
        elif isinstance(my_analysis, MapperAnalysis):
            return render(request, 'analysis/mapperanalysis_detail.html', context={'analysis': my_analysis})


@method_decorator(non_atomic_requests, name='dispatch')
class AnalysisDeleteView(DeleteView):
    model = Analysis
    context_object_name = 'analysis'
    template_name = "analysis/analysis_confirm_delete.html"

    def get_object(self):
        return (FiltrationAnalysis.objects.filter(
            research__slug=self.kwargs['research_slug'],
            slug=self.kwargs['analysis_slug']
            ).first()
            or
            MapperAnalysis.objects.filter(
            research__slug=self.kwargs['research_slug'],
            slug=self.kwargs['analysis_slug']
            ).first())

    def get_success_url(self):
        return reverse_lazy('analysis:analysis-list', kwargs={
                'research_slug': self.kwargs['research_slug']
        })


class AnalysisListView(ListView):
    model = Analysis
    context_object_name = 'analyses'
    paginate_by = 10
    template_name = "analysis/analysis_list.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['research'] = self.research
        return context

    def get_queryset(self):
        self.research = get_object_or_404(
            Research,
            slug=self.kwargs['research_slug']
        )
        filtration_analyses = FiltrationAnalysis.objects.filter(
            research=self.research
        ).only('name', 'creation_date', 'slug', 'research')
        mapper_analyses = MapperAnalysis.objects.filter(
            research=self.research
        ).only('name', 'creation_date', 'slug', 'research')
        return sorted(chain(filtration_analyses, mapper_analyses), key=lambda x: x.creation_date, reverse=True)


@method_decorator(non_atomic_requests, name='dispatch')
class FiltrationAnalysisCreateView(CreateView):
    model = FiltrationAnalysis

    def get_template_names(self):
        print(self.get_form_class())
        if self.get_form_class() is FiltrationAnalysisCreationForm_Dataset:
            return "analysis/filtrationanalysis_dataset_form.html"
        elif self.get_form_class() is FiltrationAnalysisCreationForm_Precomputed:
            return "analysis/filtrationanalysis_precomputed_form.html"

    def get_form_class(self):
        return form_dict['filtration_analysis'][self.kwargs['form']]

    def get_success_url(self):
        return reverse_lazy('analysis:analysis-detail', kwargs={
                'research_slug': self.kwargs['research_slug'],
                'analysis_slug': self.analysis.slug
        })

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['research'] = self.research
        return context

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        self.research = get_object_or_404(
            Research,
            slug=self.kwargs['research_slug']
        )
        kwargs['research'] = self.research
        return kwargs

    def form_valid(self, form):
        self.analysis = form.save(commit=False)
        self.analysis.precomputed_distance_matrix_json = self.precomputed_distance_matrix_json
        return super().form_valid(form)

    def post(self, request, *args, **kwargs):
        self.object = None
        form_class = self.get_form_class()
        form = self.get_form(form_class)
        files = request.FILES.getlist('precomputed_distance_matrix')
        if form.is_valid():
            precomputed_distance_matrixes = []
            for f in files:
                print(f)
                precomputed_distance_matrixes.append(numpy.loadtxt(f).tolist())
            self.precomputed_distance_matrix_json = json.dumps(precomputed_distance_matrixes)
            return self.form_valid(form)
        else:
            return self.form_invalid(form)


@method_decorator(non_atomic_requests, name='dispatch')
class MapperAnalysisCreateView(CreateView):
    model = MapperAnalysis

    def get_template_names(self):
        if self.get_form_class() is MapperAnalysisCreationForm_Dataset:
            return "analysis/mapperanalysis_dataset_form.html"
        elif self.get_form_class() is MapperAnalysisCreationForm_Precomputed:
            return "analysis/mapperanalysis_precomputed_form.html"

    def get_form_class(self):
        return form_dict['mapper_analysis'][self.kwargs['form']]

    def get_success_url(self):
        return reverse_lazy('analysis:analysis-detail', kwargs={
                'research_slug': self.kwargs['research_slug'],
                'analysis_slug': self.analysis.slug
        })

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['research'] = self.research
        return context

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        self.research = get_object_or_404(
            Research,
            slug=self.kwargs['research_slug']
        )
        kwargs['research'] = self.research
        return kwargs

    def form_valid(self, form):
        self.analysis = form.save(commit=False)
        self.analysis.precomputed_distance_matrix_json = self.precomputed_distance_matrix_json
        return super().form_valid(form)

    def post(self, request, *args, **kwargs):
        self.object = None
        form_class = self.get_form_class()
        form = self.get_form(form_class)
        files = request.FILES.getlist('precomputed_distance_matrix')
        if form.is_valid():
            precomputed_distance_matrixes = []
            for f in files:
                print(f)
                precomputed_distance_matrixes.append(numpy.loadtxt(f).tolist())
            self.precomputed_distance_matrix_json = json.dumps(precomputed_distance_matrixes)
            return self.form_valid(form)
        else:
            return self.form_invalid(form)


class MapperAnalysisView(View):
    def get(self, request, *args, **kwargs):
        my_analysis = get_object_or_404(
            MapperAnalysis,
            research__slug=self.kwargs['research_slug'],
            slug=self.kwargs['analysis_slug']
        )
        my_window = get_object_or_404(
            MapperWindow,
            analysis=my_analysis,
            slug=self.kwargs['window_slug']
        )
        return HttpResponse(my_window.graph)


class WindowDetailView(DetailView):
    def get(self, request, *args, **kwargs):
        my_analysis = (FiltrationAnalysis.objects.filter(
            research__slug=self.kwargs['research_slug'],
            slug=self.kwargs['analysis_slug']
            ).first()
            or
            MapperAnalysis.objects.filter(
                research__slug=self.kwargs['research_slug'],
                slug=self.kwargs['analysis_slug']
            ).first())
        if type(my_analysis) is FiltrationAnalysis:
            my_window = get_object_or_404(
                        FiltrationWindow,
                        analysis=my_analysis,
                        slug=self.kwargs['window_slug']
                        )
        elif type(my_analysis) is MapperAnalysis:
            my_window = get_object_or_404(
                        MapperWindow,
                        analysis=my_analysis,
                        slug=self.kwargs['window_slug']
                        )
        if isinstance(my_window, FiltrationWindow):
            return render(request, 'analysis/window/filtrationwindow_detail.html', context={'window': my_window,
                          'homology': range(my_analysis.max_homology_dimension + 1)})
        elif isinstance(my_window, MapperWindow):
            return render(request, 'analysis/window/mapperwindow_detail.html', context={'window': my_window})


class WindowListView(ListView):
    model = Window
    context_object_name = 'windows'
    paginate_by = 10
    template_name = "analysis/window/window_list.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['analysis'] = self.analysis
        return context

    def get_queryset(self):
        self.analysis = (FiltrationAnalysis.objects.filter(
            research__slug=self.kwargs['research_slug'],
            slug=self.kwargs['analysis_slug']
            ).first()
            or
            MapperAnalysis.objects.filter(
                research__slug=self.kwargs['research_slug'],
                slug=self.kwargs['analysis_slug']
            ).first())
        if type(self.analysis) is FiltrationAnalysis:
            windows = FiltrationWindow.objects.filter(
                      analysis=self.analysis
                      )
        elif type(self.analysis) is MapperAnalysis:
            windows = MapperWindow.objects.filter(
                      analysis=self.analysis
                      )
        return windows.only('name', 'creation_date', 'slug').order_by('name')


@method_decorator(non_atomic_requests, name='dispatch')
class WindowBottleneckView(View):
    def get(self, request, *args, **kwargs):
        my_analysis = get_object_or_404(
                        FiltrationAnalysis,
                        research__slug=self.kwargs['research_slug'],
                        slug=self.kwargs['analysis_slug']
                        )
        my_window = get_object_or_404(
            FiltrationWindow,
            analysis=my_analysis,
            slug=self.kwargs['window_slug']
        )
        my_window.bottleneck_calculation_onetoall(self.kwargs['homology'])
        bottleneck = my_window.get_bottleneck(self.kwargs['homology'])
        diagram_list = bottleneck.get_diagrams()
        page = request.GET.get('page', 1)
        paginator = Paginator(diagram_list, 10)
        try:
            diagrams = paginator.page(page)
        except PageNotAnInteger:
            diagrams = paginator.page(1)
        except EmptyPage:
            diagrams = paginator.page(paginator.num_pages)
        return render(request, 'analysis/window/filtrationwindow_bottleneck.html', context={'diagrams': diagrams,
                                                                                            'window': my_window})


class AnalysisConsecutiveBottleneckView(View):
    def get(self, request, *args, **kwargs):
        my_analysis = get_object_or_404(
                        FiltrationAnalysis,
                        research__slug=self.kwargs['research_slug'],
                        slug=self.kwargs['analysis_slug']
                        )
        my_analysis.bottleneck_calculation_consecutive(self.kwargs['homology'])
        bottleneck = my_analysis.get_bottleneck(Bottleneck.CONS, self.kwargs['homology'])
        diagram_list = bottleneck.get_diagrams()
        page = request.GET.get('page', 1)
        paginator = Paginator(diagram_list, 10)
        try:
            diagrams = paginator.page(page)
        except PageNotAnInteger:
            diagrams = paginator.page(1)
        except EmptyPage:
            diagrams = paginator.page(paginator.num_pages)
        return render(request, 'analysis/filtrationanalysis_bottleneck_consecutive.html',
                      context={'diagrams': diagrams,
                               'analysis': my_analysis
                               })


class AnalysisAlltoallBottleneckView(View):
    def get(self, request, *args, **kwargs):
        my_analysis = get_object_or_404(
                        FiltrationAnalysis,
                        research__slug=self.kwargs['research_slug'],
                        slug=self.kwargs['analysis_slug']
                        )

        my_analysis.bottleneck_calculation_alltoall(self.kwargs['homology'])
        bottleneck = my_analysis.get_bottleneck(Bottleneck.ALL, self.kwargs['homology'])
        diagram_list = bottleneck.get_diagrams()
        page = request.GET.get('page', 1)
        paginator = Paginator(diagram_list, 10)
        try:
            diagrams = paginator.page(page)
        except PageNotAnInteger:
            diagrams = paginator.page(1)
        except EmptyPage:
            diagrams = paginator.page(paginator.num_pages)
        return render(request, 'analysis/filtrationanalysis_bottleneck_alltoall.html',
                      context={'diagrams': diagrams,
                               'analysis': my_analysis
                               })


class RipserDownloadView(VirtualDownloadView):

    def get_object(self):
        my_analysis = get_object_or_404(
                        FiltrationAnalysis,
                        research__slug=self.kwargs['research_slug'],
                        slug=self.kwargs['analysis_slug']
                        )
        return get_object_or_404(
            FiltrationWindow,
            analysis=my_analysis,
            slug=self.kwargs['window_slug']
        )

    def get_file(self):
        window_analysis = self.get_object()
        return ContentFile(window_analysis.result_matrix, name=window_analysis.analysis.research.name + '_' +
                           window_analysis.analysis.name + '_' + str(window_analysis.name) + '.dat')


class EntropyDownloadView(VirtualDownloadView):

    def get_object(self):
        return get_object_or_404(
                        FiltrationAnalysis,
                        research__slug=self.kwargs['research_slug'],
                        slug=self.kwargs['analysis_slug']
                        )

    def get_file(self):
        analysis = self.get_object()
        return ContentFile(json.dumps(analysis.get_entropy_csv()), name=analysis.research.name + '_' +
                           analysis.name + '_entropy.csv')


class BottleneckONEDownloadView(VirtualDownloadView):
    def get_object(self):
        my_analysis = get_object_or_404(
                        FiltrationAnalysis,
                        research__slug=self.kwargs['research_slug'],
                        slug=self.kwargs['analysis_slug']
                        )
        my_window = get_object_or_404(
            FiltrationWindow,
            analysis=my_analysis,
            slug=self.kwargs['window_slug']
        )
        return get_object_or_404(
            Bottleneck,
            window=my_window,
            kind=Bottleneck.ONE,
            homology=self.kwargs['homology']
        )

    def get_file(self):
        bottleneck = self.get_object()
        return ContentFile(json.dumps(bottleneck.get_bottleneck_matrix()),
                           name=bottleneck.window.analysis.research.name + '_' +
                           bottleneck.window.analysis.name + '_' +
                           str(bottleneck.window.name) + '_bottleneck_distance_one_to_all_H' + str(bottleneck.homology)
                           + '.csv')


class BottleneckALLDownloadView(VirtualDownloadView):
    def get_object(self):
        my_analysis = get_object_or_404(
                        FiltrationAnalysis,
                        research__slug=self.kwargs['research_slug'],
                        slug=self.kwargs['analysis_slug']
                        )
        return get_object_or_404(
            Bottleneck,
            analysis=my_analysis,
            kind=Bottleneck.ALL,
            homology=self.kwargs['homology']
        )

    def get_file(self):
        bottleneck = self.get_object()
        return ContentFile(json.dumps(bottleneck.get_bottleneck_matrix()), name=bottleneck.analysis.research.name + '_'
                           + bottleneck.analysis.name + '_bottleneck_distance_all_to_all_H' +
                           str(bottleneck.homology) + '.csv')


class BottleneckCONSDownloadView(VirtualDownloadView):
    def get_object(self):
        my_analysis = get_object_or_404(
                        FiltrationAnalysis,
                        research__slug=self.kwargs['research_slug'],
                        slug=self.kwargs['analysis_slug']
                        )
        return get_object_or_404(
            Bottleneck,
            analysis=my_analysis,
            kind=Bottleneck.CONS,
            homology=self.kwargs['homology']
        )

    def get_file(self):
        bottleneck = self.get_object()
        return ContentFile(json.dumps(bottleneck.get_bottleneck_matrix()), name=bottleneck.analysis.research.name + '_'
                           + bottleneck.analysis.name + '_bottleneck_distance_consecutive_H' +
                           str(bottleneck.homology) + '.csv')


def AnalysisBottleneckCreateView(request, research_slug, analysis_slug):
    research = get_object_or_404(Research, slug=research_slug)
    analysis = get_object_or_404(FiltrationAnalysis, research=research, slug=analysis_slug)
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = AnalysisBottleneckCreationForm(analysis.max_homology_dimension, request.POST)
        # check whether it's valid:
        if form.is_valid():
            cleaned_data = form.cleaned_data
            bottleneck_type = cleaned_data.get("bottleneck_type")
            homology = cleaned_data.get("homology")
            if bottleneck_type == Bottleneck.CONS:
                return redirect('analysis:analysis-bottleneck-consecutive', homology=homology,
                                analysis_slug=analysis.slug, research_slug=research.slug)
            elif bottleneck_type == Bottleneck.ALL:
                return redirect('analysis:analysis-bottleneck-alltoall', homology=homology,
                                analysis_slug=analysis.slug, research_slug=research.slug)

    # if a GET (or any other method) we'll create a blank form
    else:
        form = AnalysisBottleneckCreationForm(analysis.max_homology_dimension)

    return render(request, 'analysis/analysis_bottleneck_form.html',
                  {'form': form, 'research': research, 'analysis': analysis})


def WindowBottleneckCreateView(request, research_slug, analysis_slug, window_slug):
    research = get_object_or_404(Research, slug=research_slug)
    analysis = get_object_or_404(FiltrationAnalysis, research=research, slug=analysis_slug)
    window = get_object_or_404(FiltrationWindow, analysis=analysis, slug=window_slug)
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = WindowBottleneckCreationForm(analysis.max_homology_dimension, request.POST)
        # check whether it's valid:
        if form.is_valid():
            cleaned_data = form.cleaned_data
            homology = cleaned_data.get("homology")
            return redirect('analysis:window-bottleneck-onetoall', homology=homology,
                            window_slug=window.slug,
                            analysis_slug=analysis.slug, research_slug=research.slug)

    # if a GET (or any other method) we'll create a blank form
    else:
        form = WindowBottleneckCreationForm(analysis.max_homology_dimension)

    return render(request, 'analysis/window/window_bottleneck_form.html',
                  {'form': form, 'research': research, 'analysis': analysis, 'window': window})


class ALLBottleneckDeleteView(DeleteView):
    model = Bottleneck
    context_object_name = 'bottleneck'
    template_name = "analysis/bottleneck_confirm_delete_ALL.html"

    def get_object(self):
        my_analysis = get_object_or_404(
            FiltrationAnalysis,
            research__slug=self.kwargs['research_slug'],
            slug=self.kwargs['analysis_slug']
            )
        return get_object_or_404(
            Bottleneck,
            analysis=my_analysis,
            homology=self.kwargs['homology'],
            kind=Bottleneck.ALL
        )

    def get_success_url(self):
        return reverse_lazy('analysis:analysis-detail', kwargs={
                'research_slug': self.kwargs['research_slug'],
                'analysis_slug': self.kwargs['analysis_slug']
        })


class CONSBottleneckDeleteView(DeleteView):
    model = Bottleneck
    context_object_name = 'bottleneck'
    template_name = "analysis/bottleneck_confirm_delete_CONS.html"

    def get_object(self):
        my_analysis = get_object_or_404(
            FiltrationAnalysis,
            research__slug=self.kwargs['research_slug'],
            slug=self.kwargs['analysis_slug']
            )
        return get_object_or_404(
            Bottleneck,
            analysis=my_analysis,
            homology=self.kwargs['homology'],
            kind=Bottleneck.CONS
        )

    def get_success_url(self):
        return reverse_lazy('analysis:analysis-detail', kwargs={
                'research_slug': self.kwargs['research_slug'],
                'analysis_slug': self.kwargs['analysis_slug']
        })


class ONEBottleneckDeleteView(DeleteView):
    model = Bottleneck
    context_object_name = 'bottleneck'
    template_name = "analysis/window/bottleneck_confirm_delete_ONE.html"

    def get_object(self):
        my_analysis = get_object_or_404(
            FiltrationAnalysis,
            research__slug=self.kwargs['research_slug'],
            slug=self.kwargs['analysis_slug']
            )
        my_window = get_object_or_404(
            FiltrationWindow,
            analysis=my_analysis,
            slug=self.kwargs['window_slug']
        )
        return get_object_or_404(
            Bottleneck,
            window=my_window,
            homology=self.kwargs['homology'],
            kind=Bottleneck.ONE
        )

    def get_success_url(self):
        return reverse_lazy('analysis:window-detail', kwargs={
                'research_slug': self.kwargs['research_slug'],
                'analysis_slug': self.kwargs['analysis_slug'],
                'window_slug': self.kwargs['window_slug']
        })
