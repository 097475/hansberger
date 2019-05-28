from itertools import chain
from django.views.generic import (
    View,
    CreateView,
    DeleteView,
    DetailView,
    ListView,
    RedirectView,
    FormView,
)
from django.http import HttpResponse
from django.urls import reverse_lazy
from django.shortcuts import get_object_or_404
from django_downloadview import VirtualDownloadView
from django.core.files.base import ContentFile
from django.shortcuts import redirect
from .models import (
    Research,
    Dataset,
    TextDataset,
    FiltrationAnalysis,
    MapperAnalysis,
    Analysis,
    FiltrationWindow,
    MapperWindow,
    Window
)
from .forms import (
    DatasetCreationForm,
    TextDatasetProcessForm,
    FiltrationAnalysisCreationForm,
    MapperAnalysisCreationForm
)


class ResearchCreateView(CreateView):
    model = Research
    fields = ['name', 'description']
    template_name = "research/research_form.html"


class ResearchDeleteView(DeleteView):
    model = Research
    context_object_name = 'research'
    slug_field = 'slug'
    slug_url_kwarg = 'research_slug'
    success_url = reverse_lazy('research:research-list')
    template_name = "research/research_confirm_delete.html"


class ResearchDetailView(DetailView):
    model = Research
    context_object_name = 'research'
    slug_field = 'slug'
    slug_url_kwarg = 'research_slug'
    template_name = "research/research_detail.html"


class ResearchListView(ListView):
    model = Research
    context_object_name = 'research_list'
    queryset = Research.objects.all().values('name', 'creation_date', 'slug')
    template_name = "research/research_list.html"


class DatasetCreateView(CreateView):
    model = Dataset
    form_class = DatasetCreationForm
    template_name = "research/datasets/dataset_form.html"

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
        self.dataset = form.save()
        return super().form_valid(form)

    def get_success_url(self):
        return reverse_lazy('research:dataset-process-redirect', kwargs={
            'research_slug': self.kwargs['research_slug'],
            'dataset_slug': self.dataset.slug
        })


class DatasetProcessRedirectView(RedirectView):

    process_routes = {
        Dataset.TEXT: 'research:dataset-process-text',
    }

    @property
    def dataset_process_route(self):
        dataset = get_object_or_404(
            Dataset,
            research__slug=self.kwargs['research_slug'],
            slug=self.kwargs['dataset_slug']
        )
        return self.process_routes.get(dataset.source_type)

    def get_redirect_url(self, **kwargs):
        return reverse_lazy(self.dataset_process_route, kwargs={
            'research_slug': kwargs['research_slug'],
            'dataset_slug': kwargs['dataset_slug']
        })


class TextDatasetProcessFormView(FormView):
    form_class = TextDatasetProcessForm
    template_name = 'research/datasets/dataset_process_form.html'

    def get_success_url(self):
        return reverse_lazy('research:dataset-detail', kwargs={
                'research_slug': self.kwargs['research_slug'],
                'dataset_slug': self.kwargs['dataset_slug'],
        })

    def form_valid(self, form):
        dataset = get_object_or_404(
            TextDataset,
            research__slug=self.kwargs['research_slug'],
            slug=self.kwargs['dataset_slug'],
        )
        dataset.process_source_and_save_information(
            form.cleaned_data.get('values_separator_character'),
            form.cleaned_data.get('identity_column_index'),
            form.cleaned_data.get('header_row_index'),
        )
        return super().form_valid(form)


class DatasetDeleteView(DeleteView):
    model = Dataset
    context_object_name = 'dataset'
    template_name = "research/datasets/dataset_confirm_delete.html"

    def get_object(self):
        return get_object_or_404(
            Dataset,
            research__slug=self.kwargs['research_slug'],
            slug=self.kwargs['dataset_slug']
        )

    def get_success_url(self):
        return reverse_lazy('research:dataset-list', kwargs={
                'research_slug': self.kwargs['research_slug']
        })


class DatasetDetailView(DetailView):
    model = Dataset
    context_object_name = 'dataset'
    template_name = "research/datasets/dataset_detail.html"

    def get_object(self):
        return get_object_or_404(
            Dataset,
            research__slug=self.kwargs['research_slug'],
            slug=self.kwargs['dataset_slug']
        )


class DatasetListView(ListView):
    model = Dataset
    context_object_name = 'datasets'
    paginate_by = 10
    template_name = "research/datasets/dataset_list.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['research'] = self.research
        return context

    def get_queryset(self):
        self.research = get_object_or_404(
            Research,
            slug=self.kwargs['research_slug']
        )
        datasets = Dataset.objects.filter(
            research=self.research
        ).only('name', 'creation_date', 'slug', 'research')
        return datasets


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
        if type(my_analysis) is FiltrationAnalysis:
            windows = FiltrationWindow.objects.filter(
                      analysis=my_analysis
                      )
        elif type(my_analysis) is MapperAnalysis:
            windows = MapperWindow.objects.filter(
                      analysis=my_analysis
                      )
        if windows.count() > 1:
            return redirect('research:window-list',
                            permanent=False,
                            analysis_slug=self.kwargs['analysis_slug'],
                            research_slug=self.kwargs['research_slug']
                            )
        else:
            return redirect('research:window-detail',
                            permanent=False,
                            analysis_slug=self.kwargs['analysis_slug'],
                            research_slug=self.kwargs['research_slug'],
                            window_slug=windows.get().slug
                            )


class FiltrationAnalysisCreateView(CreateView):
    model = FiltrationAnalysis
    form_class = FiltrationAnalysisCreationForm
    template_name = "research/analysis/filtrationanalysis_form.html"

    def get_success_url(self):
        return reverse_lazy('research:analysis-detail', kwargs={
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
        return super().form_valid(form)


class TextDownloadView(VirtualDownloadView):
    model = FiltrationWindow

    def get_object(self):
        return get_object_or_404(
            FiltrationWindow,
            analysis__slug=self.kwargs['analysis_slug'],
            slug=self.kwargs['window_slug']
        )

    def get_file(self):
        window_analysis = self.get_object()
        return ContentFile(window_analysis.result_matrix, name=window_analysis.analysis.research.name + '_' +
                           window_analysis.analysis.name + '_' + window_analysis.name + '.dat')


class MapperAnalysisCreateView(CreateView):
    model = MapperAnalysis
    form_class = MapperAnalysisCreationForm
    template_name = "research/analysis/mapperanalysis_form.html"

    def get_success_url(self):
        return reverse_lazy('research:analysis-detail', kwargs={
                'research_slug': self.kwargs['research_slug'],
                'analysis_slug': self.mapperanalysis.slug
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
        self.mapperanalysis = form.save(commit=False)
        return super().form_valid(form)


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


class AnalysisListView(ListView):
    model = Analysis
    context_object_name = 'analyses'
    paginate_by = 10
    template_name = "research/analysis/analysis_list.html"

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
        return list(chain(filtration_analyses, mapper_analyses))


class WindowDetailView(DetailView):
    model = Window
    context_object_name = 'window'
    template_name = "research/window/window_detail.html"

    def get_object(self):
        return (FiltrationWindow.objects.filter(
            analysis__slug=self.kwargs['analysis_slug'],
            slug=self.kwargs['window_slug']
            ).first()
            or
            MapperWindow.objects.filter(
            analysis__slug=self.kwargs['analysis_slug'],
            slug=self.kwargs['window_slug']
            ).first())


class WindowListView(ListView):
    model = Window
    context_object_name = 'windows'
    paginate_by = 10
    template_name = "research/window/window_list.html"

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


class AnalysisDeleteView(DeleteView):
    model = Analysis
    context_object_name = 'analysis'
    template_name = "research/analysis/analysis_confirm_delete.html"

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
        return reverse_lazy('research:analysis-list', kwargs={
                'research_slug': self.kwargs['research_slug']
        })
