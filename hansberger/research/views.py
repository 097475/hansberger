from django.views.generic import (
    CreateView,
    DeleteView,
    DetailView,
    ListView,
    RedirectView,
    FormView,
)
from django.urls import reverse_lazy
from django.shortcuts import get_object_or_404
from django_downloadview import VirtualDownloadView
from django.core.files.base import ContentFile
from .models import Research, Dataset, TextDataset, FiltrationAnalysis
from .forms import DatasetCreationForm, TextDatasetProcessForm, FiltrationAnalysisCreationForm


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


class FiltrationAnalysisDetailView(DetailView):
    model = FiltrationAnalysis
    context_object_name = 'analysis'
    template_name = "research/analysis/filtrationanalysis_detail.html"

    def get_object(self):
        return get_object_or_404(
            FiltrationAnalysis,
            research__slug=self.kwargs['research_slug'],
            slug=self.kwargs['filtrationanalysis_slug']
        )


class FiltrationAnalysisCreateView(CreateView):
    model = FiltrationAnalysis
    form_class = FiltrationAnalysisCreationForm
    template_name = "research/analysis/filtrationanalysis_form.html"

    def get_success_url(self):
        return reverse_lazy('research:filtrationanalysis-detail', kwargs={
                'research_slug': self.kwargs['research_slug'],
                'filtrationanalysis_slug': self.filtrationanalysis.slug
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
        self.filtrationanalysis = form.save(commit=False)
        return super().form_valid(form)


class TextDownloadView(VirtualDownloadView):
    model = FiltrationAnalysis

    def get_object(self):
        return get_object_or_404(
            FiltrationAnalysis,
            research__slug=self.kwargs['research_slug'],
            slug=self.kwargs['filtrationanalysis_slug']
        )

    def get_file(self):
        analysis = self.get_object()
        return ContentFile(analysis.result_matrix, name=analysis.research.name + '_' + analysis.name + '.dat')
