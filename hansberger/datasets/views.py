from django.views.generic import (
    CreateView,
    DetailView,
    ListView,
    DeleteView,
)
from django.shortcuts import get_object_or_404
from django.urls import reverse_lazy
from .models import Dataset, TextDataset, EDFDataset
from research.models import Research
from .forms import TextDatasetCreationForm, EDFDatasetCreationForm


class DatasetCreateMixin:
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['research'] = get_object_or_404(
            Research,
            slug=self.kwargs['research_slug']
        )
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


class TextDatasetCreateView(DatasetCreateMixin, CreateView):
    model = TextDataset
    form_class = TextDatasetCreationForm

    def get_success_url(self):
        return reverse_lazy('datasets:dataset-detail', kwargs={
            'research_slug': self.kwargs['research_slug'],
            'dataset_slug': self.dataset.slug
        })


class EDFDatasetCreateView(DatasetCreateMixin, CreateView):
    model = EDFDataset
    form_class = EDFDatasetCreationForm

    def get_success_url(self):
        return reverse_lazy('datasets:dataset-detail', kwargs={
            'research_slug': self.kwargs['research_slug'],
            'dataset_slug': self.dataset.slug
        })


class DatasetDetailView(DetailView):
    model = Dataset
    context_object_name = 'dataset'
    template_name = "datasets/dataset_detail.html"

    def get_object(self):
        my_dataset = (TextDataset.objects.filter(
            research__slug=self.kwargs['research_slug'],
            slug=self.kwargs['dataset_slug']
            ).first()
            or
            EDFDataset.objects.filter(
                research__slug=self.kwargs['research_slug'],
                slug=self.kwargs['dataset_slug']
            ).first())
        return my_dataset


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


class DatasetDeleteView(DeleteView):
    model = Dataset
    context_object_name = 'dataset'

    def get_object(self):
        return get_object_or_404(
            Dataset,
            research__slug=self.kwargs['research_slug'],
            slug=self.kwargs['dataset_slug']
        )

    def get_success_url(self):
        return reverse_lazy(
            'datasets:dataset-list',
            kwargs={'research_slug': self.kwargs['research_slug']}
        )
