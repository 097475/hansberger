from django.views.generic import (
    CreateView,
    DetailView,
)
from django.shortcuts import get_object_or_404
from django.urls import reverse_lazy
from .models import TextDataset
from research.models import Research
from .forms import TextDatasetCreationForm


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
        return reverse_lazy('datasets:text-dataset-detail', kwargs={
            'research_slug': self.kwargs['research_slug'],
            'dataset_slug': self.dataset.slug
        })


class TextDatasetDetailView(DetailView):
    model = TextDataset
    context_object_name = 'dataset'
    template_name = "datasets/dataset_detail.html"

    def get_object(self):
        return get_object_or_404(
            TextDataset,
            research__slug=self.kwargs['research_slug'],
            slug=self.kwargs['dataset_slug']
        )
