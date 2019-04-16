from django.views.generic import (
    CreateView,
    DeleteView,
    DetailView,
    ListView,
)
from django.urls import reverse_lazy
from django.shortcuts import get_object_or_404
from .models import Research, Dataset


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

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        datasets = self.object.datasets.all()
        context['datasets'] = datasets
        return context


class ResearchListView(ListView):
    model = Research
    context_object_name = 'research_list'
    paginate_by = 5
    queryset = Research.objects.all()

    template_name = "research/research_list.html"


class DatasetDetailView(DetailView):
    model = Dataset
    context_object_name = 'dataset'
    slug_field = 'slug'
    slug_url_kwarg = 'dataset_slug'

    template_name = "research/datasets/dataset_detail.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['research'] = get_object_or_404(
            Research,
            slug=self.kwargs['research_slug']
        )
        return context

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
        context['research'] = get_object_or_404(
            Research,
            slug=self.kwargs['research_slug']
        )
        return context

    def get_queryset(self):
        datasets = Dataset.objects.filter(
            research__slug=self.kwargs['research_slug']
        )
        return datasets
