from django.views.generic import (
    CreateView,
    DeleteView,
)
from django.urls import reverse_lazy
from .models import Research


class ResearchCreateView(CreateView):
    model = Research
    fields = ['name', 'description']


class ResearchDeleteView(DeleteView):
    model = Research
    context_object_name = 'research'
    slug_field = 'slug'
    slug_url_kwarg = 'research_slug'
    success_url = reverse_lazy('research: research-list')
