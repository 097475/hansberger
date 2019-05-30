from django.views.generic import (
    CreateView,
    DeleteView,
    DetailView,
    ListView,
)
from django.urls import reverse_lazy
from .models import (
    Research
)
from .forms import (
    ResearchCreationForm
)


class ResearchCreateView(CreateView):
    model = Research
    form_class = ResearchCreationForm
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
