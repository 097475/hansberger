from django.views.generic import (
    CreateView,
)
from .models import Research


class ResearchCreateView(CreateView):
    model = Research
    fields = ['name', 'description']
