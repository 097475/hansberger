from django.urls import path
from .views import (
    ResearchCreateView,
    ResearchDeleteView,
    ResearchDetailView,
    ResearchListView,
)


app_name = "research"
urlpatterns = [
    path("", view=ResearchListView.as_view(), name="research-list"),
    path("new/", view=ResearchCreateView.as_view(), name="research-create"),
    path("<slug:research_slug>/", view=ResearchDetailView.as_view(), name="research-detail"),
    path("<slug:research_slug>/delete/", view=ResearchDeleteView.as_view(), name="research-delete"),
]
