from django.urls import path
from .views import (
    ResearchCreateView,
    ResearchDeleteView,
    ResearchDetailView,
    ResearchListView,
    DatasetDetailView,
    DatasetListView,
)


app_name = "research"
urlpatterns = [
    path("", view=ResearchListView.as_view(), name="research-list"),
    path("new/", view=ResearchCreateView.as_view(), name="research-create"),
    path("<slug:research_slug>/", view=ResearchDetailView.as_view(), name="research-detail"),
    path("<slug:research_slug>/delete/", view=ResearchDeleteView.as_view(), name="research-delete"),
    path("<slug:research_slug>/datasets/", view=DatasetListView.as_view(), name="dataset-list"),
    path("<slug:research_slug>/datasets/<slug:dataset_slug>/", view=DatasetDetailView.as_view(), name="dataset-detail"),
]
