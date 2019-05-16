from django.urls import path
from .views import (
    ResearchCreateView,
    ResearchDeleteView,
    ResearchDetailView,
    ResearchListView,
    DatasetCreateView,
    DatasetDeleteView,
    DatasetDetailView,
    DatasetListView,
    DatasetProcessRedirectView,
    TextDatasetProcessFormView,
    EDFDatasetProcessView,
    FiltrationAnalysisCreateView,
    FiltrationAnalysisDetailView,
    MapperAnalysisCreateView,
    MapperAnalysisDetailView,
    TextDownloadView
)


app_name = "research"
urlpatterns = [
    path("", view=ResearchListView.as_view(), name="research-list"),
    path("new/", view=ResearchCreateView.as_view(), name="research-create"),
    path("<slug:research_slug>/", view=ResearchDetailView.as_view(), name="research-detail"),
    path("<slug:research_slug>/delete/", view=ResearchDeleteView.as_view(), name="research-delete"),
    path("<slug:research_slug>/datasets/", view=DatasetListView.as_view(), name="dataset-list"),
    path("<slug:research_slug>/datasets/add/", view=DatasetCreateView.as_view(), name="dataset-create"),
    path("<slug:research_slug>/datasets/<slug:dataset_slug>/", view=DatasetDetailView.as_view(), name="dataset-detail"),
    path("<slug:research_slug>/datasets/<slug:dataset_slug>/process/", view=DatasetProcessRedirectView.as_view(), name="dataset-process-redirect"), # noqa
    path("<slug:research_slug>/datasets/<slug:dataset_slug>/process-text/", view=TextDatasetProcessFormView.as_view(), name="dataset-process-text"), # noqa
    path("<slug:research_slug>/datasets/<slug:dataset_slug>/process-edf/", view=EDFDatasetProcessView.as_view(), name="dataset-process-edf"), # noqa
    path("<slug:research_slug>/datasets/<slug:dataset_slug>/delete/", view=DatasetDeleteView.as_view(), name="dataset-delete"), # noqa
    path("<slug:research_slug>/filtrationanalysis/add/",
         view=FiltrationAnalysisCreateView.as_view(), name="filtrationanalysis-create"),
    path("<slug:research_slug>/filtrationanalysis/<slug:filtrationanalysis_slug>/",
         view=FiltrationAnalysisDetailView.as_view(), name="filtrationanalysis-detail"),
    path("<slug:research_slug>/filtrationanalysis/<slug:filtrationanalysis_slug>/download",
        view=TextDownloadView.as_view(), name="download-view"),
    path("<slug:research_slug>/mapperanalysis/add/",
         view=MapperAnalysisCreateView.as_view(), name="mapperanalysis-create"),
    path("<slug:research_slug>/mapperanalysis/<slug:mapperanalysis_slug>/",
         view=MapperAnalysisDetailView.as_view(), name="mapperanalysis-detail")
]
