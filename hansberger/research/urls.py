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
    FiltrationAnalysisCreateView,
    AnalysisDetailView,
    TextDownloadView,
    MapperAnalysisCreateView,
    MapperAnalysisView,
    AnalysisListView,
    WindowDetailView,
    WindowListView,
    AnalysisDeleteView
)


app_name = 'research'
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
    path("<slug:research_slug>/datasets/<slug:dataset_slug>/delete/", view=DatasetDeleteView.as_view(), name="dataset-delete"), # noqa
    path("<slug:research_slug>/analysis/filtrationanalysis/add/", view=FiltrationAnalysisCreateView.as_view(), name="filtrationanalysis-create"), # noqa
    path("<slug:research_slug>/analysis/mapperanalysis/add/", view=MapperAnalysisCreateView.as_view(), name="mapperanalysis-create"), # noqa
    path("<slug:research_slug>/analysis/", view=AnalysisListView.as_view(), name="analysis-list"),
    path("<slug:research_slug>/analysis/<slug:analysis_slug>/", view=AnalysisDetailView.as_view(), name="analysis-detail"), # noqa 
    path("<slug:research_slug>/analysis/<slug:analysis_slug>/delete/", view=AnalysisDeleteView.as_view(), name="analysis-delete"), # noqa 
    path("<slug:research_slug>/analysis/<slug:analysis_slug>/windows/", view=WindowListView.as_view(), name="window-list"), # noqa
    path("<slug:research_slug>/analysis/<slug:analysis_slug>/windows/<slug:window_slug>/", view=WindowDetailView.as_view(), name="window-detail"), # noqa    
    path("<slug:research_slug>/analysis/<slug:analysis_slug>/windows/<slug:window_slug>/graph/", view=MapperAnalysisView.as_view(), name="mapperanalysis-graph"), # noqa
    path("<slug:research_slug>/analysis/<slug:analysis_slug>/windows/<slug:window_slug>/download/", view=TextDownloadView.as_view(), name="download-view") # noqa
]
