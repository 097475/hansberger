from django.urls import path
from .views import (
    AnalysisListView,
    FiltrationAnalysisCreateView,
    MapperAnalysisCreateView,
    AnalysisDetailView,
    AnalysisDeleteView,
    WindowListView,
    WindowDetailView,
    MapperAnalysisView,
    TextDownloadView
)

app_name = 'analysis'
urlpatterns = [
    path("", view=AnalysisListView.as_view(), name="analysis-list"),
    path("filtrationanalysis/add/", view=FiltrationAnalysisCreateView.as_view(), name="filtrationanalysis-create"), # noqa
    path("mapperanalysis/add/", view=MapperAnalysisCreateView.as_view(), name="mapperanalysis-create"), # noqa
    path("<slug:analysis_slug>/", view=AnalysisDetailView.as_view(), name="analysis-detail"), # noqa 
    path("<slug:analysis_slug>/delete/", view=AnalysisDeleteView.as_view(), name="analysis-delete"), # noqa 
    path("<slug:analysis_slug>/windows/", view=WindowListView.as_view(), name="window-list"), # noqa
    path("<slug:analysis_slug>/windows/<slug:window_slug>/", view=WindowDetailView.as_view(), name="window-detail"), # noqa    
    path("<slug:analysis_slug>/windows/<slug:window_slug>/graph/", view=MapperAnalysisView.as_view(), name="mapperanalysis-graph"), # noqa
    path("<slug:analysis_slug>/windows/<slug:window_slug>/download/", view=TextDownloadView.as_view(), name="download-view") # noqa
]
