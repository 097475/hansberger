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
    WindowBottleneckView,
    AnalysisConsecutiveBottleneckView,
    AnalysisAlltoallBottleneckView,
    SourceChoice,
    RipserDownloadView,
    EntropyDownloadView,
    BottleneckALLDownloadView,
    BottleneckCONSDownloadView,
    BottleneckONEDownloadView,
    AnalysisBottleneckCreateView,
    WindowBottleneckCreateView,
    ONEBottleneckDeleteView,
    CONSBottleneckDeleteView,
    ALLBottleneckDeleteView
)

app_name = 'analysis'
urlpatterns = [
    path("", view=AnalysisListView.as_view(), name="analysis-list"),
    path("add/", view=SourceChoice, name="analysis-source-choice"),
    path("add/<form>/filtrationanalysis/", view=FiltrationAnalysisCreateView.as_view(), name="filtrationanalysis-create"), # noqa
    path("add/<form>/mapperanalysis/", view=MapperAnalysisCreateView.as_view(), name="mapperanalysis-create"), # noqa
    path("<slug:analysis_slug>/", view=AnalysisDetailView.as_view(), name="analysis-detail"), # noqa 
    path("<slug:analysis_slug>/delete/", view=AnalysisDeleteView.as_view(), name="analysis-delete"), # noqa 
    path("<slug:analysis_slug>/windows/", view=WindowListView.as_view(), name="window-list"), # noqa
    path("<slug:analysis_slug>/windows/<slug:window_slug>/", view=WindowDetailView.as_view(), name="window-detail"), # noqa   
    path("<slug:analysis_slug>/windows/<slug:window_slug>/bottleneck/", view=WindowBottleneckCreateView, name="window-bottleneck-create"), # noqa 
    path("<slug:analysis_slug>/windows/<slug:window_slug>/graph/", view=MapperAnalysisView.as_view(), name="mapperanalysis-graph"), # noqa
    path("<slug:analysis_slug>/windows/<slug:window_slug>/<int:homology>/bottleneck_onetoall/", view=WindowBottleneckView.as_view(), name="window-bottleneck-onetoall"), # noqa 
    path("<slug:analysis_slug>/bottleneck/", view=AnalysisBottleneckCreateView, name="analysis-bottleneck-create"), # noqa          
    path("<slug:analysis_slug>/<int:homology>/bottleneck_consecutive/", view=AnalysisConsecutiveBottleneckView.as_view(), name="analysis-bottleneck-consecutive"), # noqa      
    path("<slug:analysis_slug>/<int:homology>/bottleneck_alltoall/", view=AnalysisAlltoallBottleneckView.as_view(), name="analysis-bottleneck-alltoall"), # noqa  
    path("<slug:analysis_slug>/windows/<slug:window_slug>/ripser_download/", view=RipserDownloadView.as_view(), name="ripser-download"), # noqa 
    path("<slug:analysis_slug>/entropy_download/", view=EntropyDownloadView.as_view(), name="entropy-download"), # noqa    
    path("<slug:analysis_slug>/<int:homology>/bottleneck_alltoall/bottleneck_ALL_download/", view=BottleneckALLDownloadView.as_view(), name="bottleneck-ALL-download"), # noqa
    path("<slug:analysis_slug>/<int:homology>/bottleneck_consecutive/bottleneck_CONS_download/", view=BottleneckCONSDownloadView.as_view(), name="bottleneck-CONS-download"), # noqa    
    path("<slug:analysis_slug>/windows/<slug:window_slug>/<int:homology>/bottleneck_onetoall/bottleneck_ONE_download/", view=BottleneckONEDownloadView.as_view(), name="bottleneck-ONE-download"), # noqa
    path("<slug:analysis_slug>/windows/<slug:window_slug>/<int:homology>/bottleneck_onetoall/delete/", view=ONEBottleneckDeleteView.as_view(), name="bottleneck-delete-ONE"), # noqa  
    path("<slug:analysis_slug>/<int:homology>/bottleneck_consecutive/delete/", view=CONSBottleneckDeleteView.as_view(), name="bottleneck-delete-CONS"), # noqa      
    path("<slug:analysis_slug>/<int:homology>/bottleneck_alltoall/delete/", view=ALLBottleneckDeleteView.as_view(), name="bottleneck-delete-ALL"), # noqa  
]
