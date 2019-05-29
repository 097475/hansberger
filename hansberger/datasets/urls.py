from django.urls import path
from .views import (
    TextDatasetCreateView,
    TextDatasetDetailView,
    EDFDatasetCreateView,
    EDFDatasetDetailView,
    DatasetListView,
    DatasetRedirectView,
    DatasetDeleteView,
)

app_name = 'datasets'
urlpatterns = [
    path('', view=DatasetListView.as_view(), name='dataset-list'),
    path('create/text/', view=TextDatasetCreateView.as_view(), name='text-dataset-create'),
    path('create/edf/', view=EDFDatasetCreateView.as_view(), name='edf-dataset-create'),
    path('detail/<slug:dataset_slug>/', view=DatasetRedirectView.as_view(), name='dataset-redirect'),
    path('detail/text/<slug:dataset_slug>/', view=TextDatasetDetailView.as_view(), name='text-dataset-detail'),
    path('detail/edf/<slug:dataset_slug>/', view=EDFDatasetDetailView.as_view(), name='edf-dataset-detail'),
    path('delete/<slug:dataset_slug>/', view=DatasetDeleteView.as_view(), name='dataset-delete'),
]
