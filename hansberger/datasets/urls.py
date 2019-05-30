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
    path('<slug:dataset_slug>/', view=DatasetRedirectView.as_view(), name='dataset-redirect'),
    path('<slug:dataset_slug>/text/', view=TextDatasetDetailView.as_view(), name='text-dataset-detail'),
    path('<slug:dataset_slug>/edf/', view=EDFDatasetDetailView.as_view(), name='edf-dataset-detail'),
    path('<slug:dataset_slug>/delete/', view=DatasetDeleteView.as_view(), name='dataset-delete'),
]
