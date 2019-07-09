from django.urls import path
from .views import (
    TextDatasetCreateView,
    EDFDatasetCreateView,
    DatasetListView,
    DatasetDeleteView,
    DatasetDetailView
)

app_name = 'datasets'
urlpatterns = [
    path('', view=DatasetListView.as_view(), name='dataset-list'),
    path('create/text/', view=TextDatasetCreateView.as_view(), name='text-dataset-create'),
    path('create/edf/', view=EDFDatasetCreateView.as_view(), name='edf-dataset-create'),
    path('<slug:dataset_slug>/', view=DatasetDetailView.as_view(), name='dataset-detail'),
    path('<slug:dataset_slug>/delete/', view=DatasetDeleteView.as_view(), name='dataset-delete'),
]
