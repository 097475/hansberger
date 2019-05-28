from django.urls import path
from .views import (
    TextDatasetCreateView,
    TextDatasetDetailView,
    DatasetListView,
    DatasetRedirectView,
    DatasetDeleteView,
)

app_name = 'datasets'
urlpatterns = [
    path('', view=DatasetListView.as_view(), name='dataset-list'),
    path('detail/<slug:dataset_slug>/', view=DatasetRedirectView.as_view(), name='dataset-redirect'),
    path('detail/text/<slug:dataset_slug>/', view=TextDatasetDetailView.as_view(), name='text-dataset-detail'),
    path('delete/<slug:dataset_slug>/', view=DatasetDeleteView.as_view(), name='dataset-delete'),
    path('create/text/', view=TextDatasetCreateView.as_view(), name='text-dataset-create'),
]
