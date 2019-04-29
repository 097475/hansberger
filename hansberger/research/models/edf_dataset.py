from django.db import models
from ..models import Dataset


class EDFDatasetManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset().filter(
            file_type=Dataset.EDF
        )


class EDFDataset(Dataset):
    objects = EDFDatasetManager()

    class Meta:
        proxy = True

    def process_file(self):
        raise(NotImplementedError)
