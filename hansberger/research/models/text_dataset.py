from django.db import models
from ..models import Dataset


class TextDatasetManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset().filter(
            file_type=Dataset.TEXT
        )


class TextDataset(Dataset):
    objects = TextDatasetManager()

    class Meta:
        proxy = True

    def process_file(self, values_separator, header_row_index, identity_column_index):
        raise(NotImplementedError)
