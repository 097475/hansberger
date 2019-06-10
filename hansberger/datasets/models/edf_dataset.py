from .dataset import Dataset, DatasetKindChoice


class EDFDataset(Dataset):

    def save(self, *args, **kwargs):
        self.kind = DatasetKindChoice.EDF.value
        super().save(*args, **kwargs)
