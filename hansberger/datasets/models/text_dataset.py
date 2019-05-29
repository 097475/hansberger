from django.db import models
import pandas as pd
import matplotlib.pyplot as plt
import mpld3
from .dataset import Dataset, DatasetKindChoice


class TextDataset(Dataset):
    values_separator_character = models.CharField(max_length=5)
    identity_column_index = models.IntegerField()
    header_row_index = models.IntegerField()

    def save(self, *args, **kwargs):
        self.kind = DatasetKindChoice.TEXT.value
        super().save(*args, **kwargs)

    @property
    def dataframe(self):
        return pd.read_csv(
            self.source.path,
            index_col=self.identity_column_index,
            sep=self.values_separator_character,
            header=self.header_row_index,
        )

    @property
    def plot(self):
        self.dataframe.plot()
        figure = plt.gcf()
        html_figure = mpld3.fig_to_html(figure, template_type='general')
        plt.clf()
        return html_figure
