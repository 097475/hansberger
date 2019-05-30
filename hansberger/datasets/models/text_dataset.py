from django.db import models
import pandas as pd
import matplotlib.pyplot as plt
import mpld3
from .dataset import Dataset, DatasetKindChoice


class TextDataset(Dataset):
    values_separator_character = models.CharField(max_length=5, default=',', help_text="""separator character of the
                                                  values in the file""")
    identity_column_index = models.PositiveIntegerField(default=0, help_text="""column number that identifies the
                                                        progressive number of rows in the file""")
    header_row_index = models.PositiveIntegerField(default=0, help_text="""row number that identifies the column in
                                                   the file""")

    def save(self, *args, **kwargs):
        self.kind = DatasetKindChoice.TEXT.value
        super().save(*args, **kwargs)
        dataframe = self.dataframe
        self.data = dataframe.values.tolist()
        self.rows = len(dataframe.values)
        self.cols = len(dataframe.values[0])
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