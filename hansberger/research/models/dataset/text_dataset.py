import os.path
from django.conf import settings
from django.db import models
import matplotlib.pyplot as plt
import pandas as pd
from .dataset import Dataset


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
        dataframe = self.__get_dataframe_from_text(identity_column_index, values_separator, header_row_index)
        self.__save_dataframe_plot(dataframe)
        self.__save_dataframe_matrix(dataframe)
        self.save()

    def __get_dataframe_from_text(self, identity_column_index, values_separator, header_row_index):
        return pd.read_csv(
            self.file.path,
            index_col=identity_column_index,
            sep=values_separator,
            header=header_row_index,
        )

    def __save_dataframe_plot(self, dataframe):
        dataframe.plot()
        plot_filename = self.slug + '_plot.svg'
        relative_plot_dir = os.path.join('research', self.research.slug, 'datasets', self.slug)
        absolute_plot_dir = os.path.join(settings.MEDIA_ROOT, relative_plot_dir)
        if not os.path.exists(absolute_plot_dir):
            os.makedirs(absolute_plot_dir)
        plt.savefig(os.path.join(absolute_plot_dir, plot_filename))
        plt.clf()
        self.plot = os.path.join(relative_plot_dir, plot_filename)

    def __save_dataframe_matrix(self, dataframe):
        self.matrix = dataframe.values.tolist()
