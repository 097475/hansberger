from django.db import models


class Source(models.Model):

    class Meta:
        abstract = True


class EdfSource(Source):
    file = models.FileField(upload_to='research/datasets/sources/edf')


class TextSource(Source):
    file = models.FileField(upload_to='research/datasets/sources/text')
    values_separator_character = models.CharField(
        max_length=5,
        verbose_name='delimiter character of the values in the file'
    )
    identity_column_index = models.IntegerField(
        verbose_name='column number that identifies the progressive number of rows in the file'
    )
    header_row_index = models.IntegerField(
        verbose_name='row number that identifies the column in the file'
    )
