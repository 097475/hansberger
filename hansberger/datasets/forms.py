from django import forms
from .models import TextDataset, EDFDataset


class DatasetCreationMixin:
    def __init__(self, *args, **kwargs):
        research = kwargs.pop('research')
        super().__init__(*args, **kwargs)
        self.fields['research'].initial = research

    class Meta:
        widgets = {'research': forms.HiddenInput}


class TextDatasetCreationForm(DatasetCreationMixin, forms.ModelForm):
    class Meta(DatasetCreationMixin.Meta):
        model = TextDataset
        fields = [
            'name',
            'description',
            'source',
            'research',
            'values_separator_character',
            'identity_column_index',
            'header_row_index',
        ]


class EDFDatasetCreationForm(DatasetCreationMixin, forms.ModelForm):
    class Meta(DatasetCreationMixin.Meta):
        model = EDFDataset
        fields = [
            'name',
            'description',
            'source',
            'research',
        ]
