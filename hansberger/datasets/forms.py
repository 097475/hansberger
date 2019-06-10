from django import forms
from .models import Dataset, TextDataset, EDFDataset


def dataset_name_unique_check(name, research):
    return bool(Dataset.objects.filter(
        research__slug=research.slug,
        name=name
    ).first())


class DatasetCreationMixin:
    def __init__(self, *args, **kwargs):
        research = kwargs.pop('research')
        super().__init__(*args, **kwargs)
        self.fields['research'].initial = research

    def clean(self):
        cleaned_data = super().clean()
        name = cleaned_data.get("name")
        research = cleaned_data.get("research")
        if dataset_name_unique_check(name, research):
            self.add_error("name", "A dataset with this name already exists.")
            raise forms.ValidationError("A dataset with this name already exists.")

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
