from django import forms
from .models import Dataset, FiltrationAnalysis


class DatasetCreationForm(forms.ModelForm):
    def __init__(self, research, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['research'].initial = research

    class Meta:
        model = Dataset
        fields = ['name', 'file_type', 'description', 'file', 'research']
        widgets = {'research': forms.HiddenInput}


class TextDatasetProcessForm(forms.Form):

    values_separator_character = forms.CharField(
        max_length=5,
        required=True,
        label="separator character of the values in the file",
        initial=','
    )
    identity_column_index = forms.IntegerField(
        widget=forms.NumberInput,
        required=True,
        label="column number that identifies the progressive number of rows in the file",
        initial=0
    )
    header_row_index = forms.IntegerField(
        widget=forms.NumberInput,
        required=True,
        label="row number that identifies the column in the file",
        initial=0
    )


class FiltrationAnalysisCreationForm(forms.ModelForm):
    def __init__(self, research, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['research'].initial = research

    class Meta:
        model = FiltrationAnalysis
        exclude = ['slug', 'result_matrix', 'result_plot', 'result_entropy']
        widgets = {'research': forms.HiddenInput}
