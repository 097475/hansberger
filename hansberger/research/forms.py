from django import forms
from .models import Dataset


class DatasetCreationForm(forms.ModelForm):
    def __init__(self, research, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['research'].initial = research

    class Meta:
        model = Dataset
        fields = ['name', 'description', 'file', 'research']
        widgets = {'research': forms.HiddenInput}
