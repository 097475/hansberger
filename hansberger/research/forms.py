from django import forms
from .models import Research


def research_name_unique_check(name):
    return bool(Research.objects.filter(name=name).first())


class ResearchCreationForm(forms.ModelForm):
    def clean(self):
        cleaned_data = super(ResearchCreationForm, self).clean()
        name = cleaned_data.get("name")
        if research_name_unique_check(name):
            self.add_error("name", "A research with this name already exists.")
            raise forms.ValidationError("A research with this name already exists.")

    class Meta:
        model = Research
        fields = ['name', 'description']
