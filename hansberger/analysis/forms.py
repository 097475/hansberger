from django import forms
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Fieldset, Field, Div
from django.urls import reverse_lazy
from .models import FiltrationAnalysis, MapperAnalysis, Bottleneck
from datasets.models import Dataset, DatasetKindChoice


def analysis_name_unique_check(name, research):
    return bool((FiltrationAnalysis.objects.filter(
        research__slug=research.slug,
        name=name
        ).first()
        or
        MapperAnalysis.objects.filter(
        research__slug=research.slug,
        name=name
        ).first()))


class SourceChoiceForm(forms.Form):
    ANALYSIS_OPTIONS = [('filtration_analysis', "Filtration Analysis with ripser"),
                        ('mapper_analysis', "Mapper Analysis with KeplerMapper")]
    SOURCE_OPTIONS = [('dataset', 'Dataset'), ('precomputed', 'Precomputed distance matrices')]
    analysis = forms.ChoiceField(widget=forms.RadioSelect, choices=ANALYSIS_OPTIONS)
    source = forms.ChoiceField(widget=forms.RadioSelect, choices=SOURCE_OPTIONS)


class DatasetAnalysisCreationForm(forms.ModelForm):
    def window_overlap_checks(self, window_size, window_overlap, dataset):
        if dataset.kind == DatasetKindChoice.EDF.value:
            dataset = dataset.edfdataset
        elif dataset.kind == DatasetKindChoice.TEXT.value:
            dataset = dataset.textdataset
        if window_size == 0:
            self.add_error("window_size", "Window size can't be equal to 0")
            raise forms.ValidationError("Window size can't be equal to 0")
        if window_size > len(dataset.get_matrix_data()[0]):
            self.add_error("window_size", "Window size can't be greater than the number of columns in the dataset")
            raise forms.ValidationError("Window size can't be greater than the number of columns in the dataset")
        if window_overlap >= window_size:
            self.add_error("window_overlap", "Window overlap can't be greater than or equal to window size")
            raise forms.ValidationError("Window overlap can't be greater than or equal to window size")


class PrecomputedAnalysisCreationForm(forms.ModelForm):
    precomputed_distance_matrix = forms.FileField(required=False)


class FiltrationAnalysisCreationForm_Dataset(DatasetAnalysisCreationForm):
    def __init__(self, research, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['research'].initial = research
        self.fields['dataset'].queryset = Dataset.objects.filter(research__slug=research.slug)
        self.fields['dataset'].required = True
        self.helper = FormHelper(self)
        self.helper.form_method = 'POST'
        self.helper.form_action = reverse_lazy('analysis:filtrationanalysis-create', kwargs={
                                 'form': 'dataset',
                                 'research_slug': research.slug})
        self.helper.form_id = "analysis_form"
        self.helper.layout = Layout(
            'name',
            'description',
            Field('research', type="hidden"),
            'dataset',
            Div(id='peek_dataset'),
            'window_size',
            'window_overlap',
            'filtration_type',
            'distance_matrix_metric',
            Fieldset(
                'Ripser arguments',
                'max_homology_dimension',
                'max_distances_considered',
                'coeff',
                'do_cocycles',
                'n_perm'
            )
        )

    def clean(self):
        cleaned_data = super().clean()
        dataset = cleaned_data.get("dataset")
        filtration_type = cleaned_data.get("filtration_type")
        distance_matrix_metric = cleaned_data.get("distance_matrix_metric")
        window_overlap = cleaned_data.get("window_overlap")
        window_size = cleaned_data.get("window_size")
        name = cleaned_data.get("name")
        research = cleaned_data.get("research")
        if analysis_name_unique_check(name, research):
            self.add_error("name", "An analysis with this name already exists.")
            raise forms.ValidationError("An analysis with this name already exists.")
        if window_size is not None:
            self.window_overlap_checks(window_size, window_overlap, dataset)
        if filtration_type == FiltrationAnalysis.VIETORIS_RIPS_FILTRATION and distance_matrix_metric == '':
            raise forms.ValidationError("You must provide a distance matrix metric for a Vietoris-Rips Filtration")
            self.add_error("distance_matrix_metric",
                           "You must provide a distance matrix metric for a Vietoris-Rips Filtration")

    class Meta:
        model = FiltrationAnalysis
        exclude = ['slug', 'precomputed_distance_matrix', 'precomputed_distance_matrix_json']


class FiltrationAnalysisCreationForm_Precomputed(PrecomputedAnalysisCreationForm):
    def __init__(self, research, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['research'].initial = research
        self.helper = FormHelper(self)
        self.helper.form_method = 'POST'
        self.helper.form_action = reverse_lazy('analysis:filtrationanalysis-create', kwargs={
                                 'form': 'precomputed',
                                 'research_slug': research.slug})
        self.helper.form_id = "analysis_form"
        self.helper.layout = Layout(
            'name',
            'description',
            Field('research', type="hidden"),
            Field('precomputed_distance_matrix', multiple=True),
            Field('precomputed_distance_matrix_json', type="hidden"),
            Fieldset(
                'Ripser arguments',
                'max_homology_dimension',
                'max_distances_considered',
                'coeff',
                'do_cocycles',
                'n_perm'
            )
        )

    def clean(self):
        cleaned_data = super().clean()
        precomputed_distance_matrix = cleaned_data.get("precomputed_distance_matrix")
        name = cleaned_data.get("name")
        research = cleaned_data.get("research")
        if analysis_name_unique_check(name, research):
            self.add_error("name", "An analysis with this name already exists.")
            raise forms.ValidationError("An analysis with this name already exists.")
        if not precomputed_distance_matrix:
            self.add_error("precomputed_distance_matrix", "You must provide a precomputed distance matrix")
            raise forms.ValidationError("You must provide a precomputed distance matrix")

    class Meta:
        model = FiltrationAnalysis
        exclude = ['slug', 'dataset', 'window_size', 'window_overlap', 'filtration_type', 'distance_matrix_metric']


class MapperAnalysisCreationForm_Dataset(DatasetAnalysisCreationForm):
    def __init__(self, research, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['research'].initial = research
        self.fields['dataset'].queryset = Dataset.objects.filter(research__slug=research.slug)
        self.fields['dataset'].required = True
        self.helper = FormHelper(self)
        self.helper.form_method = 'POST'
        self.helper.form_action = reverse_lazy('analysis:mapperanalysis-create', kwargs={
                                 'form': 'dataset',
                                 'research_slug': research.slug})
        self.helper.form_id = "analysis_form"
        self.helper.layout = Layout(
            'name',
            'description',
            Field('research', type="hidden"),
            'dataset',
            Div(id='peek_dataset'),
            'window_size',
            'window_overlap',
            'distance_matrix_metric',
            Fieldset(
                'fit_transform parameters',
                'projection',
                'knn_n_value',
                'scaler',
            ),
            Fieldset(
                'map parameters',
                'use_original_data',
                'clusterer',
                'cover_n_cubes',
                'cover_perc_overlap',
                'graph_nerve_min_intersection',
                'precomputed',
                'remove_duplicate_nodes'
            )
        )

    def clean(self):
        cleaned_data = super().clean()
        dataset = cleaned_data.get("dataset")
        window_overlap = cleaned_data.get("window_overlap")
        window_size = cleaned_data.get("window_size")
        name = cleaned_data.get("name")
        research = cleaned_data.get("research")
        projection = cleaned_data.get("projection")
        knn_n_value = cleaned_data.get("knn_n_value")
        distance_matrix_metric = cleaned_data.get("distance_matrix_metric")
        if analysis_name_unique_check(name, research):
            self.add_error("name", "An analysis with this name already exists.")
            raise forms.ValidationError("An analysis with this name already exists.")
        if window_size is not None:
            self.window_overlap_checks(window_size, window_overlap, dataset)
        if distance_matrix_metric == '':
            self.add_error('distance_matrix_metric', 'Field required')
        if projection == 'knn_distance_n' and not knn_n_value:
            self.add_error("projection", "You must provide a value for n in knn_distance_n")
            raise forms.ValidationError("You must provide a value for n in knn_distance_n")

    class Meta:
        model = MapperAnalysis
        exclude = ['slug', 'graph', 'precomputed_distance_matrix', 'precomputed_distance_matrix_json']


class MapperAnalysisCreationForm_Precomputed(PrecomputedAnalysisCreationForm):
    def __init__(self, research, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['research'].initial = research
        self.helper = FormHelper(self)
        self.helper.form_method = 'POST'
        self.helper.form_action = reverse_lazy('analysis:mapperanalysis-create', kwargs={
                                 'form': 'precomputed',
                                 'research_slug': research.slug})
        self.helper.form_id = "analysis_form"
        self.helper.layout = Layout(
            'name',
            'description',
            Field('research', type="hidden"),
            Field('precomputed_distance_matrix', multiple=True),
            Field('precomputed_distance_matrix_json', type="hidden"),
            Fieldset(
                'fit_transform parameters',
                'projection',
                'knn_n_value',
                'scaler',
            ),
            Fieldset(
                'map parameters',
                'use_original_data',
                'clusterer',
                'cover_n_cubes',
                'cover_perc_overlap',
                'graph_nerve_min_intersection',
                'precomputed',
                'remove_duplicate_nodes'
            )
        )

    def clean(self):
        cleaned_data = super().clean()
        precomputed_distance_matrix = cleaned_data.get("precomputed_distance_matrix")
        name = cleaned_data.get("name")
        research = cleaned_data.get("research")
        projection = cleaned_data.get("projection")
        knn_n_value = cleaned_data.get("knn_n_value")
        if analysis_name_unique_check(name, research):
            self.add_error("name", "An analysis with this name already exists.")
            raise forms.ValidationError("An analysis with this name already exists.")
        if not precomputed_distance_matrix:
            self.add_error("precomputed_distance_matrix", "You must provide a precomputed distance matrix")
            raise forms.ValidationError("You must provide a precomputed distance matrix")
        if projection == 'knn_distance_n' and not knn_n_value:
            self.add_error("projection", "You must provide a value for n in knn_distance_n")
            raise forms.ValidationError("You must provide a value for n in knn_distance_n")

    class Meta:
        model = MapperAnalysis
        exclude = ['slug', 'graph', 'window_size', 'window_overlap', 'filtration_type', 'distance_matrix_metric']


class AnalysisBottleneckCreationForm(forms.Form):
    BOTTLENECK_OPTIONS = [(Bottleneck.CONS, 'Bottleneck of consecutive windows'),
                          (Bottleneck.ALL, 'Bottleneck of each window to each window')]
    bottleneck_type = forms.ChoiceField(widget=forms.RadioSelect, choices=BOTTLENECK_OPTIONS)
    homology = forms.ChoiceField(widget=forms.RadioSelect)

    def __init__(self, _homology, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['homology'].choices = [(i, i) for i in range(_homology+1)]


class WindowBottleneckCreationForm(forms.Form):
    homology = forms.ChoiceField(widget=forms.RadioSelect)

    def __init__(self, _homology, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['homology'].choices = [(i, i) for i in range(_homology+1)]
