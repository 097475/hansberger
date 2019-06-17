from django import forms
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Fieldset, Field, Div
from django.urls import reverse_lazy
from .models import FiltrationAnalysis, MapperAnalysis
from datasets.models import Dataset


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
    ANALYSIS_OPTIONS = [('filtration_analysis', "Filtration Analysis with ripser"), ('mapper_analysis', "Mapper Analysis with KeplerMapper")]
    SOURCE_OPTIONS = [('dataset', 'Dataset'), ('precomputed', 'Precomputed distance matrices')]
    analysis = forms.ChoiceField(widget=forms.RadioSelect, choices=ANALYSIS_OPTIONS)
    source = forms.ChoiceField(widget=forms.RadioSelect, choices=SOURCE_OPTIONS)


class DatasetAnalysisCreationForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def window_overlap_checks(self, window_size, window_overlap, dataset):
        if window_size == 0:
            self.add_error("window_size", "Window size can't be equal to 0")
            raise forms.ValidationError("Window size can't be equal to 0")
        if window_size > len(dataset.data[0]):
            self.add_error("window_size", "Window size can't be greater than the number of columns in the dataset")
            raise forms.ValidationError("Window size can't be greater than the number of columns in the dataset")
        if window_overlap >= window_size:
            self.add_error("window_overlap", "Window overlap can't be greater than or equal to window size")
            raise forms.ValidationError("Window overlap can't be greater than or equal to window size")


class PrecomputedAnalysisCreationForm(forms.ModelForm):
    precomputed_distance_matrix = forms.FileField(required=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class FiltrationAnalysisCreationForm_Dataset(DatasetAnalysisCreationForm):
    def __init__(self, research, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['research'].initial = research
        self.fields['dataset'].queryset = Dataset.objects.filter(research__slug=research.slug)
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
            # Field('precomputed_distance_matrix', multiple=True),
            # Field('precomputed_distance_matrix_json', type="hidden"),
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
        # precomputed_distance_matrix = cleaned_data.get("precomputed_distance_matrix")
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
        if not dataset:
            raise forms.ValidationError("You must select a dataset")

        if filtration_type == FiltrationAnalysis.VIETORIS_RIPS_FILTRATION and distance_matrix_metric == '':
            raise forms.ValidationError("You must provide a distance matrix metric for a Vietoris-Rips Filtration")

    class Meta:
        model = FiltrationAnalysis
        exclude = ['slug', 'precomputed_distance_matrix', 'precomputed_distance_matrix_json']


class FiltrationAnalysisCreationForm_Precomputed(PrecomputedAnalysisCreationForm):
    def __init__(self, research, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['research'].initial = research
        # self.fields['dataset'].queryset = Dataset.objects.filter(research__slug=research.slug)
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
            # 'dataset',
            # Div(id='peek_dataset'),
            Field('precomputed_distance_matrix', multiple=True),
            Field('precomputed_distance_matrix_json', type="hidden"),
            # 'window_size',
            # 'window_overlap',
            # 'filtration_type',
            # 'distance_matrix_metric',
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
        # dataset = cleaned_data.get("dataset")
        precomputed_distance_matrix = cleaned_data.get("precomputed_distance_matrix")
        # filtration_type = cleaned_data.get("filtration_type")
        # distance_matrix_metric = cleaned_data.get("distance_matrix_metric")
        # window_overlap = cleaned_data.get("window_overlap")
        # window_size = cleaned_data.get("window_size")
        # filtration_type = cleaned_data.get("filtration_type")
        name = cleaned_data.get("name")
        research = cleaned_data.get("research")
        if analysis_name_unique_check(name, research):
            self.add_error("name", "An analysis with this name already exists.")
            raise forms.ValidationError("An analysis with this name already exists.")
        # if window_size is not None:
            # self.window_overlap_checks(window_size, window_overlap, dataset)
        if not precomputed_distance_matrix:
            raise forms.ValidationError("You must provide a precomputed distance matrix")

    class Meta:
        model = FiltrationAnalysis
        exclude = ['slug', 'dataset', 'window_size', 'window_overlap', 'filtration_type', 'distance_matrix_metric']


class MapperAnalysisCreationForm_Dataset(DatasetAnalysisCreationForm):
    def __init__(self, research, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['research'].initial = research
        self.fields['dataset'].queryset = Dataset.objects.filter(research__slug=research.slug)
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
            # Field('precomputed_distance_matrix', multiple=True),
            # Field('precomputed_distance_matrix_json', type="hidden"),
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
        # precomputed_distance_matrix = cleaned_data.get("precomputed_distance_matrix")
        window_overlap = cleaned_data.get("window_overlap")
        window_size = cleaned_data.get("window_size")
        name = cleaned_data.get("name")
        research = cleaned_data.get("research")
        projection = cleaned_data.get("projection")
        knn_n_value = cleaned_data.get("knn_n_value")
        if analysis_name_unique_check(name, research):
            self.add_error("name", "An analysis with this name already exists.")
            raise forms.ValidationError("An analysis with this name already exists.")
        if window_size is not None:
            self.window_overlap_checks(window_size, window_overlap, dataset)
        if not dataset:  # neither one was filled
            raise forms.ValidationError("You must select a dataset")
        if projection == 'knn_distance_n' and not knn_n_value:
            raise forms.ValidationError("You must provide a value for n in knn_distance_n")

    class Meta:
        model = MapperAnalysis
        exclude = ['slug', 'graph', 'precomputed_distance_matrix', 'precomputed_distance_matrix_json']


class MapperAnalysisCreationForm_Precomputed(PrecomputedAnalysisCreationForm):
    def __init__(self, research, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['research'].initial = research
        # self.fields['dataset'].queryset = Dataset.objects.filter(research__slug=research.slug)
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
            # 'dataset',
            # Div(id='peek_dataset'),
            Field('precomputed_distance_matrix', multiple=True),
            Field('precomputed_distance_matrix_json', type="hidden"),
            # 'window_size',
            # 'window_overlap',
            # 'distance_matrix_metric',
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
        # dataset = cleaned_data.get("dataset")
        precomputed_distance_matrix = cleaned_data.get("precomputed_distance_matrix")
        # window_overlap = cleaned_data.get("window_overlap")
        # window_size = cleaned_data.get("window_size")
        name = cleaned_data.get("name")
        research = cleaned_data.get("research")
        projection = cleaned_data.get("projection")
        knn_n_value = cleaned_data.get("knn_n_value")
        if analysis_name_unique_check(name, research):
            self.add_error("name", "An analysis with this name already exists.")
            raise forms.ValidationError("An analysis with this name already exists.")
        if not precomputed_distance_matrix:  # neither one was filled
            raise forms.ValidationError("You must provide a precomputed distance matrix")
        if projection == 'knn_distance_n' and not knn_n_value:
            raise forms.ValidationError("You must provide a value for n in knn_distance_n")

    class Meta:
        model = MapperAnalysis
        exclude = ['slug', 'graph', 'window_size', 'window_overlap', 'filtration_type', 'distance_matrix_metric']


'''
class AnalysisCreationForm(forms.ModelForm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def window_overlap_checks(self, window_size, window_overlap, dataset):
        if window_size == 0:
            self.add_error("window_size", "Window size can't be equal to 0")
            raise forms.ValidationError("Window size can't be equal to 0")
        if window_size > len(dataset.data[0]):
            self.add_error("window_size", "Window size can't be greater than the number of columns in the dataset")
            raise forms.ValidationError("Window size can't be greater than the number of columns in the dataset")
        if window_overlap >= window_size:
            self.add_error("window_overlap", "Window overlap can't be greater than or equal to window size")
            raise forms.ValidationError("Window overlap can't be greater than or equal to window size")


class FiltrationAnalysisCreationForm(AnalysisCreationForm):
    precomputed_distance_matrix = forms.FileField(required=False)

    def __init__(self, research, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['research'].initial = research
        self.fields['dataset'].queryset = Dataset.objects.filter(research__slug=research.slug)
        self.helper = FormHelper(self)
        self.helper.form_method = 'POST'
        self.helper.form_action = reverse_lazy('analysis:filtrationanalysis-create', kwargs={
                                 'research_slug': research.slug})
        self.helper.form_id = "analysis_form"
        self.helper.layout = Layout(
            'name',
            'description',
            Field('research', type="hidden"),
            'dataset',
            Div(id='peek_dataset'),
            Field('precomputed_distance_matrix', multiple=True),
            Field('precomputed_distance_matrix_json', type="hidden"),
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
        cleaned_data = super(FiltrationAnalysisCreationForm, self).clean()
        dataset = cleaned_data.get("dataset")
        precomputed_distance_matrix = cleaned_data.get("precomputed_distance_matrix")
        filtration_type = cleaned_data.get("filtration_type")
        distance_matrix_metric = cleaned_data.get("distance_matrix_metric")
        window_overlap = cleaned_data.get("window_overlap")
        window_size = cleaned_data.get("window_size")
        filtration_type = cleaned_data.get("filtration_type")
        name = cleaned_data.get("name")
        research = cleaned_data.get("research")
        if analysis_name_unique_check(name, research):
            self.add_error("name", "An analysis with this name already exists.")
            raise forms.ValidationError("An analysis with this name already exists.")
        if window_size is not None:
            self.window_overlap_checks(window_size, window_overlap, dataset)

        if dataset and precomputed_distance_matrix:  # both fields were filled
            raise forms.ValidationError("""You must either provide a precomputed distance matrix or select a dataset,
                                         not both.""")
        elif not (dataset or precomputed_distance_matrix):  # neither one was filled
            raise forms.ValidationError("You must either provide a precomputed distance matrix or select a dataset")

        if (filtration_type == FiltrationAnalysis.VIETORIS_RIPS_FILTRATION and distance_matrix_metric == '' and
           not precomputed_distance_matrix):
            raise forms.ValidationError("You must provide a distance matrix metric for a Vietoris-Rips Filtration")

    class Meta:
        model = FiltrationAnalysis
        exclude = ['slug', 'result_matrix', 'result_plot', 'result_entropy']


class MapperAnalysisCreationForm(AnalysisCreationForm):
    precomputed_distance_matrix = forms.FileField(required=False)

    def __init__(self, research, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['research'].initial = research
        self.fields['dataset'].queryset = Dataset.objects.filter(research__slug=research.slug)
        self.helper = FormHelper(self)
        self.helper.form_method = 'POST'
        self.helper.form_action = reverse_lazy('analysis:mapperanalysis-create', kwargs={
                                 'research_slug': research.slug})
        self.helper.form_id = "analysis_form"
        self.helper.layout = Layout(
            'name',
            'description',
            Field('research', type="hidden"),
            'dataset',
            Div(id='peek_dataset'),
            Field('precomputed_distance_matrix', multiple=True),
            Field('precomputed_distance_matrix_json', type="hidden"),
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
        cleaned_data = super(MapperAnalysisCreationForm, self).clean()
        dataset = cleaned_data.get("dataset")
        precomputed_distance_matrix = cleaned_data.get("precomputed_distance_matrix")
        window_overlap = cleaned_data.get("window_overlap")
        window_size = cleaned_data.get("window_size")
        name = cleaned_data.get("name")
        research = cleaned_data.get("research")
        projection = cleaned_data.get("projection")
        knn_n_value = cleaned_data.get("knn_n_value")
        if analysis_name_unique_check(name, research):
            self.add_error("name", "An analysis with this name already exists.")
            raise forms.ValidationError("An analysis with this name already exists.")
        if window_size is not None:
            self.window_overlap_checks(window_size, window_overlap, dataset)
        if dataset and precomputed_distance_matrix:  # both fields were filled
            raise forms.ValidationError("""You must either provide a precomputed distance matrix or select a dataset,
                                         not both.""")
        elif not (dataset or precomputed_distance_matrix):  # neither one was filled
            raise forms.ValidationError("You must either provide a precomputed distance matrix or select a dataset")
        if projection == 'knn_distance_n' and not knn_n_value:
            raise forms.ValidationError("You must provide a value for n in knn_distance_n")

    class Meta:
        model = MapperAnalysis
        exclude = ['slug', 'graph']
'''
