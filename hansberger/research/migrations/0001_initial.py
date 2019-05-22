# Generated by Django 2.0.13 on 2019-05-22 07:44

import django.contrib.postgres.fields.jsonb
from django.db import migrations, models
import django.db.models.deletion
import research.models.dataset


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Dataset',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=150)),
                ('slug', models.SlugField(max_length=150)),
                ('description', models.TextField(blank=True, max_length=500, null=True)),
                ('creation_date', models.DateField(auto_now_add=True)),
                ('source', models.FileField(upload_to=research.models.dataset.dataset_directory_path)),
                ('source_type', models.CharField(choices=[('TXT', 'Text')], max_length=3)),
                ('storage_path', models.CharField(max_length=500)),
                ('data', django.contrib.postgres.fields.jsonb.JSONField(blank=True, null=True)),
                ('plot', models.ImageField(blank=True, max_length=500, null=True, upload_to='')),
            ],
            options={
                'verbose_name': 'dataset',
                'verbose_name_plural': 'datasets',
                'ordering': ['-creation_date'],
            },
        ),
        migrations.CreateModel(
            name='FiltrationAnalysis',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('slug', models.SlugField(max_length=110)),
                ('description', models.TextField(blank=True, max_length=500)),
                ('creation_date', models.DateField(auto_now_add=True)),
                ('precomputed_distance_matrix', models.FileField(blank=True, default=None, null=True, upload_to='research/precomputed/')),
                ('window_size', models.PositiveIntegerField(blank=True, default=None, help_text='Leave window size blank or 0 to not use windows', null=True)),
                ('window_overlap', models.PositiveIntegerField(default=0)),
                ('filtration_type', models.CharField(choices=[('VRF', 'Vietoris Rips Filtration'), ('CWRF', 'Clique Weighted Rank Filtration')], max_length=50)),
                ('distance_matrix_metric', models.CharField(blank=True, choices=[('braycurtis', 'Braycurtis'), ('canberra', 'Canberra'), ('chebyshev', 'Chebyshev'), ('cityblock', 'City block'), ('correlation', 'Correlation'), ('cosine', 'Cosine'), ('dice', 'Dice'), ('euclidean', 'Euclidean'), ('hamming', 'Hamming'), ('jaccard', 'Jaccard'), ('jensenshannon', 'Jensen Shannon'), ('kulsinski', 'Kulsinski'), ('mahalanobis', 'Mahalonobis'), ('matching', 'Matching'), ('minkowski', 'Minkowski'), ('rogerstanimoto', 'Rogers-Tanimoto'), ('russellrao', 'Russel Rao'), ('seuclidean', 'Seuclidean'), ('sokalmichener', 'Sojal-Michener'), ('sokalsneath', 'Sokal-Sneath'), ('sqeuclidean', 'Sqeuclidean'), ('yule', 'Yule')], max_length=20)),
                ('max_homology_dimension', models.IntegerField(default=1)),
                ('max_distances_considered', models.FloatField(blank=True, default=None, null=True)),
                ('coeff', models.IntegerField(default=2)),
                ('do_cocycles', models.BooleanField(default=False)),
                ('n_perm', models.IntegerField(blank=True, default=None, null=True)),
                ('result_matrix', django.contrib.postgres.fields.jsonb.JSONField(blank=True, null=True)),
                ('result_plot', models.ImageField(blank=True, max_length=300, null=True, upload_to='')),
                ('result_entropy', django.contrib.postgres.fields.jsonb.JSONField(blank=True, null=True)),
                ('dataset', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='filtrationanalysis_requests_created', related_query_name='analysis', to='research.Dataset')),
            ],
            options={
                'verbose_name': 'filtration analysis',
                'verbose_name_plural': 'filtration analyses',
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='MapperAnalysis',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('slug', models.SlugField(max_length=110)),
                ('description', models.TextField(blank=True, max_length=500)),
                ('creation_date', models.DateField(auto_now_add=True)),
                ('precomputed_distance_matrix', models.FileField(blank=True, default=None, null=True, upload_to='research/precomputed/')),
                ('window_size', models.PositiveIntegerField(blank=True, default=None, help_text='Leave window size blank or 0 to not use windows', null=True)),
                ('window_overlap', models.PositiveIntegerField(default=0)),
                ('distance_matrix_metric', models.CharField(choices=[('braycurtis', 'Braycurtis'), ('canberra', 'Canberra'), ('chebyshev', 'Chebyshev'), ('cityblock', 'City block'), ('correlation', 'Correlation'), ('cosine', 'Cosine'), ('dice', 'Dice'), ('euclidean', 'Euclidean'), ('hamming', 'Hamming'), ('jaccard', 'Jaccard'), ('jensenshannon', 'Jensen Shannon'), ('kulsinski', 'Kulsinski'), ('mahalanobis', 'Mahalonobis'), ('matching', 'Matching'), ('minkowski', 'Minkowski'), ('rogerstanimoto', 'Rogers-Tanimoto'), ('russellrao', 'Russel Rao'), ('seuclidean', 'Seuclidean'), ('sokalmichener', 'Sojal-Michener'), ('sokalsneath', 'Sokal-Sneath'), ('sqeuclidean', 'Sqeuclidean'), ('yule', 'Yule')], default='euclidean', max_length=20)),
                ('projection', models.CharField(choices=[('sum', 'Sum'), ('mean', 'Mean'), ('median', 'Median'), ('max', 'Max'), ('min', 'Min'), ('std', 'Std'), ('dist_mean', 'Dist_mean'), ('l2norm', 'L2norm'), ('knn_distance_n', 'knn_distance_n')], max_length=50)),
                ('scaler', models.CharField(choices=[('None', 'None'), ('MinMaxScaler', 'MinMaxScaler'), ('MaxAbsScaler', 'MaxAbsScaler'), ('RobustScaler', 'RobustScaler'), ('StandardScaler', 'StandardScaler')], max_length=50)),
                ('use_original_data', models.BooleanField(default=False)),
                ('clusterer', models.CharField(choices=[('k-means', 'K-Means'), ('affinity_propagation', 'Affinity propagation'), ('mean-shift', 'Mean-shift'), ('spectral_clustering', 'Spectral clustering'), ('agglomerative_clustering', 'StandardScaler'), ('DBSCAN', 'DBSCAN'), ('gaussian_mixtures', 'Gaussian mixtures'), ('birch', 'Birch')], default='DBSCAN', max_length=50)),
                ('cover_n_cubes', models.IntegerField(default=10)),
                ('cover_perc_overlap', models.FloatField(default=0.5)),
                ('graph_nerve_min_intersection', models.IntegerField(default=1)),
                ('remove_duplicate_nodes', models.BooleanField(default=False)),
                ('graph', models.TextField(blank=True, null=True)),
                ('dataset', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='mapperanalysis_requests_created', related_query_name='analysis', to='research.Dataset')),
            ],
            options={
                'verbose_name': 'mapper algorithm analysis',
                'verbose_name_plural': 'mapper algoritm analyses',
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='Research',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=150)),
                ('slug', models.SlugField(max_length=150, unique=True)),
                ('description', models.TextField(blank=True, max_length=500, null=True)),
                ('creation_date', models.DateField(auto_now_add=True)),
                ('storage_path', models.CharField(max_length=300)),
            ],
            options={
                'verbose_name': 'research',
                'verbose_name_plural': 'research',
                'ordering': ['-creation_date'],
            },
        ),
        migrations.AddField(
            model_name='mapperanalysis',
            name='research',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='mapperanalysis_requests_created', related_query_name='analysis', to='research.Research'),
        ),
        migrations.AddField(
            model_name='filtrationanalysis',
            name='research',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='filtrationanalysis_requests_created', related_query_name='analysis', to='research.Research'),
        ),
        migrations.AddField(
            model_name='dataset',
            name='research',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='text_datasets', related_query_name='text_dataset', to='research.Research'),
        ),
        migrations.CreateModel(
            name='TextDataset',
            fields=[
            ],
            options={
                'proxy': True,
                'indexes': [],
            },
            bases=('research.dataset',),
        ),
        migrations.AlterUniqueTogether(
            name='mapperanalysis',
            unique_together={('slug', 'research')},
        ),
        migrations.AlterUniqueTogether(
            name='filtrationanalysis',
            unique_together={('slug', 'research')},
        ),
        migrations.AlterUniqueTogether(
            name='dataset',
            unique_together={('slug', 'research')},
        ),
    ]
