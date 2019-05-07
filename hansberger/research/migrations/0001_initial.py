# Generated by Django 2.0.13 on 2019-05-07 16:38

import django.contrib.postgres.fields.jsonb
from django.db import migrations, models
import django.db.models.deletion


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
                ('file', models.FileField(upload_to='research/datasets/')),
                ('file_type', models.CharField(choices=[('EDF', 'EDF file'), ('TXT', 'Text file')], max_length=3)),
                ('plot', models.ImageField(blank=True, max_length=300, null=True, upload_to='')),
                ('matrix', django.contrib.postgres.fields.jsonb.JSONField(blank=True, null=True)),
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
                ('description', models.TextField(blank=True, max_length=500, null=True)),
                ('creation_date', models.DateField(auto_now_add=True)),
                ('filtration_type', models.CharField(choices=[('VRF', 'Vietoris Rips Filtration'), ('CWRF', 'Clique Weighted Rank Filtration')], max_length=50)),
                ('distance_matrix_metric', models.CharField(choices=[('braycurtis', 'Braycurtis'), ('canberra', 'Canberra'), ('chebyshev', 'Chebyshev'), ('cityblock', 'City block'), ('correlation', 'Correlation'), ('cosine', 'Cosine'), ('dice', 'Dice'), ('euclidean', 'Euclidean'), ('hamming', 'Hamming'), ('jaccard', 'Jaccard'), ('jensenshannon', 'Jensen Shannon'), ('kulsinski', 'Kulsinski'), ('mahalanobis', 'Mahalonobis'), ('matching', 'Matching'), ('minkowski', 'Minkowski'), ('rogerstanimoto', 'Rogers-Tanimoto'), ('russellrao', 'Russel Rao'), ('seuclidean', 'Seuclidean'), ('sokalmichener', 'Sojal-Michener'), ('sokalsneath', 'Sokal-Sneath'), ('sqeuclidean', 'Sqeuclidean'), ('yule', 'Yule')], max_length=20)),
                ('max_homology_dimension', models.IntegerField(default=1)),
                ('max_distances_considered', models.FloatField(blank=True, default=None, null=True)),
                ('coeff', models.IntegerField(default=2)),
                ('do_cocycles', models.BooleanField(default=False)),
                ('n_perm', models.IntegerField(blank=True, default=None, null=True)),
                ('result_matrix', django.contrib.postgres.fields.jsonb.JSONField(blank=True, null=True)),
                ('result_plot', models.ImageField(blank=True, null=True, upload_to='')),
                ('dataset', models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, related_name='analysis_set', related_query_name='analysis', to='research.Dataset')),
            ],
            options={
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
            ],
            options={
                'verbose_name': 'research',
                'verbose_name_plural': 'research',
                'ordering': ['-creation_date'],
            },
        ),
        migrations.AddField(
            model_name='filtrationanalysis',
            name='research',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='analysis_set', related_query_name='analysis', to='research.Research'),
        ),
        migrations.AddField(
            model_name='dataset',
            name='research',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='datasets', related_query_name='dataset', to='research.Research'),
        ),
        migrations.CreateModel(
            name='EDFDataset',
            fields=[
            ],
            options={
                'proxy': True,
                'indexes': [],
            },
            bases=('research.dataset',),
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
            name='filtrationanalysis',
            unique_together={('slug', 'research')},
        ),
        migrations.AlterUniqueTogether(
            name='dataset',
            unique_together={('slug', 'research')},
        ),
    ]
