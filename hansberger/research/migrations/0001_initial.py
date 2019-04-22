# Generated by Django 2.0.13 on 2019-04-22 14:18

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
                ('slug', models.SlugField(blank=True, max_length=150, null=True)),
                ('description', models.TextField(blank=True, max_length=500, null=True)),
                ('creation_date', models.DateField(auto_now_add=True)),
                ('plot', models.ImageField(upload_to='')),
                ('matrix', django.contrib.postgres.fields.jsonb.JSONField()),
            ],
            options={
                'verbose_name': 'dataset',
                'verbose_name_plural': 'datasets',
                'ordering': ['-creation_date'],
            },
        ),
        migrations.CreateModel(
            name='EdfSource',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('file', models.FileField(upload_to='research/datasets/sources/edf')),
                ('dataset', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='edfsource_of', to='research.Dataset')),
            ],
        ),
        migrations.CreateModel(
            name='Research',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=150)),
                ('slug', models.SlugField(blank=True, max_length=150, null=True, unique=True)),
                ('description', models.TextField(blank=True, max_length=500, null=True)),
                ('creation_date', models.DateField(auto_now_add=True)),
            ],
            options={
                'verbose_name': 'research',
                'verbose_name_plural': 'research',
                'ordering': ['-creation_date'],
            },
        ),
        migrations.CreateModel(
            name='TextSource',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('file', models.FileField(upload_to='research/datasets/sources/text')),
                ('values_separator_character', models.CharField(max_length=5, verbose_name='delimiter character of the values in the file')),
                ('identity_column_index', models.IntegerField(verbose_name='column number that identifies the progressive number of rows in the file')),
                ('header_row_index', models.IntegerField(verbose_name='row number that identifies the column in the file')),
                ('dataset', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='textsource_of', to='research.Dataset')),
            ],
        ),
        migrations.AddField(
            model_name='dataset',
            name='research',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='datasets', related_query_name='dataset', to='research.Research'),
        ),
        migrations.AlterUniqueTogether(
            name='dataset',
            unique_together={('slug', 'research')},
        ),
    ]
