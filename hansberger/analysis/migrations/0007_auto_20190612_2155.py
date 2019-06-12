# Generated by Django 2.0.13 on 2019-06-12 19:55

import django.contrib.postgres.fields.jsonb
from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('analysis', '0006_auto_20190611_2130'),
    ]

    operations = [
        migrations.AddField(
            model_name='filtrationanalysis',
            name='bottleneck_distance_consecutive',
            field=django.contrib.postgres.fields.jsonb.JSONField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='filtrationanalysis',
            name='bottleneck_distance_consecutive_diags',
            field=django.contrib.postgres.fields.jsonb.JSONField(blank=True, null=True),
        ),
    ]
