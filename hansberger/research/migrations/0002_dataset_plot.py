# Generated by Django 2.0.13 on 2019-05-22 07:25

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('research', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='dataset',
            name='plot',
            field=models.ImageField(blank=True, null=True, upload_to=''),
        ),
    ]
