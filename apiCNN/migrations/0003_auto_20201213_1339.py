# Generated by Django 3.1.3 on 2020-12-13 18:39

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('apiCNN', '0002_auto_20201213_0021'),
    ]

    operations = [
        migrations.RenameField(
            model_name='tablaflores',
            old_name='animal_percentage',
            new_name='flower_percentage',
        ),
        migrations.RenameField(
            model_name='tablaflores',
            old_name='animal_predicted',
            new_name='flower_predicted',
        ),
    ]
