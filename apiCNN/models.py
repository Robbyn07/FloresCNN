from django.db.models import Model
from django.db.models import CharField
from django.db.models import FloatField


class TablaFlores(Model):
    image_name = CharField(max_length=100, blank=False)
    flower_predicted = CharField(max_length=30, blank=False)
    flower_percentage = FloatField()
