from django.contrib import admin

# Register your models here.
from .models import FiltrationAnalysis, MapperAnalysis

admin.site.register(FiltrationAnalysis)
admin.site.register(MapperAnalysis)
