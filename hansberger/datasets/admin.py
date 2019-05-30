from django.contrib import admin
from .models import TextDataset, EDFDataset

admin.site.register(TextDataset)
admin.site.register(EDFDataset)
