from django.contrib import admin
from .models import Research, EdfSource, TextSource, Dataset

admin.site.register(Research)
admin.site.register(EdfSource)
admin.site.register(TextSource)
admin.site.register(Dataset)
