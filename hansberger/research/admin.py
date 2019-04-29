from django.contrib import admin
from .models import (
    Research,
    Dataset,
    EDFDataset,
    TextDataset,
    FiltrationAnalysis,
    VietorisFiltrationAnalysis,
    CliqueWeightedRankFiltrationAnalysis,
)

admin.site.register(Research)
admin.site.register(Dataset)
admin.site.register(EDFDataset)
admin.site.register(TextDataset)
admin.site.register(FiltrationAnalysis)
admin.site.register(VietorisFiltrationAnalysis)
admin.site.register(CliqueWeightedRankFiltrationAnalysis)
