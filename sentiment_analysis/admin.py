from django.contrib import admin
from sentiment_analysis.models import Review
# Register your models here.

@admin.register(Review)
class ReviewAdmin(admin.ModelAdmin):
    pass