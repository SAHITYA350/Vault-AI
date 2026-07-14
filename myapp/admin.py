from django.contrib import admin
from .models import UserProfile, Document, Collection

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'storage_used', 'storage_limit']

@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'name', 'file_type', 'file_size', 'uploaded_at']
    list_filter = ['file_type', 'uploaded_at']
    search_fields = ['name', 'extracted_text']

@admin.register(Collection)
class CollectionAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'name', 'created_at']

    