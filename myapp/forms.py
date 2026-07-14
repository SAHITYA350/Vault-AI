from django import forms
from .models import Document

class DocumentForm(forms.ModelForm):
    class Meta:
        model = Document
        fields = ['file', 'name']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control bg-dark border-secondary text-white', 
                'placeholder': 'Optional: Rename File'
            }),
            'file': forms.ClearableFileInput(attrs={
                'class': 'form-control bg-dark border-secondary text-white'
            }),
        }