import imp
from django.forms import ModelForm
from .models import AudioStore


class AudioStoreForm(ModelForm):
    class Meta:
        model = AudioStore
        fields = ["record"]
