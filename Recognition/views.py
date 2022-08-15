from http.client import HTTPResponse
from django.shortcuts import render
from .forms import AudioStoreForm
from .models import AudioStore

# Create your views here.
def index(request):
    if request.method == "POST":
        form = AudioStoreForm(request.POST, request.FILES or None)
        if form.is_valid():
            form.save()
            audio_files = AudioStore.objects.all()
            context = {"audio_files": audio_files}
            return render(request, "recognition/index.html", context)
    elif request.method == "GET":
        form = AudioStoreForm()
        context = {"form": form}
        return render(request, "recognition/index.html", context)
