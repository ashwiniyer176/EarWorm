from http.client import HTTPResponse
from django.shortcuts import render
from .forms import AudioStoreForm
from .models import AudioStore
from .recognizer import Recognizer
import os
from django.conf import settings
import yaml


def read_yaml(path_to_yaml):
    try:
        with open(path_to_yaml, "r") as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print("Error reading the config file")


def get_genre_from_prediction(prediction):
    config = read_yaml("config.yaml")
    encoding = config["label_encodings"]["genre"]
    print(type(encoding))
    for k, v in encoding.items():
        if v == prediction:
            return k


model = Recognizer("models/MFCC_Recognition")

# Create your views here.
def index(request):
    if request.method == "POST":
        form = AudioStoreForm(request.POST, request.FILES or None)

        if form.is_valid():
            audio_file = request.FILES["record"].name
            form.save()
            prediction = model.predict(
                os.path.join(settings.MEDIA_ROOT, "documents", audio_file)
            )
            prediction = get_genre_from_prediction(prediction)

            context = {"prediction": prediction}
            return render(request, "recognition/index.html", context)
    elif request.method == "GET":
        form = AudioStoreForm()
        context = {"form": form}
        return render(request, "recognition/index.html", context)
