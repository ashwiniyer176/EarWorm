from django.db import models

# Create your models here.
class AudioStore(models.Model):
    record = models.FileField(upload_to="./documents")

    class Meta:
        db_table = "audio_store"
