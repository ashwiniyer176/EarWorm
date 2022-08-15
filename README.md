# EarWorm - An audio based Music Recommender

EarWorm is a music recommendation engine that recommends music based on audio. You can upload advertisement jingles, movie themes and any kind of music audio file (.wav only for now) and EarWorm recommends music based on the audio clip.

## Dataset

The dataset used is the [GTZAN Dataset for Music Genre Classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) from Kaggle. The task is a classification task, with 9 classes:

    1. Blues
    2. Classical
    3. Country
    4. Disco
    5. Hip-Hop
    6. Metal
    7. Pop
    8. Reggae
    9. Rock

## Stack

**Audio:** I have used scipy to read and parse .wav files. Librosa and soundfile were used for feature extraction of .wav file to MFCCs.

**Neural Network**: I used Tensorflow 2.0 for the purpose of Neural Network creation and training.

**Web Application:** For the purpose of creating a web application, I have used Django 3.0 on the backend.

## Usage:

1. Download [GTZAN Dataset for Music Genre Classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) and save it in `PROJECT_ROOT/data/`

2. Run `recognition/notebooks/Dataset Creation.ipynb` to create `data/Song_Data_MFCC.csv`.

3. Create `PROJECT_ROOT/models`

4. Run `recognition/notebooks/Data Cleaning.ipynb` to create `recognition/input` folder.

5. Create `PROJECT_ROOT/.env`. Add the line SECRET_KEY="<some_random_set_of_characters>"

6. Run the following set of commands to get Django running

```
    python manage.py makemigrations
    python manage.py migrate
    python manage.py runserver
```

6. Go to http://127.0.0.1:8000/. Upload your file and you will get your prediction.

**NOTE:** To check if the files are being uploaded properly, check if your root directory contains a `media/documents` directory. This is where files get saved.

## Future Work
Good ideas or strategies that you were not able to implement which you think can help  improve performance.
