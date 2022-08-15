from scipy.io import wavfile
from scipy.signal import spectrogram
import numpy as np
import librosa.display
import pandas as pd
import matplotlib.pyplot as plt
from librosa.feature import mfcc
import librosa
import tensorflow.keras as keras
import tensorflow as tf
from scipy import spatial
import soundfile as sf
import os
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class RecommendationEngine:
    def __init__(self, audio_nn_path, meta_nn_path, data_path):
        """
        Initialize the models and knowledge base

        Args:
            audio_nn_path (str): Path to the Audio Music Recognition model
            meta_nn_path (str): Path to the Metadata Music Recommendation model
            data_path (str): Path to the Knowledge base(.csv file)
        """
        self.meta_nn_path = meta_nn_path
        self.audio_nn = self.create_model(audio_nn_path)
        self.df = pd.read_csv(data_path, index_col=0)

    def create_model(self, model_path):
        """
        Creates and intializes a tensorflow Sequential model based on path given

        Args:
            model_path (str): Path to saved tensorflow model

        Returns:
            (tensorflow.keras.Sequential): Returns a tensorflow Sequential model
        """
        return tf.keras.models.load_model(model_path)

    def preprocess_input(self, audio_file_path):
        """
        Preprocesses input to the correct format as required for the neural networks

        Args:
            audio_file_path (str): A path to a .wav file

        Returns:
            (numpy.ndarray): A (1,39) shaped numpy array containing mean MFCC features
        """
        samplerate, audio = wavfile.read(audio_file_path)
        if np.count_nonzero(audio) > 1600000:
            print("Audio is unkown(probably not music")
            return None
        mfcc_features = self.convert_to_mfcc(audio)
        return mfcc_features

    def convert_to_mfcc(self, data):
        """
        Given the path to an audio file calculates its MFCC features(39)

        Args:
            data (numpy.ndarray): A numpy array representing a mono sound signal

        Returns:
            (numpy.ndarray): A (1,39) shaped numpy array containing mean MFCC features
        """
        if len(data.shape) == 2:
            data = np.mean(data, axis=1)
        start = data.shape[0] // 3
        stop = start + data.shape[0] // 3
        data = data[start:stop]
        features = mfcc(y=data.astype("float16"), n_mfcc=39).T
        return np.mean(features, axis=0)

    def getCosineSimilarity(self, v1, v2):
        """
        Get cosine similarity of two numpy arrays

        Args:
            v1 (numpy array): Vector 1
            v2 (numpy array): Vector 2

        Returns:
            float: Cosine similarity value
        """
        cosine_similarity = 1 - spatial.distance.cosine(v1, v2)
        return round(float(cosine_similarity), 3)

    def recommendSongs(self, songGenre=None, trackName=None, numberOfSongs=5):
        """
        Main recommendation function.

        Args:
            trackName (str, optional):Recommend songs based on a known piece of music.
            numberOfSongs (int, optional): Number of song recommendations required. Defaults to 5.
            songGenre (int, optional): Used by audio_nn for recommendation based on unkown audio file. Defaults to None.

        Returns:
            _type_: _description_
        """
        df = self.df
        if trackName is None:
            trackName = df[df.Song_Label == songGenre]["track"].sample(1).values[0]
            # print(tracks)
        songData = df[df.track == trackName.lower()].values.tolist()
        featureVector = np.array([songData[0][:-3]])
        if songGenre is None:
            self.meta_nn = self.create_model(self.meta_nn_path)
            songGenre = np.argmax(self.meta_nn.predict(featureVector)[0])
        similarSongs = df[df.Song_Label == songGenre]
        X = similarSongs.drop(columns=["track", "artist", "Song_Label"])
        similarSongs["similarity"] = X.apply(
            lambda v2: self.getCosineSimilarity(featureVector, v2), axis=1
        )
        return (
            similarSongs[["track", "artist", "similarity"]]
            .sort_values(by="similarity", ascending=False)
            .iloc[: numberOfSongs + 1, :]
        )

    def check_input(self, filepath):
        assert os.path.splitext()[-1] in [".wav", ".mp3"]

    def predict(self, audio_file_path):
        """
        Given an audio file, recommends music

        Args:
            audio_file_path (str): Path to a .wav file

        Returns:
            (pandas.DataFrame): Pandas DataFrame containing track name, artist and similarity score
        """
        mfcc_features = self.preprocess_input(audio_file_path)
        # print(mfcc_features)
        if mfcc_features is not None:
            audio_nn_feature_vector = np.array([mfcc_features])
            output = self.audio_nn.predict(audio_nn_feature_vector)
            # print(output)
            audio_genre = np.argmax(output)
            return self.recommendSongs(songGenre=audio_genre)


if __name__ == "__main__":
    print("Loading models")
    model = RecommendationEngine(
        "./models/MFCC_Recognition/",
        "./models/Neural_Genre_Prediction/",
        "./data/scaled_music_metadata.csv",
    )
    print(
        "===================================================================================="
    )
    print("Recommending based on file")
    print(model.predict("./Rock_Music.wav"))
    print(
        "===================================================================================="
    )

    print("Recommending based on a known song")
    print(model.recommendSongs(trackName="Back in Black"))
    print(
        "===================================================================================="
    )

    print("Predicting based on Speech")
    print(model.predict("./Trek Monologue.wav"))
    print(
        "===================================================================================="
    )
