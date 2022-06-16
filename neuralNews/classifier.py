import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf


class Classifier:

    def __init__(self):
        self.model = tf.keras.models.load_model("neuralNews/models/classifier/classifier_model")
        self.vectorizer = TfidfVectorizer(decode_error="replace", vocabulary=pickle.load(
            open("neuralNews/models/classifier/feature.pkl", "rb")))
        self.encoder = LabelEncoder()
        self.encoder.classes_ = np.load('neuralNews/models/classifier/classes.npy', allow_pickle=True)
        self.selector = pickle.load(open("neuralNews/models/classifier/selector.pkl", "rb"))

    def classify(self, text):
        prepared_text = self.selector.transform(self.vectorizer.fit_transform([text])).todense()
        prediction = self.model.predict(prepared_text)
        print(prediction)
        evaluated_class = np.argmax(prediction, axis=1)
        print(evaluated_class)
        return self.encoder.inverse_transform(evaluated_class)[0]
