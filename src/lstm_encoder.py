import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model, Model
from keras.layers import LSTM, Dense, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.losses import BinaryCrossentropy
import pickle
from nltk.stem import WordNetLemmatizer
import re

class LSTMEmbedding:
    def __init__(self):
        self.model = None
        self.input_shape = None
        self.encoder = None
        self.tokenizer = None
        self.embedding_size = 16
        self.filepath = 'data/utilities/data_encoder/'

    def _tokenize_data(self,concatenated_data):
        if self.tokenizer is None:
            self.tokenizer = Tokenizer()
            self.tokenizer.fit_on_texts(concatenated_data)
        encoded_data = self.tokenizer.texts_to_sequences(concatenated_data)

        X_pad = pad_sequences(encoded_data, padding='post')
        self.max_seq_length = X_pad.shape[1]

        return X_pad

    def _tokenize_batch(self,concatenated_data):
        self._read_max_length()
        if self.tokenizer is None:
            filepath = self.filepath
            with open(filepath + 'tokenizer.pkl', 'rb') as tokenizer_file:
                self.tokenizer = pickle.load(tokenizer_file)

        encoded_data = self.tokenizer.texts_to_sequences(concatenated_data)
        X_pad = pad_sequences(encoded_data, maxlen=self.max_seq_length, padding='post')
        return X_pad

    def _text_normalization(self,s:str):
            text = s.lower()
            text = re.sub(r'[^a-z\s]', '', text)

            lemmatizer = WordNetLemmatizer()
            words = [lemmatizer.lemmatize(word) for word in text]

            normalized_text = ' '.join(words)

            return normalized_text

    def preprocess_data(self, X, needs_tokenization=True):
        string_columns = X.select_dtypes(include=['object']).columns
        filtered_data = X[string_columns].fillna('').astype(str)
        concatenated_data = filtered_data.apply(lambda x: ' '.join(x), axis=1)
        concatenated_data = concatenated_data.apply(self._text_normalization)

        if needs_tokenization:
            return self._tokenize_data(concatenated_data)[:,:self.embedding_size]
        else:
            return concatenated_data

    def _build_model(self, input_shape):
        self.model = Sequential([
            Input(shape=input_shape),
            LSTM(64),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')   ])

    def train_model(self, X_train, y_train, epochs=10, batch_size=32):
        self.input_shape = (X_train.shape[1], 1)
        self._build_model(self.input_shape)

        self.model.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=['accuracy'])
        X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

        self.model.fit(X_train_reshaped, np.array(y_train), epochs=epochs, batch_size=batch_size)

    def evaluate_model(self, X_test, y_test):
        X_test = self.preprocess_data(X_test, needs_tokenization=True)
        return self.model.evaluate(X_test, np.array(y_test))[0]

    def save_model(self):
        filepath = self.filepath

        self.model.save(filepath + 'model.keras', include_optimizer=True)

        with open(filepath + 'tokenizer.pkl', 'wb') as tokenizer_file:
            pickle.dump(self.tokenizer, tokenizer_file)

        with open(filepath + 'max_len.txt', 'w') as text_file:
            text_file.write(str(self.max_seq_length))

    def load_model(self):
        filepath = self.filepath

        self.model = load_model(filepath+'model.keras')

        with open(filepath + 'tokenizer.pkl', 'rb') as tokenizer_file:
            self.tokenizer = pickle.load(tokenizer_file)

        with open(filepath + 'max_len.txt', 'r') as text_file:
            self.max_seq_length = int(text_file.read())

    def _read_max_length(self):
        with open(self.filepath + 'max_len.txt', 'r') as text_file:
            self.max_seq_length = int(text_file.read())

    def embedding_vector(self, X_batch, reload_model=True):
        if reload_model:
            self.load_model()

        if self.model is None or self.tokenizer is None:
            raise Exception("Model has not been trained yet.")

        X_batch_preprocessed = self.preprocess_data(pd.DataFrame(X_batch), needs_tokenization=False)
        X_batch_preprocessed = X_batch_preprocessed.apply(self._text_normalization)

        X_batch_preprocessed = self._tokenize_batch(X_batch_preprocessed)

        return X_batch_preprocessed