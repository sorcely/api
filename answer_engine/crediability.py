import tensorflow as tf
from tensorflow import keras
import numpy as np

from tensorflow.keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    GRU,
    Dense,
    Dropout)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

import google_storage

import os
import sys
import pickle

MODEL_PATH = 'models' if __name__ != '__main__' else '../models/' # Changing the path to files in the model folder

class crediability_scoring:

    def __init__(self, model, tokenizer):
        raise NotImplementedError()

        self.model = model
        self.tokenizer = tokenizer
        self.config = model.config

    def __call__(self, inputs:list):
        # inputs: the articles we want to classify

        # Preprocess data
        preprocessed_data = []
        for i in inputs:
            i = i[:self.config.seqlen]
            i = self.tokenizer.encode(i)
            preprocessed_data.append(i)
        inputs = preprocessed_data

        # Run a prediction
        preds = self.model(inputs)

        return preds

class tokenizer:

    def __init__(self, config):
        # config: the model configuartion object
        self.config = config

    def encode(self, inputs):
        # Tokenize
        inputs.split(' ')

        # Turn it into lowercase
        inputs = inputs.lower()

        # Encode using onehot
        one_hot_arr = [one_hot(words, self.config.vocab_sz) for words in inputs]

        # Embed using the built-in keras function
        embedded = pad_sequences(one_hot_arr, padding='post', maxlen=self.config.seqlen)

        # Turn it into a np array
        return embedded

class config:

    vocab_sz = 10000
    embedding_sz = 64
    seqlen = 384
    units = [512, 256, 1]
    dropout_rate = .2

class model:

    def __init__(self, model_path, config):
        # model_path: The name of the model excluding the models folder
        # config: the model configuartion object

        # # Create model_path including the model folder
        # if os.path.exists(os.path.join('../models', model_path)):
        #     model_path = os.path.join('../models', model_path)
        # elif os.path.exists(os.path.join('models', model_path)):
        #     model_path = os.path.join('models', model_path)            

        # Load the config
        self.config = config

        # Load the model
        self.model = self.create_model(config)
        # self.model.load_weights(model_path)

    def create_model(self, config):
        model=Sequential()
        model.add(Embedding(
            config.vocab_sz, 
            config.embedding_sz, 
            input_length=config.seqlen))
        model.add(Bidirectional(GRU(
            config.units[0])))
        model.add(Dropout(
            config.dropout_rate))
        model.add(Dense(
            config.units[1], 
            activation='sigmoid'))
        model.add(Dropout(
            config.dropout_rate))
        model.add(Dense(
            1, 
            activation='sigmoid'))
        return model

    def __call__(self, inputs):
        print('*'*50)
        print(np.array(inputs).shape)
        print('*'*50)
        outputs = self.model.predict(np.array(inputs), batch_size=len(inputs))
        print(outputs, outputs.shape)
        # Do some post processing
        # Basically turn the answer into a binary number
        outputs = np.sum(outputs) / self.config.seqlen
        bin_outputs = int(round(outputs))

        return bin_outputs, outputs

if __name__ == "__main__":
    model_ = model('../models/GRU-article-crediability.h5', config=config)
    model_.model.summary()
    tokenizer_ = tokenizer(config)
    crediability = crediability_scoring(model_, tokenizer_)
    text = """There have been so many conversations on the impact of fake news on the recent US elections. An already polarized public is pushed further apart by stories that affirm beliefs or attack the other side. Yes. Fake news is a serious problem that should be addressed. But by focusing solely on that issue, we are missing the larger, more harmful phenomenon of misleading, biased propaganda."""

    out = crediability([text]*2)
    print(out,len(out))

