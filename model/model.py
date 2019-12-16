import json
import logging
import os

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Embedding, dot, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model

import tensorflow as tf

from DataGenerator import DataGenerator
from mysql_client import mysql_client

from annoy import AnnoyIndex
from tqdm import tqdm

class Business2Vec:
    def __init__(self):
        self.model = None
        self.train_generator = None
        self.validation_generator = None
        self.trained_weights_path = './weights/best.hdf5'

        self.mysql_client = mysql_client()
        self.mysql_client.init()
        self.business_size = self.mysql_client.total_businesses
        self.embedding_file = './embeddings.json'
        self.embedding_size = 5
        self.business_embeddings = {}
        self.annoyIndex = None
        self.inverted_index_file = "./inverted_index.json"
        self.annyIndexToBusiness = {}
        self.inverted_index = {}

        if not os.path.exists('./weights'):
            os.mkdir('./weights')

    def build(self, vector_dim=5, learn_rate=0.1):
        self.embedding_size = vector_dim

        if os.path.exists(self.trained_weights_path):
            self.model = load_model(self.trained_weights_path)
        else:
            stddev = 1.0 / vector_dim
            initializer = tf.random_normal_initializer(mean=0.0, stddev=stddev, seed=None)

            business_input = Input(shape=(1,), name="business_input")
            business_emnbedding = Embedding(input_dim=self.business_size,
                                            output_dim=vector_dim,
                                            input_length=1,
                                            name="input_embedding",
                                            embeddings_initializer=initializer)(business_input)

            target_input = Input(shape=(1,), name="business_target")
            target_embedding = Embedding(input_dim=self.business_size,
                                         output_dim=vector_dim,
                                         input_length=1,
                                         name="target_embedding", embeddings_initializer=initializer)(target_input)

            merged = dot([business_emnbedding, target_embedding], axes=2, normalize=False, name="dot")
            merged = Flatten()(merged)
            output = Dense(1, activation='sigmoid', name="output")(merged)

            model = Model(inputs=[business_input, target_input], outputs=output)
            model.compile(loss="binary_crossentropy", optimizer=Adam(learn_rate), metrics=['accuracy'])

            self.model = model

        logging.info(self.model.summary())

    def __create_generator(self):
        train_start_seq = self.mysql_client.train_start_seq
        train_end_seq = self.mysql_client.train_end_seq

        validation_start_seq = self.mysql_client.validation_start_seq
        validation_end_seq = self.mysql_client.validation_end_seq

        self.train_generator = DataGenerator(train_start_seq, train_end_seq)
        self.validation_generator = DataGenerator(validation_start_seq, validation_end_seq)

    def train(self, epochs=2):
        logging.info("Training model")

        self.__create_generator()

        if os.path.exists(self.trained_weights_path):
            self.model = load_model(self.trained_weights_path)

        checkpointer = ModelCheckpoint(filepath=self.trained_weights_path, monitor='val_loss', save_best_only=True,
                                       mode='auto')
        earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

        self.model.fit_generator(generator=self.train_generator,
                                 validation_data=self.validation_generator,
                                 epochs=epochs,
                                 # use_multiprocessing=False,
                                 # workers=4,
                                 callbacks=[
                                     checkpointer,
                                     earlystopping
                                 ])

        self.model.load_weights(self.trained_weights_path)

    def retrieval_embeddings(self):
        if os.path.exists(self.trained_weights_path):
            self.model = load_model(self.trained_weights_path)
        else:
            raise FileNotFoundError("self.trained_weights_path does not exist")

        target_weights = self.model.get_layer('target_embedding').get_weights()

        self.business_embeddings = {}

        with open(self.embedding_file, 'w') as fp:
            for idx, embedding in enumerate(target_weights[0]):
                business = self.mysql_client.index2business[idx]
                self.business_embeddings[business] = embedding.tolist()

            json.dump(self.business_embeddings, fp)

    def calc_knn_annoy(self, top_k):
        """
        1. build annoy tree to compute k nearest neighbors
        2. saves top k nearest neighbors using annoy into inverted_index
        """
        if not os.path.exists(self.embedding_file):
            raise Exception("Invalid Operation", f"{self.embedding_file} does not exists")

        with open(self.embedding_file, 'r') as fp:
            self.business_embeddings = json.load(fp)

        self.annoyIndex = AnnoyIndex(self.embedding_size, 'angular')

        self.annyIndexToBusiness = {}
        for idx, key in tqdm(enumerate(self.business_embeddings.keys())):
            self.annoyIndex.add_item(idx, self.business_embeddings[key])
            self.annyIndexToBusiness[idx] = key

        self.annoyIndex.build(-1)

        self.inverted_index = {}

        print(len(self.business_embeddings))
        for idx, key in tqdm(enumerate(self.business_embeddings.keys())):
            indexes = self.annoyIndex.get_nns_by_item(idx, top_k + 1)

            values = []
            for target in indexes:
                if idx == target:
                    continue
                values.append(self.annyIndexToBusiness[target])

            self.inverted_index[key] = values

        with open(self.inverted_index_file, 'w') as fp:
            json.dump(self.inverted_index, fp)


if __name__ is '__main__':
    b2v = Business2Vec()
    b2v.build()
    # b2v.train(2)
    # b2v.retrieval_embeddings()
    b2v.calc_knn_annoy(5)
