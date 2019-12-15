import logging
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Embedding, dot, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping

import tensorflow as tf

from DataGenerator import DataGenerator
from mysql_client import mysql_client


class Business2Vec:
    def __init__(self):
        self.model = None
        self.train_generator = None
        self.validation_generator = None
        self.trained_weights_path = "./weights"

        self.mysql_client = mysql_client()
        self.mysql_client.init()
        self.business_size = self.mysql_client.total_businesses

    def build(self, vector_dim, learn_rate):
        stddev = 1.0 / vector_dim
        initializer = tf.random_normal_initializer(mean=0.0, stddev=stddev, seed=None)

        business_input = Input(shape=(1,), name="business_input")
        business_emnbedding = Embedding(input_dim=self.business_size,
                                        output_dim=vector_dim,
                                        input_length=1,
                                        name="business_embedding",
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

        logging.info(model.summary())

        self.model = model

        # self.model.load_weights(self.trained_weights_path)

    def __create_generator(self):
        train_start_seq = self.mysql_client.train_start_seq
        train_end_seq = self.mysql_client.train_end_seq

        validation_start_seq = self.mysql_client.validation_start_seq
        validation_end_seq = self.mysql_client.validation_end_seq

        self.train_generator = DataGenerator(train_start_seq, train_end_seq, mysql_client=self.mysql_client)
        self.validation_generator = DataGenerator(validation_start_seq, validation_end_seq, mysql_client=self.mysql_client)

    def train(self):
        logging.info("Training model")

        self.__create_generator()

        checkpointer = ModelCheckpoint(filepath=self.trained_weights_path,
                                          monitor='val_loss',
                                          save_best_only=True,
                                          mode='auto')
        1
        earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

        self.model.fit_generator(generator=self.train_generator,
                                 validation_data=self.validation_generator,
                                 # use_multiprocessing=True,
                                 # workers=4,
                                 callbacks=[checkpointer, earlystopping])

        self.model.load_weights(self.trained_weights_path)


if __name__ is '__main__':
    b2v = Business2Vec()
    b2v.build(vector_dim=10, learn_rate=0.1)
    b2v.train()