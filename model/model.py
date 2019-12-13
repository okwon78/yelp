import logging
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Embedding, Dot, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint

from tensorflow.initializers import RandomNormal

from tqdm import tqdm


class Business2Vec:
    def __init__(self):
        self.model = None
        self.train_generator = None
        self.validation_generator = None
        self.trained_weights_pat = "./weights"

    def build(self, vector_dim, business_size, learn_rate):
        stddev = 1.0 / vector_dim
        initializer = RandomNormal(mean=0.0, stddev=stddev, seed=None)

        business_input = Input(shape=(1,), name="business_input")
        business_emnbedding = Embedding(input_dim=business_size,
                                        output_dim=vector_dim,
                                        input_length=1,
                                        name="business_embedding",
                                        embeddings_initializer=initializer)(business_input)

        target_input = Input(shape=(1,), name="business_target")
        target_embedding = Embedding(input_dim=business_size,
                                     output_dim=vector_dim,
                                     input_length=1,
                                     name="target_embedding", embeddings_initializer=initializer)(target_input)

        merged = Dot([business_emnbedding, target_embedding], axes=2, normalize=False, name="dot")
        merged = Flatten()(merged)
        output = Dense(1, activation='sigmoid', name="output")(merged)

        model = Model(inputs=[business_input, target_input], outputs=output)
        model.compile(loss="binary_crossentropy", optimizer=Adam(learn_rate), metrics=['accuracy'])

        self.model = model

    def create_generator(self):
        pass

    def train(self, epochs, steps_per_epoch, validation_steps):
        logging.info("Training model")

        cb_checkpointer = ModelCheckpoint(filepath=self.trained_weights_path,
                                          monitor='val_loss',
                                          save_best_only=True,
                                          mode='auto')

        self.model.fit_generator(self.train_generator,
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=epochs,
                                 validation_data=self.validation_generator,
                                 validation_steps=validation_steps,
                                 callbacks=[cb_checkpointer])

        self.model.load_weights(self.trained_weights_path)