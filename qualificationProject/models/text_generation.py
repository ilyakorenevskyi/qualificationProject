from __future__ import unicode_literals
import tensorflow as tf
import numpy as np
import math
import string


class CustomGRUModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, gru_units):
        super().__init__(self)
        self.embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.gru_layer = tf.keras.layers.GRU(gru_units, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding_layer(x, training=training)
        if states is None:
            states = self.gru_layer.get_initial_state(x)
        x, states = self.gru_layer(x, initial_state=states, training=training)
        x = self.dense_layer(x, training=training)
        if return_state:
            return x, states
        return x


class Generator(tf.keras.Model):
    def __init__(self, model, encoder_layer, decoder_layer):
        super().__init__()
        self.model = model
        self.encoder_layer = encoder_layer
        self.decoder_layer = decoder_layer
        skip_tokens = self.encoder_layer(['[UNK]'])[:, None]
        skip_mask = tf.SparseTensor(values=[-float('inf')] * len(skip_tokens), indices=skip_tokens,
                                    dense_shape=[len(encoder_layer.get_vocabulary())])
        self.mask = tf.sparse.to_dense(skip_mask)

    @tf.function
    def generate(self, inputs, states=None, temperature=0.4):
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_tokens = self.encoder_layer(input_chars).to_tensor()
        logits, states = self.model(inputs=input_tokens, states=states, return_state=True)
        logits = logits[:, -1, :]
        logits = np.log(logits) / temperature
        exp_logits = np.exp(logits)
        logits = exp_logits / np.sum(exp_logits)
        logits = logits + self.mask
        output_tokens = tf.random.categorical(logits, num_samples=1)
        output_tokens = tf.squeeze(output_tokens, axis=-1)
        predicted_chars = self.decoder_layer(output_tokens)
        return predicted_chars, states


def get_encoders(text):
    vocabulary = sorted(set(text))
    encoder = tf.keras.layers.StringLookup(vocabulary=list(vocabulary), mask_token=None)
    decoder = tf.keras.layers.StringLookup(vocabulary=encoder.get_vocabulary(), invert=True, mask_token=None)
    return encoder, decoder


def split(sequence):
    return sequence[:-1], sequence[1:]


def prepare_data(text, encoder, decoder, sequence_len, batch_size, shuffle_buffer):
    encoded_text = encoder(tf.strings.unicode_split(text, 'UTF-8'))
    temp = tf.data.Dataset.from_tensor_slices(encoded_text)
    sequences = temp.batch(sequence_len + 1, drop_remainder=True)
    dataset = sequences.map(split)
    dataset = dataset.shuffle(shuffle_buffer).batch(batch_size, drop_remainder=True).prefetch(
        tf.data.experimental.AUTOTUNE)
    return dataset


def train_model(encoder, epochs, dataset):
    model = CustomGRUModel(vocab_size=len(encoder.get_vocabulary), embedding_size=256, gru_units=2048)
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss)
    history = model.fit(dataset, epochs=epochs)
    return history, model
