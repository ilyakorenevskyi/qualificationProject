import tensorflow as tf


class Generator:

    def __init__(self, model_name):
        self.model = tf.saved_model.load('neuralNews/models/generation/' + model_name)

    def generate(self, seed, count):
        states = None
        next_char = tf.constant([seed])
        result = [next_char]
        for n in range(count):
            next_char, states = self.model.generate(next_char, states=states)
            result.append(next_char)
        result = tf.strings.join(result)
        return result[0].numpy().decode('utf-8')
