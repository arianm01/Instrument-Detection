import tensorflow as tf
from tensorflow.keras.layers import Layer


class EuclideanDistanceLayer(Layer):
    def __init__(self, **kwargs):
        super(EuclideanDistanceLayer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        featsA, featsB = inputs
        sum_squared = tf.reduce_sum(tf.square(featsA - featsB), axis=1, keepdims=True)
        return tf.sqrt(tf.maximum(sum_squared, tf.keras.backend.epsilon()))

    def get_config(self):
        config = super(EuclideanDistanceLayer, self).get_config()
        return config
