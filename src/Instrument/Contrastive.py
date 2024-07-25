from keras import layers
from tensorflow import keras
import tensorflow as tf
import tensorflow_addons as tfa

INPUT_SHAPE = (44, 64, 1)


class SupervisedContrastiveLoss(keras.losses.Loss):
    def __init__(self, temperature=1, name=None, reduction=keras.losses.Reduction.AUTO):
        super().__init__(name=name, reduction=reduction)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)

    def get_config(self):
        config = super().get_config()
        config.update({"temperature": self.temperature})
        return config


def add_projection_head(encoder):
    inputs = keras.Input(shape=INPUT_SHAPE)
    features = encoder(inputs)
    outputs = layers.Dense(projection_units, activation="relu")(features)
    model = keras.Model(
        inputs=inputs, outputs=outputs, name="cifar-encoder_with_projection-head"
    )
    return model


def create_encoder():
    resnet = keras.applications.ResNet50V2(
        include_top=False, weights=None, input_shape=INPUT_SHAPE, pooling="avg"
    )

    inputs = keras.Input(shape=INPUT_SHAPE)
    outputs = resnet(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar10-encoder")
    return model


def create_classifier(encoder, num_output, trainable=True):
    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = keras.Input(shape=INPUT_SHAPE)
    features = encoder(inputs)
    features = layers.Dropout(dropout_rate)(features)
    features = layers.Dense(512, activation="relu")(features)
    features = layers.Dropout(dropout_rate)(features)
    features = layers.Dense(hidden_units, activation="relu")(features)
    features = layers.Dropout(dropout_rate)(features)
    features = layers.Dense(128, activation="relu")(features)
    features = layers.Dropout(dropout_rate)(features)
    outputs = layers.Dense(num_output, activation="softmax")(features)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar10-classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


learning_rate = 0.001
hidden_units = 256
projection_units = 1024
dropout_rate = 0.5
