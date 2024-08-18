from keras import layers, Sequential, Input, models
from tensorflow import keras
import tensorflow as tf
import tensorflow_addons as tfa


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


def add_projection_head(encoder, input_shape):
    inputs = keras.Input(shape=input_shape)
    features = encoder(inputs)
    outputs = layers.Dense(projection_units, activation="relu")(features)
    model = keras.Model(
        inputs=inputs, outputs=outputs, name="cifar-encoder_with_projection-head"
    )
    return model


def create_encoder(layer_sizes, input_shape):
    # resnet = keras.applications.ResNet50V2(
    #     include_top=False, weights=None, input_shape=INPUT_SHAPE, pooling="avg"
    # )
    #
    # inputs = keras.Input(shape=INPUT_SHAPE)
    # outputs = resnet(inputs)
    # model = keras.Model(inputs=inputs, outputs=outputs, name="cifar10-encoder")

    model = Sequential()
    model.add(layers.Conv2D(layer_sizes[0], (3, 3), activation='relu', input_shape=input_shape, padding='same'))
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    for size in layer_sizes[1:]:
        model.add(layers.Conv2D(size, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.3))

    return model


def create_classifier(encoder, num_output, input_shape, trainable=True):
    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = keras.Input(shape=input_shape)
    features = encoder(inputs)
    features = layers.Dropout(dropout_rate)(features)
    # features = layers.Dense(256, activation="relu")(features)
    # features = layers.Dropout(dropout_rate)(features)
    features = layers.Dense(hidden_units, activation="relu")(features)
    features = layers.Dropout(dropout_rate)(features)
    features = layers.Dense(64, activation="relu")(features)
    features = layers.Dropout(dropout_rate)(features)
    features = layers.Dense(32, activation="relu")(features)
    features = layers.Dropout(dropout_rate)(features)
    outputs = layers.Dense(num_output, activation="softmax")(features)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar10-classifier")
    model.compile(
        loss='categorical_crossentropy', metrics=['accuracy'],
        optimizer=keras.optimizers.Adam(learning_rate),
    )
    return model


learning_rate = 0.001
hidden_units = 128
projection_units = 128
dropout_rate = 0.3
