import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout
from tensorflow.keras.models import Model

from src.main.main import MERGE_FACTOR, TIME_FRAME
from src.utility.InstrumentDataset import read_data, save_spectrogram_image


def build_generator(latent_dim):
    model = tf.keras.Sequential()
    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(44 * 64, activation='tanh'))
    model.add(Reshape((44, 64, 1)))
    return model


def build_discriminator(input_shape):
    model = tf.keras.Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    return model


def train_gan(generator, discriminator, gan, data, latent_dim, epochs=10000, batch_size=64, save_interval=1000):
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_images = data[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_images = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_images, real)
        d_loss_fake = discriminator.train_on_batch(generated_images, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, real)

        if epoch % save_interval == 0:
            print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}%] [G loss: {g_loss}]")
            save_generated_images(generator, epoch)


def save_generated_images(generator, epoch, samples=5, latent_dim=100):
    noise = np.random.normal(0, 1, (samples, latent_dim))
    gen_images = generator.predict(noise)
    gen_images = 0.5 * gen_images + 0.5

    # save_image(epoch, gen_images, samples)
    save_spectrogram_image(gen_images[0], f"../../output/gan_images/gen_image_{epoch}_{10}.png")


def save_image(epoch, gen_images, samples):
    for i in range(samples):
        plt.imshow(gen_images[i, :, :, 0], cmap='gray')
        plt.axis('off')
        plt.savefig(f"../../output/gan_images/gen_image_{epoch}_{i}.png")
        plt.close()


latent_dim = 100
generator = build_generator(latent_dim)
discriminator = build_discriminator((44, 64, 1))

# Compile discriminator
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Build and compile GAN
discriminator.trainable = False
gan_input = Input(shape=(latent_dim,))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)
gan = Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

data, _, _ = read_data('../../Dataset', MERGE_FACTOR, TIME_FRAME, folder='../../Models/Instrument/splits/val')
data = np.expand_dims(data, axis=-1)
data = (data - 127.5) / 127.5

# Train the GAN
train_gan(generator, discriminator, gan, data, latent_dim)
