import numpy as np
import tensorflow as tf


def testGPU():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    tf.config.list_physical_devices('GPU')


def reshape_data(data, time_steps):
    num_samples = data.shape[0]
    num_frames = int(data.shape[1] / time_steps) * time_steps
    num_mfcc = 13
    num_channels = 1
    frames_per_step = num_frames // time_steps  # This should be 43 time steps

    # Reshape the data
    new_shape = (num_samples, time_steps, frames_per_step, num_mfcc, num_channels)

    return np.reshape(data[:, :num_frames, :, :], new_shape)
