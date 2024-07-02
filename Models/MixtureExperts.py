import numpy as np
from keras.layers import Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.utils.version_utils import callbacks
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Softmax
from tensorflow.keras.regularizers import l2


def create_gating_network(input_shape, num_experts, dropout_rate=0.5):
    input_layer = Input(shape=input_shape)

    x = Conv2D(16, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = Flatten()(x)

    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_rate)(x)

    output_layer = Dense(num_experts, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train(X_train, y_train, X_val, y_val, experts):
    # Assuming X_train, y_train, X_val, y_val are your dataset splits
    input_shape = X_train.shape[1:]
    num_experts = len(experts)
    # Create the enhanced gating network
    gating_network = create_gating_network(input_shape, num_experts)

    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath='mixture_ensemble.keras', save_best_only=True, monitor='val_loss', mode='min', verbose=1)
    # Train the gating network
    gating_network.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=16,
                       callbacks=[model_checkpoint_callback])


def split_into_chunks(X, chunk_size):
    """Split the input array into chunks of specified size."""
    chunks = []
    num_samples, total_length, num_features, _ = X.shape
    num_chunks = (total_length + chunk_size - 1) // chunk_size  # Ceiling division

    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, total_length)
        chunk = X[:, start:end, ...]
        # If the chunk is smaller than chunk_size, pad it with zeros
        if end - start < chunk_size:
            padding_shape = (num_samples, chunk_size - (end - start), num_features, 1)
            chunk = np.pad(chunk, [(0, 0), (0, padding_shape[1]), (0, 0), (0, 0)], mode='constant')
        chunks.append(chunk)
    return np.array(chunks)


def get_moe_prediction(input_data, experts, gating_network, chunk_size=44):
    """Compute predictions using Mixture of Experts (MoE) approach."""
    total_length = input_data.shape[1]
    num_chunks = (total_length + chunk_size - 1) // chunk_size  # Ceiling division

    # Step 1: Split input data into chunks
    chunks = split_into_chunks(input_data, chunk_size)  # Shape: (num_chunks, num_samples, chunk_size, num_features, 1)

    # Step 2: Compute gating weights for the complete input data
    gating_weights = gating_network.predict(input_data)  # Shape: (num_samples, num_experts)

    # Step 3: Get predictions from each expert for each chunk
    expert_predictions = []
    for i in range(num_chunks):
        chunk_predictions = np.array([expert.predict(chunks[i]) for expert in experts])
        expert_predictions.append(chunk_predictions)
    expert_predictions = np.array(expert_predictions)  # Shape: (num_chunks, num_experts, num_samples, num_classes)

    # Step 4: Reshape expert_predictions to match gating_weights shape
    expert_predictions = expert_predictions.transpose(2, 0, 1,
                                                      3)  # Shape: (num_samples, num_chunks, num_experts, num_classes)
    # Step 5: Perform weighted sum of predictions
    weighted_predictions = np.einsum('il,iklm->ilm', gating_weights,
                                     expert_predictions)  # Shape: (num_samples, num_experts, num_classes)

    # Step 7: Average the predictions over all chunks
    weighted_predictions = weighted_predictions.mean(axis=1)

    return weighted_predictions


def split_into_batches(X, batch_size):
    """Split the input array into batches of specified size."""
    batches = []
    num_samples = X.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division

    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, num_samples)
        batch = X[start:end]
        batches.append(batch)

    return batches


def generate_performance_labels(experts, X_train, y_train, chunk_size=44, batch_size=32):
    """Generate performance labels for the given training data using expert models."""
    num_samples = X_train.shape[0]
    num_experts = len(experts)
    performance_matrix = np.zeros((num_samples, num_experts))

    chunks = split_into_chunks(X_train, chunk_size)
    all_predictions = np.zeros((num_samples, y_train.shape[1]))

    for i, expert in enumerate(experts):
        for chunk in chunks:
            batches = split_into_batches(chunk, batch_size)
            batch_start_idx = 0

            for batch in batches:
                current_batch_size = batch.shape[0]
                batch_predictions = expert.predict(batch)
                all_predictions[batch_start_idx:batch_start_idx + current_batch_size] += batch_predictions
                batch_start_idx += current_batch_size

        # Normalize predictions by the number of chunks to get the average prediction
        all_predictions /= len(chunks)

        accuracy = np.mean(np.argmax(all_predictions, axis=1) == np.argmax(y_train, axis=1))
        performance_matrix[:, i] = accuracy

    labels = np.argmax(performance_matrix, axis=1)
    first_level_labels_categorical = to_categorical(labels, num_classes=num_experts)
    return first_level_labels_categorical
