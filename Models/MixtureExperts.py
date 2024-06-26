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
    x = MaxPooling2D((2, 2))(x)
    x = LayerNormalization(epsilon=1e-6)(x)
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


def get_moe_prediction(input_data, experts, gating_network):
    # Step 1: Compute gating weights
    gating_weights = gating_network.predict(input_data)

    # Step 2: Get predictions from each expert
    expert_predictions = np.array([expert.predict(input_data) for expert in experts])

    # Step 3: Compute weighted predictions
    # Note: gating_weights has shape (num_samples, num_experts)
    #       expert_predictions has shape (num_experts, num_samples, num_classes)
    #       weighted_predictions will have shape (num_samples, num_classes)
    weighted_predictions = np.einsum('ij,jik->ik', gating_weights, expert_predictions)

    return weighted_predictions


def generate_performance_labels(experts, X_train, y_train, batch_size=32):
    num_samples = X_train.shape[0]
    num_experts = len(experts)

    performance_matrix = np.zeros((num_samples, num_experts))

    for i, expert in enumerate(experts):
        predictions = np.zeros((num_samples, y_train.shape[1]))

        # Make predictions in batches
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_predictions = expert.predict(X_train[start_idx:end_idx])
            predictions[start_idx:end_idx] = batch_predictions

        accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_train, axis=1))
        performance_matrix[:, i] = accuracy

    labels = np.argmax(performance_matrix, axis=1)
    first_level_labels_categorical = to_categorical(labels, num_classes=len(experts))
    return first_level_labels_categorical
