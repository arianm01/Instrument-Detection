import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from keras.saving.save import load_model
from keras.utils import to_categorical

from src.Infrence.InfrenceInstrument import load_files, extract_label, preprocess_audio
from src.main.main import TIME_FRAME, MERGE_FACTOR, ensemble_learning
from src.utility.utils import get_model_feature

# Constants
SAMPLE_RATE = 22050
DATASET_BASE_PATH = Path('../../../../../archive/NavaDataset')
MODEL_BASE_PATH = Path('../../results/Dastgah/1 sec')
NUM_CHUNKS = 5

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_nava_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess the Nava dataset.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Processed audio features and their labels
    """
    files = load_files(DATASET_BASE_PATH / 'all_train.txt')
    features, labels = [], []

    for file in files:
        true_label = extract_label(file)
        logger.info(f"Loading {file} with label {true_label}")

        segments = preprocess_audio(
            DATASET_BASE_PATH / 'Data' / f'{file}.mp3',
            step_size=4 * TIME_FRAME * SAMPLE_RATE,
            segment_duration=TIME_FRAME * MERGE_FACTOR * SAMPLE_RATE
        )

        features.extend(np.array(segments, dtype=np.float32))
        labels.extend([true_label] * len(segments))

    return np.array(features, dtype=np.float32), to_categorical(labels)


def split_list_into_arrays(
        data: List,
        models: List,
        num_chunks: int = NUM_CHUNKS
) -> np.ndarray:
    """
    Split data into chunks and process with models.
    
    Args:
        data: List of data to process
        models: List of trained models
        num_chunks: Number of chunks to split data into
    
    Returns:
        np.ndarray: Processed and concatenated features
    """
    chunk_size = len(data) // num_chunks
    array_chunks = [
        get_model_feature(
            np.array(data[i * chunk_size:(i + 1) * chunk_size])[..., np.newaxis],
            models
        )
        for i in range(num_chunks)
    ]

    return np.concatenate(array_chunks, axis=0)[..., np.newaxis]


def load_pretrained_models() -> List:
    """Load pretrained models from disk."""
    models = []
    for i in range(1, 6):
        model_path = MODEL_BASE_PATH / f'model_best_classifier_{i}.keras'
        try:
            models.append(load_model(model_path))
        except Exception as e:
            logger.error(f"Failed to load model {i}: {e}")
            raise
    return models


def main():
    """Main execution function."""
    try:
        # Load and preprocess data
        features, labels = load_nava_data()

        # Load pretrained models
        models = load_pretrained_models()

        # Perform ensemble learning
        ensemble_learning(features, labels, models)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


if __name__ == '__main__':
    main()
