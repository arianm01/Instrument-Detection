from typing import Dict, List
from dataclasses import dataclass
from pathlib import Path
import numpy as np

class Config:
    """Configuration class for model constants and mappings."""
    
    LABEL_MAPPING: Dict[int, int] = {i: i for i in range(5)}
    LABEL_MAPPING_DASTGAH: Dict[int, int] = {i: i for i in range(7)}
    ALL_LABEL_MAPPING: Dict[int, int] = {i: i for i in range(15)}
    
    DASTGAHS: List[str] = [
        'Shour', 'Segah', 'Mahour', 'Homayoun', 
        'RastPanjgah', 'Nava', 'Chahargah'
    ]
    
    CLASSES: List[str] = ['Tar', 'Kamancheh', 'Santur', 'Setar', 'Ney', '']
    ALL_CLASSES: List[str] = [
        'Daf', 'Divan', 'Dutar', 'Gheychak', 'Kamancheh', 
        'Ney', 'Ney Anban', 'Oud', 'Qanun', 'Rubab', 'Santur',
        'Setar', 'Tanbour', 'Tar', 'Tonbak', ''
    ] 

@dataclass
class ModelPaths:
    """Paths for model files."""
    ensemble_model: Path
    base_models: List[Path]
    base_model: Path

@dataclass
class PredictionResults:
    """Container for prediction results."""
    true_labels: List[int]
    predicted_labels: List[int]
    class_names: List[str] 