


from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path



@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir : Path
    original_dataset_dir : Path
    train_data : Path
    val_data : Path


@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir : Path
    train_data : Path
    val_data : Path
    base_model_path : Path
    updated_base_model_path : Path
    params_augmentation : bool
    params_image_size : list
    params_batch_size : int
    params_include_top : bool
    params_epochs : int
    params_classes : int 
    params_weights : str
    params_learning_rate : float

    