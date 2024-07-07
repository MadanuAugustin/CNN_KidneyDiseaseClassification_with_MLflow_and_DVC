

from cnnClassifier.constants import *
import os
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import (DataIngestionConfig, DataPreprocessingConfig, ModelTrainingConfig, EvaluationConfig)
import tensorflow as tf

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    

    
    def get_data_preprocessing(self) -> DataPreprocessingConfig:
        config = self.config.data_preprocessing

        create_directories([config.root_dir])

        data_preprocessing_config = DataPreprocessingConfig(
            root_dir = config.root_dir,
            original_dataset_dir = config.original_dataset_dir,
            train_data = config.train_data,
            val_data = config.val_data
        )

        return data_preprocessing_config
    


    def get_model_training(self) -> ModelTrainingConfig:
        config = self.config.model_training
        params = self.params

        create_directories([config.root_dir])

        model_training_config = ModelTrainingConfig(
            root_dir = config.root_dir,
            train_data= config.train_data,
            val_data= config.val_data,
            base_model_path= config.base_model_path,
            updated_base_model_path= config.updated_base_model_path,
            params_augmentation = params.AUGMENTATION,
            params_image_size = params.IMAGE_SIZE,
            params_batch_size = params.BATCH_SIZE,
            params_include_top = params.INCLUDE_TOP,
            params_epochs = params.EPOCHS,
            params_classes = params.CLASSES,
            params_weights = params.WEIGHTS,
            params_learning_rate = params.LEARNING_RATE
        )

        return model_training_config
    

    def get_evaluation_config(self) -> EvaluationConfig:

        config = self.config.model_evaluation
        params = self.params

        create_directories([config.root_dir])
        
        eval_config = EvaluationConfig(
            root_dir = config.root_dir,
            path_of_model = config.path_of_model,
            metric_file_name  = config.metric_file_name,
            val_data = config.val_data,
            mlflow_uri = "https://dagshub.com/augustin7766/VGG-16_KidneyDiseaseClassification_with_MLflow_and_DVC.mlflow",
            all_params = params,
            params_image_size = params.IMAGE_SIZE,
            params_batch_size = params.BATCH_SIZE
        )

        return eval_config

    

