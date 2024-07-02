

from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_preprocessing_com2 import DataPreprocessing
from cnnClassifier.entity.config_entity import (DataPreprocessingConfig)
from cnnClassifier import logger

STAGE_NAME = "Data Preprocesssing Stage"


class DataPreprocessingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_processing_config = config.get_data_preprocessing()
        data_preprocessing = DataPreprocessing(config=data_processing_config)
        data_preprocessing.directory_creation()
        data_preprocessing.split_data()