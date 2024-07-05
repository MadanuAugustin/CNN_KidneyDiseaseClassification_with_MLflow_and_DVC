

from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_training_com3 import ModelTraining
from cnnClassifier import logger

STAGE_NAME = "Data Ingestion stage"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_training_config = config.get_model_training()
        model_training = ModelTraining(config=model_training_config)
        model_training.training_model()
        