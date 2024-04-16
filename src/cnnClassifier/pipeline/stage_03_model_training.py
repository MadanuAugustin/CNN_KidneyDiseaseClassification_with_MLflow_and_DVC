
from src.cnnClassifier.config.configuration import ConfigurationManager
# from src.cnnClassifier.components.prepare_callbacks import PrepareCallback
from src.cnnClassifier.components.model_training import Training
from src.cnnClassifier import logger



STAGE_NAME = 'Model_training'

class ModelTrainingPipeline:
    def __init__(self):
        pass


    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config = training_config)
        training.preprocess_and_save_images()
        # train_generator, test_generator = training.get_updated_base_model()
        # training.train(train_generator, test_generator)
        
        