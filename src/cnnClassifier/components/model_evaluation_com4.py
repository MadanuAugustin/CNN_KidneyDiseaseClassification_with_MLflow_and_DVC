

import tensorflow as tf
from pathlib import Path
import mlflow
import os
import mlflow.keras
import mlflow.sklearn
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories, save_json
from src.cnnClassifier import logger
from tensorflow.keras.models import load_model

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    
    def _valid_generator(self):

        logger.info(f'----------Entered _valid-generator function----------------')

        datagenerator_kwargs = dict(
            rescale = 1./255,
            # validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            class_mode='binary'
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.val_data,
            # subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        logger.info(f'---------------Existed _valid-generator function----------------------')





    
    def log_into_mlflow(self):
        
        logger.info(f'-----------Entered log_into_mlflow function-----------------')

        model = load_model(self.config.path_of_model)

        logger.info(f'------------Model loaded successfully--------------------')

        self._valid_generator()

        score = model.evaluate(self.valid_generator)

        os.environ["MLFLOW_TRACKING_URI"]='https://dagshub.com/augustin7766/VGG-16_KidneyDiseaseClassification_with_MLflow_and_DVC.mlflow'
        os.environ["MLFLOW_TRACKING_USERNAME"]="augustin7766"
        os.environ["MLFLOW_TRACKING_PASSWORD"]="8a01ee4bec043666cf3ced22edc7d308526b4b42"


        mlflow.set_experiment('01_first_exp')
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": score[0], "accuracy": score[1]}
            )

            mlflow.sklearn.log_model(model, 'model', registered_model_name = 'VGG-16')
        

        save_json(path = Path(self.config.metric_file_name), data = {"loss": score[0], "accuracy": score[1]})

        logger.info(f'-----------params and metrics logged successfully--------------')

        logger.info(f'-------------Existed log_into_mlflow function-------------------')

        logger.info(f'------The loss of the model is : {score[0]} and accuracy is {score[1]}--------------------')
