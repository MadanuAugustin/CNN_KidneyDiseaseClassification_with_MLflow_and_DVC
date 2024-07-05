

import sys
from cnnClassifier import logger
from ExceptionFile.exception import CustomException
from cnnClassifier.pipeline.stage_01_DataIngestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_DataPreprocessing import DataPreprocessingPipeline
from cnnClassifier.pipeline.stage_03_ModelTraining import ModelTrainingPipeline

STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        raise CustomException(e , sys)




STAGE_NAME = 'Data Preprocessing stage'

try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_preprocessing = DataPreprocessingPipeline()
   data_preprocessing.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
     raise CustomException(e, sys)




STAGE_NAME = 'Model Training stage'

try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   model_training = ModelTrainingPipeline()
   model_training.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
     raise CustomException(e, sys)


