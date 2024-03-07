

import sys
from cnnClassifier import logger
from ExceptionFile.exception import CustomException
from cnnClassifier.pipeline.stage_01_DataIngestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline



STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        raise CustomException(e , sys)



STAGE_NAME = 'Prepare base Model'

try:
   logger.info(f'---------{STAGE_NAME} started')
   prepare_base_model = PrepareBaseModelTrainingPipeline()
   prepare_base_model.main()
   logger.info(f'-------------{STAGE_NAME} completed----------------')
except Exception as e:
     raise CustomException(e, sys)