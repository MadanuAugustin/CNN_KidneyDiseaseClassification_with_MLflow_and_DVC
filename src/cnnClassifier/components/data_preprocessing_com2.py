
import os
import sys
from src.cnnClassifier import logger
from cnnClassifier.entity.config_entity import (DataPreprocessingConfig)
from ExceptionFile.exception import CustomException
import shutil
from sklearn.model_selection import train_test_split





class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config


    def directory_creation(self):
        try:
            logger.info('-----------Entered data-splitting function--------')
            # Paths to the new directories
            base_dir = self.config.root_dir
            train_dir = os.path.join(base_dir, 'train')
            validation_dir = os.path.join(base_dir, 'val')

            # Create directories
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(validation_dir, exist_ok=True)

            Tumor_train_dir = os.path.join(train_dir, 'Tumor')
            Tumor_val_dir = os.path.join(validation_dir, 'Tumor')
            Normal_train_dir = os.path.join(train_dir, 'Normal')
            Normal_val_dir = os.path.join(validation_dir, 'Normal')

            os.makedirs(Tumor_train_dir, exist_ok=True)
            os.makedirs(Tumor_val_dir, exist_ok=True)
            os.makedirs(Normal_train_dir, exist_ok=True)
            os.makedirs(Normal_val_dir, exist_ok=True)
            logger.info(f'-----------existed data-splitting function-------------')

        except Exception as e:
            raise CustomException(e, sys)


    
    # Helper function to split data
    def split_data(self, class_name, src_dir, train_dir, val_dir, val_size=0.2):
        try:
            logger.info(f'-------------Entered split_data function---------------')
            # obj = self.directory_creation()
            src_class_dir = os.path.join(src_dir, class_name)
            filenames = os.listdir(src_class_dir)
    
            train_filenames, val_filenames = train_test_split(filenames, test_size=val_size, random_state=42)

            logger.info(f'--------The shape of train_data is {train_filenames} and validation data is{val_filenames}')

            print(len(train_filenames), len(val_filenames))
    
            for filename in train_filenames:
                src_file = os.path.join(src_class_dir, filename)
                dst_file = os.path.join(train_dir, class_name, filename)
                shutil.copyfile(src_file, dst_file)
    
            for filename in val_filenames:
                src_file = os.path.join(src_class_dir, filename)
                dst_file = os.path.join(val_dir, class_name, filename)
                shutil.copyfile(src_file, dst_file)
            logger.info(f'----------------Existed split_data function-------------------')
        except Exception as e:
            raise CustomException(e, sys)
