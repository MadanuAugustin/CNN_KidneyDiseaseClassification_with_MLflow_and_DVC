

from src.cnnClassifier.entity.config_entity import TrainingConfig
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from ExceptionFile.exception import CustomException
import sys
import tensorflow
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img, save_img
import os
from pathlib import Path
from keras.preprocessing import image
import numpy as np
from keras.applications.vgg16 import preprocess_input
from PIL import Image


class Training:
    def __init__(self, config : TrainingConfig):
        self.config = config


    def preprocess_and_save_images(self):
        source_dir = self.config.training_data
        dest_dir = self.config.preprocessed_data
    # Check if destination directory exists, if not, create it
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

    # Walk through the source directory
        for subdir, dirs, files in os.walk(source_dir):
            for filename in files:
            # Check if the file is an image (simple filter by extension)
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Form the full file path
                    file_path = os.path.join(subdir, filename)
                # Load image
                    img = image.load_img(file_path, target_size=(224, 224))
                # Convert image to array
                    img_array = image.img_to_array(img)
                # Expand dimensions to match the VGG16 model input shape
                    img_array = np.expand_dims(img_array, axis=0)
                # Preprocess the image
                    processed_img = preprocess_input(img_array)

                # Determine folder structure in destination directory
                    relative_path = os.path.relpath(subdir, source_dir)
                    save_dir = os.path.join(dest_dir, relative_path)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                
                # Save the processed image
                    save_path = os.path.join(save_dir, filename)

                    img_to_save = Image.fromarray(np.uint8(processed_img[0]))
                    
                    img_to_save.save(save_path)
                


    
    # def image_preprocessing(self):
    #     try:

    #         original_directory = self.config.training_data
    #         resized_directory = self.config.resized_data
    #         image_size = (224, 224, 3)

    #         os.makedirs(self.config.resized_data, exist_ok=True)

    #         subdirectories = [sub for sub in os.listdir(original_directory) if os.path.isdir(os.path.join(original_directory, sub))]

    #         for subdir in subdirectories:

    #             image_filenames = os.listdir(os.path.join(original_directory, subdir))

    #             resized_subdir = os.path.join(resized_directory, subdir)

    #             os.makedirs(resized_subdir, exist_ok= True)

    #         for filename in image_filenames:

    #             img = load_img(os.path.join(original_directory, subdir, filename), target_size = image_size)

    #             resized_image_path = os.path.join(resized_subdir, filename)

    #             save_img(resized_image_path, img)

    #             print(f'Resized image saved at : {resized_image_path}')

    #     except Exception as e:
    #         raise CustomException(e, sys)
        
        ###########################################################################################################################
        


    
    # the below function loads the model from the artifacts
        

    # def get_updated_base_model(self):
    #     self.model = tensorflow.keras.models.load_model(
    #         self.config.updated_base_model_path
    #     )


    # # the below function generates train and test data
    # # refer the keras documentation for train_valid_generator
    #     data_augmentation = ImageDataGenerator(
    #         rescale = 1./255.0,
    #         # shear_range = 0.2,
    #         # zoom_range = 0.2,
    #         # horizontal_flip = True,
    #         # # vertical_flip = True,
    #         # validation_split = 0.2,
    #         # rotation_range=40,
    #         # width_shift_range=0.2,
    #         # height_shift_range=0.2
    #     )
    #     ######################################################################################
    #     # we are using the keras image generator to split train-test
    #     # we can also perform manually using train_test_split function
    #     train_generator = data_augmentation.flow_from_directory(
    #         self.config.resized_data,
    #         target_size =self.config.params_image_size,
    #         batch_size = self.config.params_batch_size,
    #         subset='training',
    #         shuffle=True,
    #         interpolation="bilinear"
    #     )
        #########################################################################################

        # performing the data-agumentation
    #     test_generator = data_augmentation.flow_from_directory(
    #         self.config.resized_data,
    #         target_size = self.config.params_image_size,
    #         batch_size = self.config.params_batch_size,
    #         subset = 'validation',
    #         shuffle=False,
    #         interpolation="bilinear"
    #     )

    #     return train_generator, test_generator
    
    # #################################################################################################


    # def train(self, train_generator, test_generator):
    #     self.training_steps = train_generator.samples // self.config.params_batch_size
    #     self.validation_steps = test_generator.samples // self.config.params_batch_size

    #     self.model.fit(
    #         train_generator,
    #         epochs = self.config.params_epochs,
    #         steps_per_epoch = self.training_steps,
    #         validation_steps = self.validation_steps,
    #         validation_data = test_generator
    #     )

    #     tensorflow.keras.models.save_model(model=self.model, filepath=self.config.trained_model_path)


