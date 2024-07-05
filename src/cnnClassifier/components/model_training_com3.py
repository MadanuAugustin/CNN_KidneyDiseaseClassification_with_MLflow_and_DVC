
import os
import sys
import math
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from ExceptionFile.exception import CustomException
from src.cnnClassifier import logger
from cnnClassifier.entity.config_entity import (ModelTrainingConfig)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping





class ModelTraining:
    def __init__(self, config : ModelTrainingConfig):
        self.config = config


    def training_model(self):
        try:

            logger.info(f'-----------Entered model_training function------------------')

            datagen = ImageDataGenerator(
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest')


            test_datagen = ImageDataGenerator(rescale=1./255)


            train_generator = datagen.flow_from_directory(   
                self.config.train_data,  # this is the target directory
                target_size=(224, 224),  # all images will be resized to 244x244
                batch_size=self.config.params_batch_size,
                class_mode='binary') 

            logger.info(f'------------train_generator found {train_generator.samples} samples---------------')

            validation_generator = test_datagen.flow_from_directory(               
                self.config.val_data,
                target_size=(224, 224),
                batch_size=self.config.params_batch_size,
                class_mode='binary')

            logger.info(f'------------test_generator found {validation_generator.samples} samples---------------')

            logger.info(f'----------creating base model-----------------------------')

            base_model = VGG16(weights=self.config.params_weights, include_top=self.config.params_include_top, input_shape=self.config.params_image_size)     

            base_model.save(self.config.base_model_path)  

            logger.info(f'---------customizing base model-------------------')

            for layer in base_model.layers:
                layer.trainable = False

            # Adding custom layers
            x = Flatten()(base_model.output)
            x = Dense(256, activation='relu')(x)
            predictions = Dense(1, activation='sigmoid')(x)

            model = Model(inputs=base_model.input, outputs=predictions)

            model.compile(optimizer=Adam(learning_rate=self.config.params_learning_rate), loss='binary_crossentropy', metrics=['accuracy'])


            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                self.config.updated_base_model_path,
                monitor='val_accuracy',  # You can change this to 'val_loss' if you prefer
                save_best_only=True,
                save_weights_only=False,
                mode='max',  # Use 'min' for 'val_loss'
                verbose=1
                )
            

            early_stopping = EarlyStopping(
                monitor='val_accuracy',  # You can change this to 'val_loss' if you prefer
                patience=5,  # Number of epochs with no improvement after which training will be stopped
                mode='max',  # Use 'min' for 'val_loss'
                verbose=1
            )

            logger.info(f'------------model training started---------------')

            model.fit(
                train_generator,
                steps_per_epoch=math.ceil(train_generator.samples // 20),
                epochs=self.config.params_epochs,
                validation_data=validation_generator,
                validation_steps= math.ceil(validation_generator.samples // 20),
                callbacks = [model_checkpoint_callback, early_stopping]
                )

        except Exception as e:
            raise CustomException(e,sys)
