import tensorflow as tf
import keras
from keras import layers, models
from keras.models import Sequential
import logging

from keras.layers import RandomFlip, RandomRotation
from keras.layers import Conv2D, BatchNormalization, Dropout, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dense

logging.basicConfig(level=logging.INFO)



def build_model(
        class_count: int,

        input_height: int,
        input_width: int,
        input_channel: int,

        use_augmentation: bool,

        conv_set: int,
        conv_filter: int,
        conv_kernel: int,

        use_batch_normalization: bool,  
   
        use_dropout: bool,
        dropout_rate: float,

        use_max_pooling_2d: bool,
        
        final_dense_neuron: int
    ) -> Sequential:
    """
        Build a Convolutional Neural Network (CNN) for image classification using TensorFlow Keras.

        The model allows flexibility in defining the architecture with the following parameters:
        - Whether to apply image augmentation layer
            - RandomFlip both horizontally and vertically
            - Random rotation within 0.2 radian
        - Number of sets of Conv2D layers
            - Number of filters and kernel size of the layer
            - Whether to apply Batch Normalization after each Conv2D layer
            - Whether to apply Dropout after each Conv2D layer
                - The dropout rate of the Dropout layer
            - Whether to apply MaxPooling2D layer after each Conv2D layer
        - Number of neurons in the last hidden layer of the model

        IMPORTANT NOTE: The value of conv_filter and final_dense_neuron are the base exponent of 2.
    """

    if class_count < 2:
        raise ValueError("Function build_model aborted: Provided class count is not applicable for classification task.")

    if input_height <= 0 or input_width <= 0 or input_channel <= 0:
        raise ValueError("Function build_model aborted: Input dimension and channel must be positive numbers.")
    
    if conv_set < 1:
        raise ValueError(f"Function build_model aborted: Model should contain at least 1 Conv2D block set.")
    
    if conv_filter < 1 or final_dense_neuron < 1:
        raise ValueError("Function build_model aborted: Filter depth and dense neuron counts must be greater than zero.")
    
    if conv_kernel < 1 or conv_kernel % 2 == 0:
        raise ValueError("Function build_model aborted: Kernel sizes should be positive odd integers.")
        
    if use_dropout:
        if dropout_rate is None or not (0.0 <= dropout_rate <= 0.5):
            raise ValueError("Model generation aborted: Dropout rate must be bounded between 0.0 and 0.5 to prevent dead nodes.")

    keras.backend.clear_session()

    model = keras.Sequential()
    model.add(layers.Input(shape = (input_height, input_width, input_channel)))
    
    if use_augmentation:
        model.add(RandomFlip("horizontal_and_vertical"))
        model.add(RandomRotation(0.2))
    
    for i in range(conv_set):
        model.add(Conv2D(
            filters = 2 ** (conv_filter + i), 
            kernel_size = conv_kernel, 
            activation = "relu"
        ))

        if use_batch_normalization:
            model.add(BatchNormalization())
        
        if use_dropout:
            model.add(Dropout(dropout_rate))
        
        if use_max_pooling_2d:
            model.add(MaxPooling2D())
    
    model.add(GlobalAveragePooling2D())
    model.add(Dense(2 ** final_dense_neuron, activation = "relu"))
    model.add(Dense(class_count, activation = "softmax"))

    return model