# Here, models are defined. Each model is a class that can be imported by the SDGModelTrainer.
from keras.models import Sequential
from keras import layers
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K

def make_model(model_key, num_classes, input_shape, opt):
    #num_classes = len(img_flow_train.class_indices)

    models = {}
    #inputShape = (img_height, img_width, depth)
    # Not necessary for TensorFlow
    chanDim = -1
    if K.image_data_format() == "channels_first":
        inputShape = (depth, img_height, img_width)
        chanDim = 1

    name = "SDG_l5_k3_f32_d1024_do_bn"
    models[name] = Sequential()
    #models[0].add(layers.experimental.preprocessing.Rescaling(1./255, input_shape=inputShape))
    # First layer
    models[name].add(Conv2D(32, (3, 3), padding="same"))
    models[name].add(Activation("relu"))
    models[name].add(BatchNormalization(axis=chanDim))
    models[name].add(MaxPooling2D(pool_size=(3, 3)))
    models[name].add(Dropout(0.25))

    # Second layer
    models[name].add(Conv2D(64, (3, 3), padding="same"))
    models[name].add(Activation("relu"))
    models[name].add(BatchNormalization(axis=chanDim))

    # Third Layer
    models[name].add(Conv2D(64, (3, 3), padding="same"))
    models[name].add(Activation("relu"))
    models[name].add(BatchNormalization(axis=chanDim))
    models[name].add(MaxPooling2D(pool_size=(2, 2)))
    models[name].add(Dropout(0.25))

    # Fourth Layer
    models[name].add(Conv2D(128, (3, 3), padding="same"))
    models[name].add(Activation("relu"))
    models[name].add(BatchNormalization(axis=chanDim))

    # Fifth Layer
    models[name].add(Conv2D(128, (3, 3), padding="same"))
    models[name].add(Activation("relu"))
    models[name].add(BatchNormalization(axis=chanDim))
    models[name].add(MaxPooling2D(pool_size=(2, 2)))
    models[name].add(Dropout(0.25))
    # Flattens the input
    models[name].add(Flatten())

    models[name].add(Dense(1024))
    models[name].add(Activation("relu"))
    models[name].add(BatchNormalization())
    models[name].add(Dropout(0.5))
    models[name].add(Dense(num_classes))
    models[name].add(Activation("softmax"))
    
    #############################################################################################
    #Begin of second model
    
    name = 'SDG_l6_k3_f16_d1024'
    models[name] = Sequential()
    # Layer 1
    #models[1].add(layers.experimental.preprocessing.Rescaling(1./255, input_shape=inputShape))
    models[name].add(Conv2D(16, (3, 3), padding="same"))
    models[name].add(Activation("relu"))
    #models[1].add(MaxPooling2D(pool_size=(2, 2)))
    
    # Layer 2
    models[name].add(Conv2D(16, (3, 3), padding="same"))
    models[name].add(Activation("relu"))
    models[name].add(MaxPooling2D(pool_size=(2, 2)))
    
    # Layer 3
    models[name].add(Conv2D(32, (3, 3), padding="same"))
    models[name].add(Activation("relu"))
    #models[1].add(MaxPooling2D(pool_size=(2, 2)))
    
    # Layer 4
    models[name].add(Conv2D(32, (3, 3), padding="same"))
    models[name].add(Activation("relu"))
    models[name].add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 5
    models[name].add(Conv2D(64, (3, 3), padding="same"))
    models[name].add(Activation("relu"))
    models[name].add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 6
    models[name].add(Conv2D(64, (3, 3), padding="same"))
    models[name].add(Activation("relu"))
    models[name].add(MaxPooling2D(pool_size=(2, 2)))
    
    models[name].add(Flatten())
    
    models[name].add(Dense(1024))
    models[name].add(Activation("relu"))
    models[name].add(Dense(num_classes))
    models[name].add(Activation("softmax"))

    models[model_key].compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return models[model_key]