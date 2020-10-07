# %% [code]
import numpy as np # linear algebra
import cv2 #OpenCV2
import tensorflow as tf
from keras.models import Sequential
from keras import layers
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image_dataset_from_directory
from keras.models import load_model
from keras import callbacks
import matplotlib.pyplot as plt
import os
import PIL
import PIL.Image
from datetime import datetime

# %% [code]
dir_name = os.path.dirname(__file__)
data_dir = os.path.join(dir_name,'..', 'data_n_1500_c_12','train','color')

batch_size = 64
img_height = 256
img_width = 256
depth=3

EPOCHS = 50
INIT_LR = 1e-3

mdl = 1
if mdl == 0:
    save = '/Tairu-E25'
else:
    save = '/SDGModel-E50'

# %% [code]
img_gen = ImageDataGenerator(rotation_range = 180,width_shift_range = 0.1, 
                         height_shift_range = 0.1, shear_range = 0.2, zoom_range  = 0.2, brightness_range = [0.5, 1.5], horizontal_flip = True, fill_mode = "nearest",
                            vertical_flip = True, validation_split = 0.2)

# %% [code]
img_flow_train = img_gen.flow_from_directory(data_dir, 
                                             shuffle = True,
                                             batch_size = batch_size, 
                                             target_size=(img_height,img_width),
                                             subset = 'training'
                                            )

img_flow_val = img_gen.flow_from_directory(data_dir, 
                                             shuffle = True,
                                             batch_size = batch_size, 
                                             target_size=(img_height,img_width),
                                             subset = 'validation')

# %% [code]
images, labels = next(img_flow_train)
print(images.shape, labels.shape)
print(img_flow_train.class_indices)

# %% [code]
# Prepare a directory to store all the checkpoints.
checkpoint_dir = './ckpt'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


def make_model(i):
    num_classes = len(img_flow_train.class_indices)

    models = []
    inputShape = (img_height, img_width, depth)
    # Not necessary for TensorFlow
    chanDim = -1
    if K.image_data_format() == "channels_first":
        inputShape = (depth, img_height, img_width)
        chanDim = 1

    models.append(Sequential())
    models[0].add(layers.experimental.preprocessing.Rescaling(1./255, input_shape=inputShape))
    # First layer
    models[0].add(Conv2D(32, (3, 3), padding="same"))
    models[0].add(Activation("relu"))
    models[0].add(BatchNormalization(axis=chanDim))
    models[0].add(MaxPooling2D(pool_size=(3, 3)))
    models[0].add(Dropout(0.25))

    # Second layer
    models[0].add(Conv2D(64, (3, 3), padding="same"))
    models[0].add(Activation("relu"))
    models[0].add(BatchNormalization(axis=chanDim))

    # Third Layer
    models[0].add(Conv2D(64, (3, 3), padding="same"))
    models[0].add(Activation("relu"))
    models[0].add(BatchNormalization(axis=chanDim))
    models[0].add(MaxPooling2D(pool_size=(2, 2)))
    models[0].add(Dropout(0.25))

    # Fourth Layer
    models[0].add(Conv2D(128, (3, 3), padding="same"))
    models[0].add(Activation("relu"))
    models[0].add(BatchNormalization(axis=chanDim))

    # Fifth Layer
    models[0].add(Conv2D(128, (3, 3), padding="same"))
    models[0].add(Activation("relu"))
    models[0].add(BatchNormalization(axis=chanDim))
    models[0].add(MaxPooling2D(pool_size=(2, 2)))
    models[0].add(Dropout(0.25))
    # Flattens the input
    models[0].add(Flatten())

    models[0].add(Dense(1024))
    models[0].add(Activation("relu"))
    models[0].add(BatchNormalization())
    models[0].add(Dropout(0.5))
    models[0].add(Dense(num_classes))
    models[0].add(Activation("softmax"))
    
    #############################################################################################
    #Begin of second model
    
    models.append(Sequential())
    # Layer 1
    models[1].add(layers.experimental.preprocessing.Rescaling(1./255, input_shape=inputShape))
    models[1].add(Conv2D(16, (3, 3), padding="same"))
    models[1].add(Activation("relu"))
    models[1].add(MaxPooling2D(pool_size=(2, 2)))
    
    # Layer 2
    models[1].add(Conv2D(32, (3, 3), padding="same"))
    models[1].add(Activation("relu"))
    models[1].add(MaxPooling2D(pool_size=(2, 2)))
    
    # Layer 3
    models[1].add(Conv2D(64, (3, 3), padding="same"))
    models[1].add(Activation("relu"))
    models[1].add(MaxPooling2D(pool_size=(2, 2)))
    
    # Layer 4
    models[1].add(Conv2D(128, (3, 3), padding="same"))
    models[1].add(Activation("relu"))
    models[1].add(MaxPooling2D(pool_size=(2, 2)))
    
    models[1].add(Flatten())
    
    models[1].add(Dense(1024))
    models[1].add(Activation("relu"))
    models[1].add(Dense(num_classes))
    models[1].add(Activation("softmax"))
    
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

    models[i].compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return models[i]


def make_or_restore_model(mdl):
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    # TODO: adapt to multiple models, at the moment there is no differentiation between the models
    checkpoints = [checkpoint_dir + '/' + name
                   for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print('Restoring from', latest_checkpoint)
        return load_model(latest_checkpoint)
    print('Creating a new model')
    return make_model(mdl)


model = make_or_restore_model(mdl)

model.build((None, 256, 256, 3))

model.summary()

model_callbacks = [
    # We include the training loss in the folder name.
    callbacks.ModelCheckpoint(
        filepath=checkpoint_dir + save,
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        mode="auto",
        save_freq="epoch")#, no early stops at the moment
    #callbacks.EarlyStopping(
    #    monitor="val_accuracy",
    #    patience=3,
    #    verbose=1,
    #    mode="auto")
]

# %% [code]
history = model.fit(
    img_flow_train,
    steps_per_epoch = img_flow_train.samples // batch_size,
    validation_data = img_flow_val, 
    validation_steps = img_flow_val.samples // batch_size,
    epochs = EPOCHS,
    verbose=1,
    workers=8,
    callbacks = model_callbacks)

# %% [code]
# Generate generalization metrics
score = model.evaluate(img_flow_val, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

# %% [code]
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
#Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and Validation accurarcy')
plt.legend()
plt.savefig(dir_name + '/ckpt/' + save + '/' + save + '_accuracy.png')

plt.figure()
#Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()
plt.savefig(dir_name + '/ckpt/' + save + '/' +save + '_Loss.png')