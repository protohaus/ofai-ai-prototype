# %% [code]
import cv2 #OpenCV2
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image_dataset_from_directory
from keras.models import load_model
from keras import callbacks
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import sys
import PIL
import PIL.Image
from datetime import datetime

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Utilities import SDGUtilities
from AIPrototypeTrain import ModelDefinitions

# %% [code]
dir_name = os.path.dirname(__file__)
data_dir = os.path.join(dir_name,'..', 'data_n_1500_c_12','train','segmented')

batch_size = 64
img_height = 256
img_width = 256
depth=3

EPOCHS = 2
INIT_LR = 1e-3

mdl = 'SDG_l6_k3_f16_d1024'
#if mdl == 0:
#    save = 'Tairu-E25'
#else:
#    save = 'SDGModel_2-E100_segmented'

# Prepare a directory to store all the checkpoints.
checkpoint_dir = os.path.join(dir_name,'ckpt')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Prepare log directory for tensorboard
log_dir = os.path.join(dir_name,'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# %% [code]
img_gen = ImageDataGenerator(preprocessing_function=SDGUtilities.preprocess_image,rotation_range = 180,width_shift_range = 0.1, 
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

def make_or_restore_model(mdl):
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    # TODO: adapt to multiple models, at the moment there is no differentiation between the models
    checkpoints = [checkpoint_dir + '/' + name
                   for name in os.listdir(checkpoint_dir) if name == mdl]
    if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            print('Restoring from', latest_checkpoint)
            return load_model(latest_checkpoint)
    print('Creating a new model')       
    optimizer = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    return ModelDefinitions.make_model(mdl, num_classes = len(img_flow_train.class_indices), 
                input_shape = (img_height, img_width, depth), opt = optimizer)

model = make_or_restore_model(mdl)

model.build((None, 256, 256, 3))

model.summary()

model_callbacks = [
    # We include the training loss in the folder name.
    callbacks.ModelCheckpoint(
        filepath=checkpoint_dir + '/' + mdl,
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        mode="auto",
        save_freq="epoch"),
    callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1, 
        write_graph=True, 
        write_images=True,
        update_freq='epoch', 
        profile_batch=2
    )#, no early stops at the moment
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
plt.savefig(dir_name + '/ckpt/' + mdl + '/' + mdl + '_accuracy.png')

plt.figure()
#Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()
plt.savefig(dir_name + '/ckpt/' + mdl + '/' +mdl + '_Loss.png')