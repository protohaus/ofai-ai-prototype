import os
from keras.models import load_model
from tensorflow.python.keras import layers
from keras import Model
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K

# Definitions
dir_name = os.path.dirname(__file__)
model_name = 'SDGModel_2-E100_segmented'
model_dir = os.path.join(dir_name,'..','Trained_Models', model_name)

# Load model
model = load_model(model_dir)

model.summary()

# Prepare log directory for tensorboard
log_dir = os.path.join(dir_name,'transfer_logs')

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

print(len( model.layers))

for i in range(5):
    model.layers[i].trainable = False
    print(i)

for i in range(5,8):
    model.layers[i].trainable = True

ll = model.layers[7].output
ll = Dense(32)(ll)
ll = Dense(64)(ll)
ll = Dense(2,activation="softmax")(ll)

new_model = Model(inputs=model.input,outputs=ll)

new_model.summary()