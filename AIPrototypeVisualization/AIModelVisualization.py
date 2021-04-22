# Adapted from "https://towardsdatascience.com/visualizing-intermediate-activations-of-a-cnn-trained-on-the-mnist-dataset-2c34426416c8"
# This script visualizes the layers for a specific input.
#%%
import numpy as np # linear algebra
import csv
import cv2 #OpenCV2
from os import listdir
from keras.models import Sequential
from keras.models import Model
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing import image_dataset_from_directory
import os
import gc
import sys
from datetime import datetime
import matplotlib.pyplot as plt

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR,PACKAGE_PARENT)))

from Utilities import SDGUtilities

# Definitions
dir_name = os.path.dirname(__file__)
model_name = 'SDGModel_2-E100_segmented'
image_dir = os.path.join(dir_name,'visualize')
model_dir = os.path.join(dir_name,'..','Trained_Models', model_name)

# Load model

model = load_model(model_dir)

model.summary()

layer_outputs = [layer.output for layer in model.layers[1:16]]
activation_model = Model(inputs=model.input,outputs=layer_outputs)

for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')) == True:
        img = image.load_img(os.path.join(image_dir,filename))
        img.filename = filename
        print(img.filename)

    figures_dir = os.path.join(dir_name,'figures',model_name,filename)
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    image_array = []

    # image conversion to 256x256
    width = img.width
    height = img.height

    if width >= height:
        (target_width, target_height) = (int (width // (height / 256)) , 256)
        box = [(target_width - 256) // 2, 0, (target_width - 256)//2 + 256, 256]
    else:
        (target_width, target_height) = (256 , int (height // ( width / 256)))
        box = [0, (target_height - 256) // 2, 256, (target_height - 256)//2 + 256]

    img_resized = img.resize([target_width, target_height])
    img_cropped = img_resized.crop(box)

    image_array = image.img_to_array(img_cropped)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = SDGUtilities.preprocess_image(image_array)

    # Get Activations
    activations = activation_model.predict(image_array)

    layer_names = []
    for layer in model.layers[1:16]:
        layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
        
    images_per_row = 16
    for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
        n_features = layer_activation.shape[-1] # Number of features in the feature map
        size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols): # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                                :, :,
                                                col * images_per_row + row]
                channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
                std = channel_image.std()
                if abs(std) > 0.0000001:
                    channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size, # Displays the grid
                            row * size : (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.savefig(figures_dir + '/' + layer_name + '.png')
        plt.close()

# %%
