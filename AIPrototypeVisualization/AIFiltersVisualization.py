# This script visualizes the pattern that the filters or layers respond to. Sources: "https://keras.io/examples/vision/visualizing_what_convnets_learn/"
# and  "https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/" and 
# "https://www.kaggle.com/vincentman0403/visualize-output-of-cnn-filter-by-gradient-ascent" and "https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030"
#%%
import numpy as np
import tensorflow as tf
from scipy.ndimage.filters import median_filter
from tensorflow import keras
import cv2
import os
from keras.models import load_model

# Definitions
dir_name = os.path.dirname(__file__)
model_name = 'SDGModel_2-E100_segmented'
image_dir = os.path.join(dir_name,'visualize')
model_dir = os.path.join(dir_name,'..','Trained_Models', model_name)

# Load model
model = load_model(model_dir)
model.summary()

# The dimensions of our input image
img_width = 256
img_height = 256
start_height = 64
start_width = 64
# Our target layer: we will visualize the filters from this layer.
# See `model.summary()` for list of layer names, if you want to change this.
layer_name = "conv2d_5"

figures_dir = os.path.join(dir_name,'figures',model_name,layer_name)
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

# Set up a model that returns the activation values for our target layer
layer = model.get_layer(name=layer_name)
feature_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)

def compute_loss(input_image, filter_index):
    activation = feature_extractor(input_image)#(cv2.resize(input_image,(img_width,img_height),interpolation = cv2.INTER_CUBIC))
    # We avoid border artifacts by only involving non-border pixels in the loss.
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return tf.reduce_mean(filter_activation)

@tf.function
def gradient_ascent_step(img, filter_index, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, filter_index)
    # Compute gradients.
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    img += learning_rate * grads
    #img = median_filter(img,size=(3,3,1,1))
    return loss, img

def initialize_image():
    # We start from a gray image with some random noise
    img = tf.random.uniform((1, img_width, img_height, 3)) # np.uint8(np.random.uniform(-0.25, 0.25, (start_width, start_height, 3))) # 
    # SDGModel expects inputs in the range [-1, 1].
    # Here we scale our random inputs to [X, X]
    return (img - 0.5) * 0.25

def visualize_filter(filter_index):
    # We run gradient ascent for 20 steps
    iterations = 500
    learning_rate = 10.0
    img = initialize_image()
    for iteration in range(iterations):
        loss, img = gradient_ascent_step(img, filter_index, learning_rate)

    # Decode the resulting input image
    img = deprocess_image(img[0].numpy())
    return loss, img


def deprocess_image(img):
    # Normalize array: center on 0., ensure variance is 0.15
    img -= img.mean()
    img /= img.std() + 1e-5
    img *= 0.15

    # Center crop
    img = img[25:-25, 25:-25, :]

    # Clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)

    # Convert to RGB array
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")
    return img

# Compute image inputs that maximize per-filter activations
# for the first 64 filters of our target layer
all_imgs = []
for filter_index in range(16):
    print("Processing filter %d" % (filter_index,))
    loss, img = visualize_filter(filter_index)
    all_imgs.append(img)

# Build a black picture with enough space for
# our 8 x 8 filters of size 128 x 128, with a 5px margin in between
margin = 5
n = 4
m = 4
cropped_width = img_width - 25 * 2
cropped_height = img_height - 25 * 2
width = n * cropped_width + (n - 1) * margin
height = m * cropped_height + (m - 1) * margin
stitched_filters = np.zeros((width, height, 3))

# Fill the picture with our saved filters
for i in range(n):
    for j in range(m):
        img = all_imgs[i * m + j]
        stitched_filters[
            (cropped_width + margin) * i : (cropped_width + margin) * i + cropped_width,
            (cropped_height + margin) * j : (cropped_height + margin) * j
            + cropped_height,
            :,
        ] = img
keras.preprocessing.image.save_img(figures_dir +  '/'  + "stitched_filters.png", stitched_filters)

from IPython.display import Image, display

display(Image(figures_dir +  '/'  + "stitched_filters.png"))
