#%%
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import IPython.display as display
import PIL.Image
from tensorflow.keras.preprocessing import image
import os
from keras.models import load_model
import time

url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'

def preprocess_image(image):
    # pixel values between [0,1]
    image = np.divide(image, 255.0)
    # pixel values between [-0.5,0.5]
    image = np.subtract(image, 0.5)
    # pixel values between [-1,1]
    image = np.multiply(image,2.0)
    return image

# Download an image and read it into a NumPy array.
def download(url, max_dim=None):
    name = url.split('/')[-1]
    #image_path = tf.keras.utils.get_file(name, origin=url)
    img = PIL.Image.open(os.path.join(os.path.dirname(__file__),'andromeda-start.jpg'))
    #img = PIL.Image.open(image_path)
    # image conversion to 256x256
    width = img.width
    height = img.height
    # TODO: crop somewhere else
    if width >= height:
        (target_width, target_height) = (int (width // (height / max_dim)) , max_dim)
        box = [(target_width - max_dim) // 2, 0, (target_width - max_dim)//2 + max_dim, max_dim]
    else:
        (target_width, target_height) = (max_dim , int (height // ( width / max_dim)))
        box = [0, (target_height - max_dim) // 2, max_dim, (target_height - max_dim)//2 + max_dim]

    img_resized = img.resize([target_width, target_height])
    img_cropped = img_resized.crop(box)
    #if max_dim:
        #img.thumbnail((max_dim, max_dim))
    return tf.cast(np.array(img_cropped),tf.float32)

# Normalize an image
def deprocess(img):
  img = 255*(img + 1.0)/2.0
  return tf.cast(img, tf.uint8)

# Display an image
def show(img):
  display.display(PIL.Image.fromarray(np.array(img)))


# Downsizing the image makes it easier to work with.
original_img = download(url, max_dim=500)
show(deprocess(original_img))
display.display(display.HTML('Image cc-by: <a "href=https://commons.wikimedia.org/wiki/File:Felis_catus-cat_on_snow.jpg">Von.grzanka</a>'))

# Definitions
dir_name = os.path.dirname(__file__)
model_name = 'SDGModel_2-E100_segmented'
model_dir = os.path.join(dir_name,'..','Trained_Models', model_name)

figures_dir = os.path.join(dir_name,'figures',model_name)
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

# Load model
base_model = load_model(model_dir)
base_model.summary()

# Maximize the activations of these layers
names = ['conv2d_9']
filename = '_andromeda_highres_noloss_conv_9'
layers = [base_model.get_layer(name).output for name in names]

# Create the feature extraction model
dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

def calc_loss(img, model):
    # Pass forward the image through the model to retrieve the activations.
    # Converts the image into a batch of size 1.
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(tf.image.resize(img_batch,(256,256)))
    if len(layer_activations) == 1:
        layer_activations = [layer_activations]

    losses = []
    for act in layer_activations:
        loss = tf.math.reduce_mean(act)
        losses.append(loss)

    loss = tf.reduce_sum(losses)
    #print("Loss:")
    #tf.print(loss)
    total_variation = tf.image.total_variation(img)
    #print("Total Variation:")
    #tf.print(total_variation)
    return loss# - 0.0000005*total_variation

class DeepDream(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None,None,3], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.int32),
            tf.TensorSpec(shape=[], dtype=tf.float32),)
            )
    def __call__(self, img, steps, step_size):
        print("Tracing")
        loss = tf.constant(0.0)
        for n in tf.range(steps):
            with tf.GradientTape() as tape:
                # This needs gradients relative to `img`
                # `GradientTape` only watches `tf.Variable`s by default
                tape.watch(img)
                loss = calc_loss(img, self.model)

            # Calculate the gradient of the loss with respect to the pixels of the input image.
            gradients = tape.gradient(loss, img)

            # Normalize the gradients.
            gradients /= tf.math.reduce_std(gradients) + 1e-8 

            # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
            # You can update the image by directly adding the gradients (because they're the same shape!)
            img = img + gradients*step_size
            img = tf.clip_by_value(img, -1, 1)
        return loss, img

deepdream = DeepDream(dream_model)

def run_deep_dream_simple(img, steps=100, step_size=0.01):
    # Convert from uint8 to the range expected by the model.
    img = preprocess_image(img)
    img = tf.convert_to_tensor(img)
    step_size = tf.convert_to_tensor(step_size)
    steps_remaining = steps
    step = 0
    while steps_remaining:
        if steps_remaining>100:
            run_steps = tf.constant(100)
        else:
            run_steps = tf.constant(steps_remaining)
        steps_remaining -= run_steps
        step += run_steps

        loss, img = deepdream(img, run_steps, tf.constant(step_size))

        display.clear_output(wait=True)
        show(deprocess(img))
        print ("Step {}, loss {}".format(step, loss))


    result = deprocess(img)
    display.clear_output(wait=True)
    show(result)

    return result

dream_img = run_deep_dream_simple(img=original_img, steps=1000, step_size=0.02)
PIL.Image.fromarray(np.array(dream_img)).save(figures_dir + '/' + "Image_Simple_" + filename +".jpg")
#%%
start = time.time()

OCTAVE_SCALE = 1.30

img = tf.constant(np.array(original_img))
base_shape = tf.shape(img)[:-1]
float_base_shape = tf.cast(base_shape, tf.float32)

for n in range(-2, 3):
    new_shape = tf.cast(float_base_shape*(OCTAVE_SCALE**n), tf.int32)

    img = tf.image.resize(img, new_shape).numpy()

    img = run_deep_dream_simple(img=img, steps=500, step_size=0.02)

display.clear_output(wait=True)
# img = tf.image.resize(img, base_shape)
img = tf.image.resize(img, tf.shape(img)[:-1])
img = tf.image.convert_image_dtype(img/255.0, dtype=tf.uint8)
show(img)
PIL.Image.fromarray(np.array(img)).save(figures_dir + '/' + "Image_Octaves_" + filename + ".jpg")
end = time.time()
end-start

# %%
# %%
def random_roll(img, maxroll):
    # Randomly shift the image to avoid tiled boundaries.
    shift = tf.random.uniform(shape=[2], minval=-maxroll, maxval=maxroll, dtype=tf.int32)
    img_rolled = tf.roll(img, shift=shift, axis=[0,1])
    return shift, img_rolled

#shift, img_rolled = random_roll(np.array(original_img), 512)
#show(img_rolled)

class TiledGradients(tf.Module):
  def __init__(self, model):
    self.model = model

  @tf.function(
      input_signature=(
        tf.TensorSpec(shape=[None,None,3], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),)
  )
  def __call__(self, img, tile_size=512):
    shift, img_rolled = random_roll(img, tile_size)

    # Initialize the image gradients to zero.
    gradients = tf.zeros_like(img_rolled)
    
    # Skip the last tile, unless there's only one tile.
    xs = tf.range(0, img_rolled.shape[0], tile_size)[:-1]
    if not tf.cast(len(xs), bool):
      xs = tf.constant([0])
    ys = tf.range(0, img_rolled.shape[1], tile_size)[:-1]
    if not tf.cast(len(ys), bool):
      ys = tf.constant([0])

    for x in xs:
      for y in ys:
        # Calculate the gradients for this tile.
        with tf.GradientTape() as tape:
          # This needs gradients relative to `img_rolled`.
          # `GradientTape` only watches `tf.Variable`s by default.
          tape.watch(img_rolled)

          # Extract a tile out of the image.
          img_tile = img_rolled[x:x+tile_size, y:y+tile_size]
          loss = calc_loss(img_tile, self.model)

        # Update the image gradients for this tile.
        gradients = gradients + tape.gradient(loss, img_rolled)

    # Undo the random shift applied to the image and its gradients.
    gradients = tf.roll(gradients, shift=-shift, axis=[0,1])

    # Normalize the gradients.
    gradients /= tf.math.reduce_std(gradients) + 1e-8 

    return gradients 

get_tiled_gradients = TiledGradients(dream_model)

def run_deep_dream_with_octaves(img, steps_per_octave=500, step_size=0.01, 
                                octaves=range(-2,3), octave_scale=1.3):
  base_shape = tf.shape(img)
  img = tf.keras.preprocessing.image.img_to_array(img)
  img = preprocess_image(img)

  initial_shape = img.shape[:-1]
  img = tf.image.resize(img, initial_shape)
  for octave in octaves:
    # Scale the image based on the octave
    new_size = tf.cast(tf.convert_to_tensor(base_shape[:-1]), tf.float32)*(octave_scale**octave)
    img = tf.image.resize(img, tf.cast(new_size, tf.int32))

    for step in range(steps_per_octave):
      gradients = get_tiled_gradients(img)
      img = img + gradients*step_size
      img = tf.clip_by_value(img, -1, 1)

      if step % 10 == 0:
        display.clear_output(wait=True)
        show(deprocess(img))
        print ("Octave {}, Step {}".format(octave, step))
    
  result = deprocess(img)
  return result

img = run_deep_dream_with_octaves(img=original_img, step_size=0.01)

display.clear_output(wait=True)
#img = tf.image.resize(img, base_shape)
img = tf.image.resize(img, tf.shape(img)[:-1])
img = tf.image.convert_image_dtype(img/255.0, dtype=tf.uint8)
show(img)
PIL.Image.fromarray(np.array(img)).save(figures_dir + '/' + "Image_Octaves_Tiles_" + filename + ".jpg")