import numpy as np # linear algebra
import csv
import cv2 #OpenCV2
from os import listdir
from keras.models import Sequential
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing import image_dataset_from_directory
import os
import gc
import sys
from datetime import datetime

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT,PACKAGE_PARENT)))

from ofai_ai_prototype.Utilities import SDGUtilities

# Definitions
dir_name = os.path.dirname(__file__)
model_name = 'SDGModel_2-E100_segmented'
image_dir = os.path.join(dir_name,'images','predict_google')
model_dir = os.path.join(dir_name,'..','Trained_Models', model_name)

#class labels for 12 classes with 1500 images
class_labels = ['Apple___healthy', 'Blueberry___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 
'Soybean___healthy', 'Squash___Powdery_mildew', 'Tomato___Bacterial_spot', 'Tomato___Late_blight', 
'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
'Tomato___healthy']

images = []
for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')) == True:
        img = image.load_img(os.path.join(image_dir,filename))
        img.filename = filename
        print(img.filename)
    if img is not None:
        images.append(img)

images_array = []

for img in images:
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

    x = image.img_to_array(img_cropped)
    x = np.expand_dims(x, axis=0)
    x = SDGUtilities.preprocess_image(x)
    images_array.append(x)

# Load model

model = load_model(model_dir)

model.summary()

# Classify Images
classes = []

for img_array in images_array:
    cls = model.predict(img_array)
    classes.append(cls)

# Save Result
with open(os.path.join(image_dir,'results_' + model_name + '.csv'), 'w', newline='') as csvfile:
    result_writer = csv.writer(csvfile, dialect='excel')
    result_writer.writerow([ 'Image Name', 'Class 1', 'Propability', 'Class 2', 'Propability', 'Class 3', 'Propability'])

    for i, result in enumerate(classes):
        indices = np.flip(np.argsort(result))
        result_writer.writerow([images[i].filename, class_labels[indices[0][0]], str(result[0][indices[0][0]]*100),
                                class_labels[indices[0][1]], str(result[0][indices[0][1]]*100),
                                class_labels[indices[0][2]], str(result[0][indices[0][2]]*100)
                                ])
        print(images[i].filename + ":")
        for index in indices[0][0:3]:
            print(class_labels[index] + " with propability " + str(result[0][index]*100) + "%")