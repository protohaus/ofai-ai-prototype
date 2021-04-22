import os
import numpy as np
import shutil

# Definitions
dir_name = os.path.dirname(__file__)
subfolder = 'segmented'
image_dir = os.path.join(dir_name,'raw',subfolder)
train_target_dir = os.path.join(dir_name,'data','train',subfolder)
test_target_dir = os.path.join(dir_name,'data','test',subfolder)
# only needed if train data is not split into train and val in training script
#val_target_dir = os.path.join(dir_name,'data','val',subfolder)

n = []

min_images = 1499

for folder in os.listdir(image_dir):
    print(folder)
    image_list = os.listdir(os.path.join(image_dir,folder))
    print(len(image_list))
    n.append(len(image_list))

    if len(image_list) > min_images:
        if not os.path.exists(os.path.join(train_target_dir,folder)):
            os.makedirs(os.path.join(train_target_dir,folder))
        if len(os.listdir(os.path.join(train_target_dir,folder))) == 0:
            for image in image_list[int((min_images+1)/6):]:
                shutil.copy(os.path.join(image_dir,folder,image),os.path.join(train_target_dir,folder,image))

        if not os.path.exists(os.path.join(test_target_dir,folder)):
            os.makedirs(os.path.join(test_target_dir,folder))
        if len(os.listdir(os.path.join(test_target_dir,folder))) == 0:
            for image in image_list[:int((min_images+1)/6)]:
                shutil.copy(os.path.join(image_dir,folder,image),os.path.join(test_target_dir,folder,image))

    #if not os.path.exists(os.path.join(val_target_dir,folder)):
    #    os.makedirs(os.path.join(val_target_dir,folder))
    #if len(os.listdir(os.path.join(val_target_dir,folder))) == 0:
    #    for image in image_list[0:20]:
    #        shutil.copy(os.path.join(image_dir,folder,image),os.path.join(val_target_dir,folder,image))

print("Minimal data of %d images" % n[np.argmin(n)])