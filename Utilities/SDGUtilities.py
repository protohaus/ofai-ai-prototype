import numpy as np # linear algebra

def preprocess_image(image):
    # pixel values between [0,1]
    image = np.divide(image, 255.0)
    # pixel values between [-0.5,0.5]
    image = np.subtract(image, 0.5)
    # pixel values between [-1,1]
    image = np.multiply(image,2.0)
    return image

def crop_and_scale(img):
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
    return img_cropped