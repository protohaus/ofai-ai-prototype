def preprocess_image(image):
    # pixel values between [0,1]
    image = np.divide(image, 255.0)
    # pixel values between [-0.5,0.5]
    image = np.subtract(image, 0.5)
    # pixel values between [-1,1]
    image = np.multiply(image,2.0)
    return image