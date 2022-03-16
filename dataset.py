import os
import cv2


def loadImages(dataPath):
    """
    Load all Images in the folder and transfer a list of tuples. 
    The first element is the numpy array of shape (m, n) representing the image.
    (remember to resize and convert the parking space images to 36 x 16 grayscale images.) 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    # Begin your code (Part 1)
    # raise NotImplementedError("To be implemented")
    images = []
    num = []          #num stands for labels
    car_path = dataPath + '/car/'
    for filename in os.listdir(car_path):           #open car folder
        img = cv2.imread(os.path.join(car_path, filename))              #read image
        img = cv2.resize(img, (36, 16), interpolation=cv2.INTER_AREA)   #resize image to (36, 16)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                     #convert image to grayscale image
        if img is not None:
            images.append(img)
            num.append(1)
    non_car_path = dataPath + '/non-car/'           #open non-car folder
    for filename in os.listdir(non_car_path):
        img = cv2.imread(os.path.join(non_car_path, filename))
        img = cv2.resize(img, (36, 16), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img is not None:
            images.append(img)
            num.append(0)
    dataset = list(zip(images, num))          #turn two lists into list of tuples                            
    # End your code (Part 1)

    return dataset
