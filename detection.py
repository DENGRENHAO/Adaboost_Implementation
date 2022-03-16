import os
from turtle import Turtle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import utils
from os import walk
from os.path import join
from datetime import datetime
from google.colab.patches import cv2_imshow


def crop(x1, y1, x2, y2, x3, y3, x4, y4, img):
    """
    This function ouput the specified area (parking space image) of the input frame according to the input of four xy coordinates.

      Parameters:
        (x1, y1, x2, y2, x3, y3, x4, y4, frame)

        (x1, y1) is the lower left corner of the specified area
        (x2, y2) is the lower right corner of the specified area
        (x3, y3) is the upper left corner of the specified area
        (x4, y4) is the upper right corner of the specified area
        frame is the frame you want to get it's parking space image

      Returns:
        parking_space_image (image size = 360 x 160)

      Usage:
        parking_space_image = crop(x1, y1, x2, y2, x3, y3, x4, y4, img)
    """
    left_front = (x1, y1)
    right_front = (x2, y2)
    left_bottom = (x3, y3)
    right_bottom = (x4, y4)
    src_pts = np.array([left_front, right_front, left_bottom, right_bottom]).astype(np.float32)
    dst_pts = np.array([[0, 0], [0, 160], [360, 0], [360, 160]]).astype(np.float32)
    projective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    croped = cv2.warpPerspective(img, projective_matrix, (360, 160))
    return croped


def detect(dataPath, clf):
    """
    Please read detectData.txt to understand the format.
    Use cv2.VideoCapture() to load the video.gif.
    Use crop() to crop each frame (frame size = 1280 x 800) of video to get parking space images. (image size = 360 x 160)
    Convert each parking space image into 36 x 16 and grayscale.
    Use clf.classify() function to detect car, If the result is True, draw the green box on the image like the example provided on the spec.
    Then, you have to show the first frame with the bounding boxes in your report.
    Save the predictions as .txt file (Adaboost_pred.txt), the format is the same as GroundTruth.txt.
    (in order to draw the plot in Yolov5_sample_code.ipynb)

      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    # raise NotImplementedError("To be implemented")

    with open(dataPath, "r") as file:     #read detectData.txt
        lines = file.readlines()
    car_pos = []
    for line in lines:
        data = line.split()
        if (len(data)) == 8:
            car_pos.append(data)          #store car positions(x1, y1, x2, y2, x3, y3, x4, y4) in car_pos
    cap = cv2.VideoCapture("data/detect/video.gif")       #load video
    predictions = []
    while True:
        _, frame = cap.read()           #read a frame every time
        if frame is None:
            break
        result = frame.copy()           #make a copy from frame
        prediction = []
        for i in range(len(car_pos)):
            x1, y1, x2, y2, x3, y3, x4, y4 = car_pos[i]
            car_img = crop(x1, y1, x2, y2, x3, y3, x4, y4, frame)   #crop car image from the frame
            car_img = cv2.resize(car_img, (36, 16), interpolation=cv2.INTER_AREA)   #resize car image to (36, 16)
            car_img = cv2.cvtColor(car_img, cv2.COLOR_BGR2GRAY)       #convert car image into grayscale image
            if clf.classify(car_img):     #if true, then it predicts that the image contains a car
                pts = np.array([[x3, y3], [x1, y1], [x2, y2], [x4, y4]], np.int32)    #put car positions in numpy array
                result = cv2.polylines(result, [pts], True, (0, 255, 0), 2)       #draw quadrilateral with cv2.polylines
                prediction.append(1)
            else:
                prediction.append(0)
        cv2_imshow(result)          #show result image
        predictions.append(prediction)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print("write file")
    with open('./Adaboost_pred.txt', 'w') as f:   #write predictions in Adaboost_pred.txt
        for prediction in predictions:
            for p in prediction:
                f.write(f"{p}")
                f.write(' ')
            f.write('\n')
    print("writed")
    cap.release()
    cv2.destroyAllWindows()
    # End your code (Part 4)
