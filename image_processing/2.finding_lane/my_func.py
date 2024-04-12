import cv2
import numpy as np

#Finding lane using threshold
def thresholding(img):
    img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    #Origin threhold
    # lowerWhite = np.array([0, 0, 0])
    # upperWhite = np.array([179, 255, 255])

    #Adjusted threhold
    # lowerWhite = np.array([10, 0, 0])
    # upperWhite = np.array([100, 255, 255])
    lowerWhite = np.array([80, 0, 0])
    upperWhite = np.array([255, 160, 255])

    maskedWhite = cv2.inRange(img_hsv,lowerWhite,upperWhite)
    return maskedWhite


