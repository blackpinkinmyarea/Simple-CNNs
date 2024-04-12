import cv2
import numpy as np
import my_func

def getLaneCurve(img):
    imgThres = my_func.thresholding(img)
    cv2.imshow('Threshold', imgThres)
    return None

if __name__ == '__main__':
    cap = cv2.VideoCapture('video_lane.mp4')
    while True:
        success, img = cap.read()
        img = cv2.resize(img, (480, 240))
        getLaneCurve(img)

        cv2.imshow('video', img)
        cv2.waitKey(1)    