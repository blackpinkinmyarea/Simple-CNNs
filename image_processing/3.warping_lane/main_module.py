import cv2
import numpy as np
import my_func

def getLaneCurve(img):
    imgCopy = img.copy()

    # Step 1
    imgThres = my_func.thresholding(img)

    # Step 2
    h, w, c = img.shape
    #Initializing the track bars
    points = my_func.valTrackbars()
    imgWarp = my_func.warfImg(img, points, w, h)
    imgWarpThres = my_func.warfImg(imgThres, points, w, h)
    imgWarpPoints = my_func.drawPoints(imgCopy, points)

    cv2.imshow('Threshold', imgThres)
    cv2.imshow('Warp', imgWarp)
    cv2.imshow('Warping_Points', imgWarpPoints)
    cv2.imshow('Warp_Threshold', imgWarpThres)

    return None

if __name__ == '__main__':
    cap = cv2.VideoCapture('video_lane.mp4')

    #Intializing the track bars
    #initialTrackBarVals = [100, 100, 100, 100]
    #After adjust to find the best
    initialTrackBarVals = [105, 80, 20, 215]
    my_func.initializeTrackbars(initialTrackBarVals)
    frameCounter = 0


    while True:
        frameCounter += 1
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0
        success, img = cap.read()
        img = cv2.resize(img, (480, 240))
        getLaneCurve(img)

        cv2.imshow('video', img)
        cv2.waitKey(1)  