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

#Warping image
def warfImg(img, points, w, h):
    point1 = np.float32(points)
    point2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

#Transformation matrix
    matrix = cv2.getPerspectiveTransform(point1, point2)
    imgWarp = cv2.warpPerspective(img, matrix, (w, h))
    return imgWarp

#Track bars
def nothing(a):
    pass
    
def initializeTrackbars(intialTracbarVals,wT=480, hT=240):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Width Top", "Trackbars", intialTracbarVals[0],wT//2, nothing)
    cv2.createTrackbar("Height Top", "Trackbars", intialTracbarVals[1], hT, nothing)
    cv2.createTrackbar("Width Bottom", "Trackbars", intialTracbarVals[2],wT//2, nothing)
    cv2.createTrackbar("Height Bottom", "Trackbars", intialTracbarVals[3], hT, nothing)

def valTrackbars(wT=480, hT=240):
    widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
    heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
    points = np.float32([(widthTop, heightTop), (wT-widthTop, heightTop),
                      (widthBottom , heightBottom ), (wT-widthBottom, heightBottom)])
    return points    


# Drawing points
def drawPoints(img,points):
    for x in range( 0,4):
        cv2.circle(img,(int(points[x][0]),int(points[x][1])),15,(0,0,255),cv2.FILLED)
    return img
