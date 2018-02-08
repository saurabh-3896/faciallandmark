
import cv2
import numpy as np
import imutils



import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import os
import imutils

features = []
shape = []
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('face_landmarks.dat')

count = 0
shape = []



def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords



def getDistances(shape):
    global count
    #mouth D5 = (49-55) , D6 = (52-58)
    #eye D1 = (40-22) D2 = (44-47)
    # D3 = (37-49) D4 =(34 - 52)
    D1 = distance.euclidean(shape[40] , shape[22])
    D2 = distance.euclidean(shape[44] , shape[47])
    D3 = distance.euclidean(shape[37] , shape[49])
    D4 = distance.euclidean(shape[34] , shape[52])
    D5 = distance.euclidean(shape[49] , shape[55])
    D6 = distance.euclidean(shape[52] , shape[58])

    D = [D1,D2,D3,D4,D5,D6,count]

    return D



def getFeatures(gray):
    shape = []
    rects = detector(gray, 0)


    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        #print(shape)
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image


    return shape,getDistances(shape)



if __name__ == '__main__':
    frame = cv2.imread('training/anger/1.jpg',0)
    frame = cv2.resize(frame, (300, 300), interpolation=cv2.INTER_CUBIC)
    # frame = imutils.rotate(frame, -90)
    shape,features = getFeatures(frame)
    for (x,y) in shape:
        cv2.circle(frame,(x,y),1,(255,255,0),-1)

    print(features)
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
