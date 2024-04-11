import cv2
import numpy as np

def featureTracking(img_1, img_2, points1, points2, status):
    winSize = (21, 21)
    termcrit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 30, 0.01)
    err = []

    cv2.calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001)

    indexCorrection = 0
    for i in range(len(status)):
        pt = points2[i - indexCorrection]
        if (status[i] == 0) or (pt[0] < 0) or (pt[1] < 0):
            if (pt[0] < 0) or (pt[1] < 0):
                status[i] = 0
            points1.pop(i - indexCorrection)
            points2.pop(i - indexCorrection)
            indexCorrection += 1

def KAZEdesu(img_1, points1, keypoints_1):
    minHessian = 400
    detector = cv2.KAZE_create(minHessian)
    keypoints_1 = detector.detect(img_1)
    points1 = [kp.pt for kp in keypoints_1]

def ORBdesu(img_1, points1, keypoints_1):
    minHessian = 400
    detector = cv2.ORB_create(minHessian)
    keypoints_1 = detector.detect(img_1)
    points1 = [kp.pt for kp in keypoints_1]

def SIFTdesu(img_1, points1, keypoints_1):
    minHessian = 400
    detector = cv2.xfeatures2d.SIFT_create(minHessian)
    keypoints_1 = detector.detect(img_1)
    points1 = [kp.pt for kp in keypoints_1]

def SURFdesu(img_1, points1, keypoints_1):
    minHessian = 800
    detector = cv2.xfeatures2d.SURF_create(minHessian)
    keypoints_1 = detector.detect(img_1)
    points1 = [kp.pt for kp in keypoints_1]

def featureDetection(img_1, points1, keypoints_1):
    fast_threshold = 20
    nonmaxSuppression = True
    keypoints_1 = cv2.FAST(img_1, fast_threshold, nonmaxSuppression)
    points1 = [kp.pt for kp in keypoints_1]
