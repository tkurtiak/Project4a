#!/usr/bin/env python2
# vertex find
import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 
from scipy.stats import multivariate_normal
from scipy.cluster.vq import kmeans, whiten, kmeans2
import imutils
#Image Select
import Tkinter, tkFileDialog
from time import time
import os
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge, CvBridgeError
import rospy
import copy
import stereoDepth as SD
#import KLT
# For testing
#img = cv2.imread("frame0049.jpg")

bridge = CvBridge()
# Initiate FAST object with default values
fast = cv2.FastFeatureDetector()
img_pub1 = rospy.Publisher("/feature_img1",Image)
img_pub2 = rospy.Publisher("/feature_img2",Image)

# Define Global parameters
frame = 0
featurecount = 0
lastfeaturecount = 0
#thisimg = np.zeros((800,600))
#lastimg = np.zeros((800,600))
matches = 0

f = 202
B = 30
window=5  
skipPixel = 1


def featuredetector(img):
    # FAST Method
    #fast = cv2.FastFeatureDetector_create()
    #kp = fast.detect(img,None)
    
    # SIFT Method
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img,None)
    # find and draw the keypoints
   
    
    
    # Print all default params
    #print "Total Keypoints: "len(kp)
    #print "Threshold: ", fast.getThreshold()
    #print "nonmaxSuppression: ", fast.getNonmaxSuppression()

    return kp, des

def featurecompare(des1, des2):
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=1)
    return matches

def rundetection():
    rospy.init_node('feature_detection', anonymous=True)
    rospy.Subscriber("/duo3d/right/image_rect", Image, SaveImg,"R")
    rospy.Subscriber("/duo3d/left/image_rect", Image, OpticalFlow)


    rospy.spin()

def SaveImg(data,LorR):
    global leftImg, rightImg
    if LorR == "L":
        leftImg = bridge.imgmsg_to_cv2(data,"passthrough")
        imgout = leftImg
    else:
        rightImg = bridge.imgmsg_to_cv2(data,"passthrough")
        imgout = rightImg
    return imgout

def OpticalFlow(data):
    global frame, matches, des, lastdes, features, lastfeatures, featurecount, lastfeaturecount, thisimg, lastimg
    global f, B, window, SkipPixel, Z
    thisimg = SaveImg(data,"L")
    #thisimg = bridge.imgmsg_to_cv2(data,"passthrough")

    print frame

    # If not enough features are remaining in the image, generate new features
    if (frame == 0): #or (featurecount/lastfeaturecount < .8) or (featurecount<1000):
        features, des = featuredetector(thisimg)
        #print "SIFT: ",len(features)
    # Otherwise use feature tracking
    else:
        lastfeatures = copy.copy(features)
        lastdes = copy.copy(des)
        features, des = featuredetector(thisimg)
        matches = featurecompare(lastdes, des)

        # Extract points from matches
        points = np.zeros((len(matches),2))
        delta = np.zeros((len(matches),2))
        #tracker = cv2.Tracker_KCF_create()
        #lastpoints = cv2.KeyPoint_convert(lastfeatures).astype(int)
        #features = KLT.calc_klt(lastimg, thisimg, lastpoints, win_size=(21, 21), max_iter=10, min_disp=0.01)
        
        #print "SIFT compare: ",len(features)
    
        # Need to draw only good matches, so create a mask
        #matchesMask = [[0,0] for i in range(len(matches))]
        
        # ratio test as per Lowe's paper
        # source: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
        for i in range(0,len(matches)):
        #    if m.distance < 0.7*n.distance:
        #        matchesMask[i]=[1,0]
            points[i] = lastfeatures[matches[i][0].queryIdx].pt#features[m.queryIdx]]
            delta[i] = np.subtract(features[matches[i][0].imgIdx].pt,lastfeatures[matches[i][0].queryIdx].pt)   
        #print points#delta[0]
        draw_params = dict(matchColor = (0,255,0),
                           singlePointColor = (255,0,0),
        #                   matchesMask = matchesMask,
                           flags = cv2.DrawMatchesFlags_DEFAULT)
        img3 = cv2.drawMatchesKnn(lastimg,lastfeatures,thisimg,features,matches[:100],None,**draw_params)
        #print matches[1]
        #print len(lastfeatures)
        #print len(features)
        #plt.imshow(img3,),plt.show()
        img_pub1.publish(bridge.cv2_to_imgmsg(img3,"bgr8"))

        Z,d = SD.sterioDepth(points[range(0,points.shape[0],5)].astype(int),leftImg,rightImg,f,B,window,skipPixel)
        print np.median(Z)
        # delta = F(points,Z)*States
        # find states.
        # Construct matrix of F by augmenting rows for each point 
        # then take SVD to solve

        # Impliment a RANSAC function to get rid of outliers

        return points, delta
        

    # Visualize features
    #img2 = cv2.drawKeypoints(thisimg, features,  None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Run Optical Flow Calaculation here


    #set parameters for next run
    frame = frame + 1
    lastimg = copy.copy(thisimg)
    lastfeatures = copy.copy(features)
    featurecount = len(features)
    lastfeaturecount = len(lastfeatures)

if __name__ == '__main__':
    try:
        rundetection()
    except rospy.ROSInterruptException:
        pass