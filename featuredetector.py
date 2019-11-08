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
from sklearn import linear_model, datasets

from nav_msgs.msg import Odometry # We need this message type to read position and attitude from Bebop nav_msgs/Odometry
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from std_msgs.msg import Empty
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
time = np.zeros(5)

f = 202
B = 30
window=2
skipPixel = 1


def featuredetector(img):
    # FAST Method
    #fast = cv2.FastFeatureDetector_create()
    #kp = fast.detect(img,None)
    
    # SIFT Method
    sift = cv2.xfeatures2d.SIFT_create(100)
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
    rospy.Subscriber('/bebop/odom', Odometry, writeOdom)
    rospy.spin()

def writeOdom(data):
    global global_pos
    global global_vel
    #rospy.loginfo(data.pose.pose)
    #rospy.loginfo(data.twist.twist)
    global_pos=data.pose.pose
    global_vel=data.twist.twist

def SaveImg(data,LorR):
    global leftImg, rightImg, time, header
    if LorR == "L":
        leftImg = bridge.imgmsg_to_cv2(data,"passthrough")
        imgout = leftImg
        header = data.header.stamp
        time[0:4] = time[1:5]
        time[4] = rospy.get_time()
        #print time
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
        print "SIFT: ",len(features)
        points = 0
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
        
        print "SIFT compare: ",len(features)
    
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

        Z,d = SD.sterioDepth(points[range(0,points.shape[0])].astype(int),leftImg,rightImg,f,B,window,skipPixel)
        
        # Filter points to only include those on Z plane
        # ransac = linear_model.RANSACRegressor()
        # ransac.fit(points, Z)
        # inlier_mask = ransac.inlier_mask_
        # outlier_mask = np.logical_not(inlier_mask)
        Z_filter =  Z#[inlier_mask]
        Points_filter = points#points[inlier_mask]
        print("Mean - filter", np.mean(Z_filter))
        print("Median - filter", np.median(Z_filter))
        print("Median - raw", np.mean(Z))
        print("Odom", global_pos.position.z)
        
        # Points for RANSAC
        # Filter this for NaN's
        #print points.shape
        nanMask = np.array(~np.isnan(Z))
        xpts = points[:,0]
        ypts = points[:,1]
        #print xpts.shape
        #print nanMask.shape

        # These points and deltas don't have any NaNs in them :)
        ranPoints = [xpts[nanMask],ypts[nanMask],Z[nanMask]]
        ranDelta = delta[nanMask]

        # RUN RANSAC HERE
       # FilterPoints = IlyasRanSack

        # Telemetry rate
        rate = (time[4]-time[0])/5
        print rate

        # for i in range(0,finalPoints.shape[0]): 
        #     A[i] = np.array([[-1/Z[i],0,x[i]/Z[i],x[i]*y[i],-(1+x[i]*x[i]),y[i]],[0,-1/Z[i],y[i]/Z[i],(1+y[i]*y[i]), -x[i]*y[i], -x[i]]])
        #     b[i] = ranDelta*rate
        #     res[i] = np.linalg.solve(A[i],b[i])
        # Now average all results...

        # Then Integrate to get odometry
      

        
        

    # Visualize features
    #img2 = cv2.drawKeypoints(thisimg, features,  None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Run Optical Flow Calaculation here


    #set parameters for next run
    frame = frame + 1
    lastimg = copy.copy(thisimg)
    lastfeatures = copy.copy(features)
    featurecount = len(features)
    lastfeaturecount = len(lastfeatures)
    return points, delta

if __name__ == '__main__':
    try:
        rundetection()
    except rospy.ROSInterruptException:
        pass