#!/usr/bin/env python2

import cv2
import numpy as np
import message_filters

from matplotlib import pyplot as plt
import imutils

from time import time
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge, CvBridgeError
import rospy
import copy

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from std_msgs.msg import Empty
import PlaneRANSAC as PR

import faulthandler
faulthandler.enable()



bridge = CvBridge()
orb = cv2.ORB_create(nfeatures=500, scoreType=cv2.ORB_FAST_SCORE)


time = np.zeros(5)
frame=0


def featuredetector(img):  
    # ORB Method
    global orb
    kp, des = orb.detectAndCompute(img,None)

    # SIFT Method
    # sift = cv2.xfeatures2d.SIFT_create(500)
    # kp, des = sift.detectAndCompute(img,None)

    return kp, des


def featurecompare(kp1,des1, kp2,des2):
     


    # ORB Method: BF
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2) #query is des1, train is des2, query is left, train is right
    matches = sorted(matches, key = lambda x:x.distance) 

    points = np.zeros((len(matches),2))
    delta = np.zeros((len(matches),2))
    dist = np.zeros((len(matches)))  
    matchMask = np.zeros((len(matches),2))

    for i in range(0,len(matches)):
        points[i] = kp1[matches[i].queryIdx].pt
        delta[i] = np.subtract(kp2[matches[i].trainIdx].pt,kp1[matches[i].queryIdx].pt)   
        dist[i] = np.sqrt(delta[i,0]**2+delta[i,1]**2)
        if dist[i]<60:    
            matchMask[i]=[1,0] 



    # SIFT Method: FLANN
    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # search_params = dict(checks=50)   # or pass empty dictionary
    # flann = cv2.FlannBasedMatcher(index_params,search_params)
    # matches = flann.knnMatch(des1,des2,k=2) #query is des1, train is des2, query is left, train is right
    # 
    # points = np.zeros((len(matches),2))
    # delta = np.zeros((len(matches),2))
    # dist = np.zeros((len(matches)))  
    # matchMask = np.zeros((len(matches),2))

    # for i in range(0,len(matches)):
    #     points[i] = kp1[matches[i][0].queryIdx].pt #want points on left image
    #     delta[i] = np.subtract(kp2[matches[i][0].trainIdx].pt,kp1[matches[i][0].queryIdx].pt)   
    #     dist[i] = np.sqrt(delta[i,0]**2+delta[i,1]**2)#matches[i][0].distance
    #     if matches[i][0].distance < 0.7*matches[i][1].distance:     
    #         if dist[i]<40:    
    #             matchMask[i]=[1,0]     
    



    matchMaskbool = matchMask.astype('bool')
    points = points[matchMaskbool[:,0]]
    delta = delta[matchMaskbool[:,0]]
    dist = dist[matchMaskbool[:,0]]


    return points,delta,dist

def stereo(leftkp,leftdes,rightkp,rightdes):
    points_left,vec2right,dist = featurecompare(leftkp,leftdes,rightkp,rightdes)

    num_matches= len(vec2right[:,0])
    matchMask = np.zeros((num_matches,2))
    for i in range(0,num_matches):
        if np.abs(vec2right[i,0]/vec2right[i,1]) > 10:
                #x is bigger than y, so is more or less horizontal
                matchMask[i]=[1,0]

    matchMaskbool = matchMask.astype('bool')
    points = points_left[matchMaskbool[:,0]]
    delta = vec2right[matchMaskbool[:,0]]
    dist = dist[matchMaskbool[:,0]]

    f = 202
    B = 30
    Z = np.divide(f*B,dist)

    return Z, dist, points

def writeOdom(data):
    global global_pos
    global global_vel
    global_pos=data.pose.pose
    global_vel=data.twist.twist

def rundetection():
    rospy.init_node('feature_detection', anonymous=True)
    right_sub=message_filters.Subscriber("/duo3d/right/image_rect", Image, queue_size=10)#,heyo1)#,queue_size=4)
    left_sub=message_filters.Subscriber("/duo3d/left/image_rect", Image, queue_size=10)#,heyo2)#,queue_size=4)
    rospy.Subscriber('/bebop/odom', Odometry, writeOdom)
    #ts = message_filters.ApproximateTimeSynchronizer([left_sub,right_sub],1,.05e9)
    ts = message_filters.TimeSynchronizer([left_sub,right_sub],10)
    ts.registerCallback(OpticalFlow)
    rospy.spin()

def OpticalFlow(leftImg,rightImg):
    #when we get into this function left,right and odom should all be synched
    global f, B, window, SkipPixel, Z, delta, slide_dist
    global time
    time[0:4] = time[1:5]
    time[4] = float(str(leftImg.header.stamp))/1e9

    global global_pos
    global global_vel
    global frame, lastdes, lastfeatures, bridge

    leftImg = bridge.imgmsg_to_cv2(leftImg,"passthrough")
    rightImg= bridge.imgmsg_to_cv2(rightImg,"passthrough")

    plt.cla()
    plt.subplot(121),plt.imshow(leftImg,cmap = 'gray')
    plt.subplot(122),plt.imshow(rightImg,cmap = 'gray')
    plt.pause(0.05)
    
    print 'Frame #:'
    print frame

    # If first frame make features, chill out
    if (frame == 0):
        lastfeatures, lastdes = featuredetector(leftImg)
    # Otherwise use feature tracking
    else:
        
        features, des = featuredetector(leftImg)
        points_temporal,delta_temporal,dist_temporal = featurecompare(features, des, lastfeatures, lastdes)
        #delta_temporal vectors are from old frame to new frame, points are on new frame
        delta_temporal= -1*delta_temporal

        lastfeatures = copy.copy(features)
        lastdes = copy.copy(des)

        featuresRight, desRight = featuredetector(rightImg)
        
        Z,d, points_spatial = stereo(features,des,featuresRight,desRight)

        #https://stackoverflow.com/questions/20230384/find-indexes-of-matching-rows-in-two-2-d-arrays
        indicies= np.array(np.all((points_temporal[:,None,:]==points_spatial[None,:,:]),axis=-1).nonzero()).T#.tolist()
        #all x == y, gives [[a,b],[c,d]] where x[a]=y[b] x[c]=y[d] and so on


        finalpoints= np.array([points_spatial[indicies[:,1],0],points_spatial[indicies[:,1],1],Z[indicies[:,1]]])
        finalflows= delta_temporal[indicies[:,0]]


        print points_temporal.shape
        print points_spatial.shape
        print finalpoints.shape

        # print ' points_temporal'
        # print points_temporal
        # print 'delta_temporal'
        # print delta_temporal
        # print ' points_spatial'
        # print points_spatial
        # print 'final flows'
        # print finalflows
        # print 'indicies'
        # print indicies



        print np.median(Z)
        print np.mean(Z)
        print np.median(d)
        print np.mean(d)




        print("Odom Height, meters", global_pos.position.z)
        

        
    frame = frame + 1
    





if __name__ == '__main__':
    try:
        rundetection()
    except rospy.ROSInterruptException:
        pass