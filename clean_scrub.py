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


def featuredetector(img):  
    # ORB Method
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(img,None)

    # SIFT Method
    # sift = cv2.xfeatures2d.SIFT_create(500)
    # kp, des = sift.detectAndCompute(img,None)

    return kp, des


def featurecompare(kp1,des1, kp2,des2):
    points = np.zeros((len(matches),2))
    delta = np.zeros((len(matches),2))
    dist = np.zeros((len(matches)))  
    matchMask = np.zeros((len(matches),2)) 


    # ORB Method: BF
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2) #query is des1, train is des2, query is left, train is right
    matches = sorted(matches, key = lambda x:x.distance) 
    for i in range(0,num_matches):
        points[i] = kp1[matches[i].queryIdx].pt
        delta[i] = np.subtract(kp2[matches[i].trainIdx].pt,kp1[matches[i].queryIdx].pt)   
        dist[i] = np.sqrt(delta[i,0]**2+delta[i,1]**2)
        if dist[i]<40:    
            matchMask[i]=[1,0] 



    # SIFT Method: FLANN
    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # search_params = dict(checks=50)   # or pass empty dictionary
    # flann = cv2.FlannBasedMatcher(index_params,search_params)
    # matches = flann.knnMatch(des1,des2,k=2) #query is des1, train is des2, query is left, train is right
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

    num_matches= len(vec2right)
    for i in range(0,num_matches):
        if np.abs(delta[i,0]/delta[i,1]) > 10:
                #x is bigger than y, so is more or less horizontal
                matchMask[i]=[1,0]
    matchMaskbool = matchMask.astype('bool')
    points = points[matchMaskbool[:,0]]
    delta = delta[matchMaskbool[:,0]]
    dist = dist[matchMaskbool[:,0]]

    f = 202
    B = 30
    Z = np.divide(f*B,d)

    return Z, d, points


def writeOdom(data):
    global global_pos
    global global_vel
    global_pos=data.pose.pose
    global_vel=data.twist.twist

def rundetection():
    rospy.init_node('feature_detection', anonymous=True)
    right_sub=message_filters.Subscriber("/duo3d/right/image_rect", Image)#,heyo1)#,queue_size=4)
    left_sub=message_filters.Subscriber("/duo3d/left/image_rect", Image)#,heyo2)#,queue_size=4)
    rospy.Subscriber('/bebop/odom', Odometry, writeOdom)
    ts = message_filters.ApproximateTimeSynchronizer([left_sub,right_sub],10,.05e9)
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
    global frame, lastdes, lastfeatures

    leftImg = bridge.imgmsg_to_cv2(leftImg,"passthrough")
    rightImg= bridge.imgmsg_to_cv2(rightImg,"passthrough")
    
    print 'Frame #:'
    print frame

    # If first frame make features, chill out
    if (frame == 0):
        lastfeatures, lastdes = featuredetector(leftImg)
    # Otherwise use feature tracking
    else:
        
        features, des = featuredetector(leftimg)
        points_temporal,delta_temporal,dist_temporal = featurecompare(features, des, lastfeatures, lastdes)
        #delta_temporal vectors are from old frame to new frame, points are on new frame
        delta_temporal= -1*delta_temporal

        lastfeatures = copy.copy(features)
        lastdes = copy.copy(des)

        featuresRight, desRight = featuredetector(rightImg)
        
        Z,d, points_spatial = stereo(features,des,featuresRight,desRight)
        
        #Z,d, temppnts = SD_o.stereoDepthORB(leftImg,rightImg,f,B)       
        #ranPoints = np.array([temppnts[:,0],temppnts[:,1],Z]).T

        # nanMask = np.array(~np.isnan(Z))
        # xpts = points[:,0]
        # ypts = points[:,1]
        # ranPoints = np.array([xpts[nanMask],ypts[nanMask],Z[nanMask]]).T
        # ranDelta = delta[nanMask]

        print("Odom Height, meters", global_pos.position.z)
        
        ## Script breaks if there are too few points input into ransac.  
        # throw an error if there are too few points
        # print ranPoints.shape 
        if 0>1: #ranPoints.shape[0]<4:
            print("Not enough Non NaN points for RANSAC")
        
        ## RUN RANSAC HERE
        else:

            # #if doing ORB stereo
            # temppnts, ransMask, bestnormal, bestD = PR.PlaneRANSAC(ranPoints)
            print("Median Z", np.median(Z))
            print("Mean Z", np.mean(Z))
            FinalDelta = delta
            FinalPoints = points



            #if doing normal stereo
            # FinalPoints, ransMask, bestnormal, bestD = PR.PlaneRANSAC(ranPoints)
            # print("Median Z", np.median(FinalPoints[:,2]))
            # print("Mean Z", np.mean(FinalPoints[:,2]))
            # #global FinalDelta
            # FinalDelta = ranDelta[ransMask]

            # plt.cla()
            # plt.imshow(thisimg)
            # plt.quiver(FinalPoints[:,0],FinalPoints[:,1],FinalDelta[:,0],FinalDelta[:,1])
            # plt.ylim((0,480))
            # plt.xlim((0,640))
            # plt.pause(0.05)
            #plt.show()

            # Calculate 5 frame rolling avg Telemetry rate
            #rate = (time[4]-time[0])/5
            rate = (time[4]-time[0])/4 #if have 5 times, then have 4 deltaTs
            print("Telemetry Rate, seconds per frame",rate)

            ## Calculate Optical Flow 
            #res = np.zeros((FinalPoints.shape[0],6))
            A = np.zeros((2*FinalPoints.shape[0],6))
            b = np.zeros((2*FinalPoints.shape[0]))#FinalDelta.T*rate
            #print thisimg.shape
            for i in range(0,FinalPoints.shape[0]): 
                # Transfer points to image frame with 0,0 at center of the image
                #finalpoints x,y is wrt top left corner of image, so this is actually:
                # x = FinalPoints[i,0]-thisimg.shape[0]
                # y = FinalPoints[i,1]-thisimg.shape[1]
                x = FinalPoints[i,0]-thisimg.shape[1]/2
                y = thisimg.shape[0]/2 - FinalPoints[i,1]
                # Try replacing Z with odom altitude, for fun...
                Z = -global_pos.position.z
                #Z = -FinalPoints[i,2]/1000
                # Populate Optical Flow Matrix for all points
                A[2*i:2*i+2] = np.array([[-1/Z,0,x/Z,x*y,-(1+x*x),y],[0,-1/Z,y/Z,(1+y*y), -x*y, -x]])
                b[2*i:2*i+2] = FinalDelta[i]/rate
                #print A
                #print b
            #print A.shape
            #print b.shape
            # Linear least squares solver on optical flow equation
            Results, res, rank, s = np.linalg.lstsq(A,b)
            #print Results
            #print Results.shape
            print("Opt Flow Velocity m/s:", Results[0:3])
            print("Opt Flow Rotations:", Results[3:])
            print("Odometry Velocity m/s:", global_vel.linear)
            # Then Integrate to get odometry
      

        
        

    # Visualize features
    #img2 = cv2.drawKeypoints(thisimg, features,  None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Run Optical Flow Calaculation here

    #plt.show()
    #set parameters for next run
    frame = frame + 1
    # lastimg = copy.copy(thisimg)
    lastfeatures = copy.copy(features)
    featurecount = len(features)
    lastfeaturecount = len(lastfeatures)
    return points, delta





if __name__ == '__main__':
    try:
        rundetection()
    except rospy.ROSInterruptException:
        pass