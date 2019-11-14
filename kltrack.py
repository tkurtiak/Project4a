#!/usr/bin/env python
# from __future__ import division
import cv2
import numpy as np
import message_filters

from matplotlib import pyplot as plt
import imutils
from time import time
# import os
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge, CvBridgeError
import rospy
import copy
import stereoDepth as SD
# from sklearn import linear_model, datasets

from nav_msgs.msg import Odometry # We need this message type to read position and attitude from Bebop nav_msgs/Odometry
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from std_msgs.msg import Empty
import PlaneRANSAC as PR
from itertools import compress
import tf
# import sys
# # insert at 1, 0 is the script path (or '' in REPL)
# # sys.path.insert(1, '/home/vdorbala/git/PyFeatureTrack/')
# sys.path.insert(1, '/home/vdorbala/git/KLT/')

# from klt import lucasKannadeTracker

import faulthandler
faulthandler.enable()

global_pos = 0
global_vel = 0
bridge = CvBridge()
# Initiate FAST object with default values

# Define/Initialize Global parameters
frame = 0

spacing = 20
margin= 40
time = np.zeros(5)

# Camera focal length [pixel]
f = 202
# Stereo base distance [mm]
B = 30
lk_params = dict( winSize  = (15,15),
                               maxLevel = 2,
                               criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

#resolution: [640, 480]
points_to_track = []


for x in range(0+margin,480-margin,spacing):
    for y in range(0+margin,640-margin,spacing):
        new_point = [y, x]
        points_to_track.append(new_point)
points_to_track = np.array(points_to_track,dtype=np.float32)
points_to_track = points_to_track.reshape(points_to_track.shape[0], 1, points_to_track.shape[1])


odom_pub = rospy.Publisher("/our_odometry",Odometry,queue_size=10)


def plotter(image, points, flow):

    color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    color_red = [0,255,0] # bgr colorspace
    linewidth = 5
    for i, point in enumerate(points):
        x = point[0,0]
        y = point[0,1]
        vx = flow[i][0,0]
        vy = flow[i][0,1]
        cv2.line(color_img, (x,y), (x+vx, y+vy), color_red, linewidth) # draw a red line from the point with vector = [vx, vy]        
    
    plt.cla()
    # plt.plot(color_img)
    plt.imshow(color_img)
    # plt.show()
    plt.pause(0.05)
    # cv2.imshow('tracked image',color_img)
    # cv2.waitKey(1)

def plotter2d(image, pointss, flows):

    points=pointss.astype('int')
    flow=flows.astype('int')
    color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    color_red = [0,255,0] # bgr colorspace
    linewidth = 3
    for i in range(points.shape[0]):
        x = points[i,0]
        y = points[i,1]
        vx = flow[i,0]
        vy = flow[i,1]
        cv2.line(color_img, (x,y), (x+vx, y+vy), color_red, linewidth) # draw a red line from the point with vector = [vx, vy]        
    
    plt.cla()
    # plt.plot(color_img)
    plt.imshow(color_img)
    # plt.show()
    plt.pause(0.05)
    # cv2.imshow('tracked image',color_img)
    # cv2.waitKey(1)


def writeOdom(data):
    global global_pos
    global global_vel
    global_pos=data.pose.pose
    global_vel=data.twist.twist

def rundetection():
    rospy.init_node('feature_detection', anonymous=True)
    right_sub = message_filters.Subscriber("/duo3d/right/image_rect", Image, queue_size=10)#,heyo1)#,queue_size=4)
    left_sub = message_filters.Subscriber("/duo3d/left/image_rect", Image, queue_size=10)#,heyo2)#,queue_size=4)
    rospy.Subscriber('/bebop/odom', Odometry, writeOdom)

    ts = message_filters.TimeSynchronizer([left_sub,right_sub],10)
    ts.registerCallback(OpticalFlow)
    rospy.spin()

def OpticalFlow(leftImg,rightImg):
    status = 0
    error = 0
    #when we get into this function left,right and odom should all be synched
    global f, B, window, SkipPixel, Z, delta, slide_dist
    global time
    global lk_params
    time[0:4] = time[1:5]
    time[4] = float(str(leftImg.header.stamp))/1e9

    global global_pos
    global global_vel
    global frame, matches, des, lastdes, features, lastfeatures, featurecount, lastfeaturecount, lastImg
    

    leftImg = bridge.imgmsg_to_cv2(leftImg,desired_encoding="mono8")
    rightImg= bridge.imgmsg_to_cv2(rightImg,desired_encoding="mono8")


    # print frame

    # If not enough features are remaining in the image, generate new features
    if (frame == 0):
        lastImg=leftImg
    else:
        newpos_wide, status, error = cv2.calcOpticalFlowPyrLK(lastImg, leftImg, points_to_track, None, **lk_params)

        print(newpos_wide-points_to_track)
        
        newpos=np.squeeze(newpos_wide) #(n,1,2) to (n,2)
        delta= newpos-np.squeeze(points_to_track)


        # delta1 = [element in delta for status(element) == int(1)]
        good_indx=np.array(np.nonzero(status[:,0]))
        temporal_delta=np.squeeze(delta[good_indx,:])
        temporal_pnts=np.squeeze(newpos[good_indx,:])
        
        points_to_track2 = np.array(newpos_wide[good_indx].flatten(),dtype=np.float32)
        points_to_track2 = points_to_track2.reshape(points_to_track2.shape[0]/2, 1, 2)

        rightpos_wide,status,error = cv2.calcOpticalFlowPyrLK(leftImg, rightImg, points_to_track2, None, **lk_params)
        
        good_indx=np.array(np.nonzero(status[:,0]))
        rightpos=np.squeeze(rightpos_wide)
        # print np.shape(rightpos)
        delta= rightpos-temporal_pnts
        spatial_delta=np.squeeze(delta[good_indx,:])
        spatial_pnts=np.squeeze(temporal_pnts[good_indx,:])

      

        flow = rightpos_wide - points_to_track2
        flow = newpos_wide - points_to_track
        plotter(leftImg, points_to_track , flow)




        #alrightyy moving on to actual mathybois
        matchMask = np.zeros((len(spatial_delta[:,0]),2))
        points = matchMask
        d= np.zeros(len(spatial_delta[:,0]))

        for i in range(spatial_delta.shape[0]):
            
            if np.abs(spatial_delta[i,0]/spatial_delta[i,1]) > 10:
                d[i]=np.sqrt(spatial_delta[i,0]**2 + spatial_delta[i,1]**2)
                if d[i] < 20:
                    matchMask[i]=[1,0] 

        matchMaskbool = matchMask.astype('bool')
        spatial_pnts = spatial_pnts[matchMaskbool[:,0]]
        d= d[matchMaskbool[:,0]]

        Zs = np.divide(f*B,d)


        print ('overall height in mm (estimate)')
        print np.median(Zs)

        indicies= np.array(np.all((temporal_pnts[:,None,:]==spatial_pnts[None,:,:]),axis=-1).nonzero()).T
        #...all((... x == y, gives [[a,b],[c,d]] where x[a]=y[b] x[c]=y[d] and so on




        FinalPoints= np.array([spatial_pnts[indicies[:,1],0],spatial_pnts[indicies[:,1],1],Zs[indicies[:,1]]]).T
        FinalFlows= temporal_delta[indicies[:,0]]


        FinalPoints, ransMask, bestnormal, bestD = PR.PlaneRANSAC(FinalPoints)
        FinalDelta = FinalFlows[ransMask]


        #debugging
        # FinalDelta[:,:]=4
        # FinalPoints[:,0]=320
        # FinalPoints[:,1]=240
        # plotter2d(leftImg, FinalPoints, FinalDelta)

        print'ended up with:'
        print FinalPoints.shape

        print '            '
        print '            '
        rate = (time[4]-time[0])/4 #if have 5 times, then have 4 deltaTs
        print("Telemetry Rate, seconds per frame",rate)

        # rate= 1./15

        ## Calculate Optical Flow 
        A = np.zeros((2*FinalPoints.shape[0],6))
        b = np.zeros((2*FinalPoints.shape[0],1))#FinalDelta.T*rate
        #print thisimg.shape
        Z = -np.median(Zs)
        for i in range(0,FinalPoints.shape[0]): 
            # Transfer points to image frame with 0,0 at center of the image
            #finalpoints x,y is wrt top left corner of image, so this is actually:
            x = FinalPoints[i,0]-leftImg.shape[1]/2
            #x = leftImg.shape[1]/2-FinalPoints[i,0]
            y = leftImg.shape[0]/2 - FinalPoints[i,1]

            # Try replacing Z with odom altitude, for fun...
            
            #Z = -FinalPoints[i,2]/1000
            # Populate Optical Flow Matrix for all points
            A[2*i:2*i+2] = np.array([[-1/Z,0,x/Z,x*y,-(1+x*x),y],[0,-1/Z,y/Z,(1+y*y), -x*y, -x]])
            b[2*i:2*i+2,0] = FinalDelta[i,:]/rate

        # Linear least squares solver on optical flow equation

        Results, res, rank, s = np.linalg.lstsq(A,b)
        #print Results
        #print Results.shape
        print("Opt Flow Velocity m/s:", Results[0:3]*0.001)
        print("Opt Flow Rotations:", Results[3:])
        #print("Odometry Velocity m/s:", global_vel.linear)

        print '            '
        print '            '

        odom_broadcast = tf.TransformBroadcaster()

        odom_quat = tf.transformations.quaternion_from_euler(0,0,0)
        
        odom_broadcast.sendTransform((Results[0]*0.001*rate, Results[1]*0.001*rate, Results[2]*0.001*rate),(odom_quat),rospy.get_rostime(),'base_link', "odom")

        odom = Odometry()

        odom.twist.twist.linear.x = Results[0]*0.001*rate
        odom.twist.twist.linear.y = Results[1]*0.001*rate
        odom.twist.twist.linear.z = Results[2]*0.001*rate
        odom.twist.twist.angular.x = 0
        odom.twist.twist.angular.y = 0
        odom.twist.twist.angular.z = 0
        
        odom.header.stamp = rospy.get_rostime()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"

        odom_pub.publish(odom)

        # matchMaskbool = matchMask.astype('bool')
        # points = points[matchMaskbool[:,0]]
        # delta = delta[matchMaskbool[:,0]]
        # dist = dist[matchMaskbool[:,0]]
        # Z,d, points_spatial = stereo(features,des,featuresRight,desRight)

        #https://stackoverflow.com/questions/20230384/find-indexes-of-matching-rows-in-two-2-d-arrays
        # indicies= np.array(np.all((points_temporal[:,None,:]==points_spatial[None,:,:]),axis=-1).nonzero()).T#.tolist()
        # #all x == y, gives [[a,b],[c,d]] where x[a]=y[b] x[c]=y[d] and so on


        # Z,d = SD.sterioDepth(points[range(0,points.shape[0])].astype(int),leftImg,rightImg,f,B,window,skipPixel,slide_dist)
        # Z = np.divide(f*B,d)
        # ## Script breaks if there are too few points input into ransac.  
        # # throw an error if there are too few points
    
        # FinalPoints, ransMask, bestnormal, bestD = PR.PlaneRANSAC(ranPoints)
        # print("Median Z", np.median(FinalPoints[:,2]))
        # print("Mean Z", np.mean(FinalPoints[:,2]))
        # #global FinalDelta
        # FinalDelta = ranDelta[ransMask]
        # #print FinalPoints.shape
        # #print FinalDelta.shape
        
        # # Plot optical Flow Vectors
        # #FinalDelta[:,:]=15


        # # plt.cla()
        # # plt.imshow(thisimg)
        # # plt.quiver(FinalPoints[:,0],FinalPoints[:,1],FinalDelta[:,0],FinalDelta[:,1])
        # # plt.ylim((0,480))
        # # plt.xlim((0,640))
        # # plt.pause(0.05)
        # #plt.show()

        # # Calculate 5 frame rolling avg Telemetry rate
        # #rate = (time[4]-time[0])/5
        # rate = (time[4]-time[0])/4 #if have 5 times, then have 4 deltaTs
        # print("Telemetry Rate, seconds per frame",rate)

        # ## Calculate Optical Flow 
        # #res = np.zeros((FinalPoints.shape[0],6))
        # A = np.zeros((2*FinalPoints.shape[0],6))
        # b = np.zeros((2*FinalPoints.shape[0]))#FinalDelta.T*rate
        # #print thisimg.shape
        # for i in range(0,FinalPoints.shape[0]): 
        #     # Transfer points to image frame with 0,0 at center of the image
        #     #finalpoints x,y is wrt top left corner of image, so this is actually:
        #     # x = FinalPoints[i,0]-thisimg.shape[0]
        #     # y = FinalPoints[i,1]-thisimg.shape[1]
        #     x = FinalPoints[i,0]-thisimg.shape[1]/2
        #     y = thisimg.shape[0]/2 - FinalPoints[i,1]
        #     # Try replacing Z with odom altitude, for fun...
        #     Z = -global_pos.position.z
        #     #Z = -FinalPoints[i,2]/1000
        #     # Populate Optical Flow Matrix for all points
        #     A[2*i:2*i+2] = np.array([[-1/Z,0,x/Z,x*y,-(1+x*x),y],[0,-1/Z,y/Z,(1+y*y), -x*y, -x]])
        #     b[2*i:2*i+2] = FinalDelta[i]/rate
        #     #print A
        #     #print b
        # #print A.shape
        # #print b.shape
        # # Linear least squares solver on optical flow equation
        # Results, res, rank, s = np.linalg.lstsq(A,b)
        # #print Results
        # #print Results.shape
        # position = Results[0:3]*rate
        # print("Position is {}".format(position))
        # print("Opt Flow Velocity m/s:", Results[0:3])
        # print("Opt Flow Rotations:", Results[3:])
        # print("Odometry Velocity m/s:", global_vel.linear)

        # # Then Integrate to get odometry
        
    frame = frame + 1
    lastImg = leftImg.copy()

if __name__ == '__main__':
    try:
        rundetection()
    except rospy.ROSInterruptException:
        pass
