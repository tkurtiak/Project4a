#!/usr/bin/env python2

import cv2
import numpy as np


def stereoDepthORB(leftimg,rightimg,f,B):
	



	orb = cv2.ORB_create()
	kp1, des1 = orb.detectAndCompute(leftimg,None)
	kp2, des2 = orb.detectAndCompute(rightimg,None)
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = bf.match(des1,des2)

	# print np.shape(matches)



	# Extract points from matches
	matches = sorted(matches, key = lambda x:x.distance) 
	print len(kp1)
	num_matches= int(.5*len(matches))

	points = np.zeros((num_matches,2))
	delta = np.zeros((num_matches,2))
	dist = np.zeros((num_matches))
	matchMask = np.zeros((num_matches,2))

		  
	# ratio test as per Lowe's paper
	# source: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
	for i in range(0,num_matches):
		print matches[i].trainIdx
		points[i] = kp2[matches[i].trainIdx].pt
		delta[i] = np.subtract(kp1[matches[i].queryIdx].pt,kp2[matches[i].trainIdx].pt)   
		dist[i] = np.sqrt(delta[i,0]**2+delta[i,1]**2)#matches[i][0].distance

		if np.abs(delta[i,0]/delta[i,1]) > 10:
			#x is bigger than y, so is more or less horizontal
			matchMask[i]=[1,0]     
		


	# print delta
	matchMaskbool = matchMask.astype('bool')
	points = points[matchMaskbool[:,0]]
	delta = delta[matchMaskbool[:,0]]
	dist = dist[matchMaskbool[:,0]]

	## Filter out bad feature matches
	# print delta


	# print dist



	d=dist
	Z = np.divide(f*B,d)
	# print Z
	return Z, d, points.astype('int')