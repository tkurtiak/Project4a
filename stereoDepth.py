#!/usr/bin/env python2

import cv2
import numpy as np

def sterioDepth(points,leftimg,rightimg,f,B,window,skipPixel):
	# points, feature points on img left to find on right img
	# 	Reccomend that we use as few points as possible to improve speed
	# left image - greyscale
	# right image - greyscale
	# f is camera focal length
	# B is distance between sterio camera pair
	# window is a scalar window dimention setting bounds around pixel to use for average intensity
	#	Default value is recommended 5.  
	#	Set higher for better matching but will cause additional compute time
	#	Set lower for lower compute time
	# skipPixel is the number of pixels between window centers
	#	Default is 1, ie do not skip any pixels.  
	# 	Set to 5 to speed up script
	#
	# epipolar line is a horixontal line for the case of parallel stereo cameras
	#
	# makes an average intensity 
	#kernel = np.ones((window, window), dtype=np.float32)/(window**2)  
	#Leftwindowedimg = leftimg#cv2.filter2D(leftimg, -1, kernel)
	#Rightwindowedimg = rightimg#cv2.filter2D(rightimg, -1, kernel)

	Leftwindowedimg = cv2.copyMakeBorder(leftimg, window, window, window, window, cv2.BORDER_REPLICATE, None)
	Rightwindowedimg = cv2.copyMakeBorder(rightimg, window, window, window, window, cv2.BORDER_REPLICATE, None)
	#Shift points by window size 
	points = points+window

	# initialize disparity array
	d = np.zeros(points.shape[0])
	i = 0
	for point in points:
		print("point ",i)
		# Assemble a range of X and Y points for a window
		rangeX = range(point[0]-window,point[0]+window)
		rangeY = range(point[1]-window,point[1]+window)
		windowpoints = np.array([rangeX,rangeY])
		#error = np.power(Leftwindowedimg[point[0],point[1]]-Rightwindowedimg[:,point[0]],2)
		
		# start window iteration at 0
		j = 0 # window number
		

		# set range of disparity values to those which lie within the image
		# Limit disparity range to -40 to + 40 to reduce iterations
		lower = -point[1]+window
		upper = Rightwindowedimg.shape[1]-point[1]-window
		if lower < -40:
			lower = -40
		if upper > 40:
			upper = 40
		d_range = range(lower,upper,skipPixel)
		# initialize error matrix
		error = np.zeros(len(d_range))

		#print len(d_range)
		# iterate through the diaparity values
		for d_temp in d_range:
			# Calculate sumsquared error between the feature point window and the particular disparity window
			error[j] = np.sum(np.power(Leftwindowedimg[windowpoints[0,:],windowpoints[1,:]]-Rightwindowedimg[windowpoints[0,:],windowpoints[1,:]+d_temp],2))
			j = j+1
		# the minimum error represents the disparity index which is a maximum
		#match = np.argmin(error)
		#print np.argmin(error)
		#print error
		
		
		d[i] = d_range[np.argmin(error)]
		# incriment point counter
		i = i+1

	Z = f*B/d
	return Z, d