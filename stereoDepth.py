#!/usr/bin/env python2

import cv2
import numpy as np

def sterioDepth(points,leftimg,rightimg,f,B,window,skipPixel,slide_dist):
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

	# hack to allow single point to work
	if points.shape[1] == 1:
		points[:,1] = [0,0]

	for point in points:
		#print("point ",i)
		# Assemble a range of X and Y points for a window
		#print point
		rangeX = range(point[1]-window,point[1]+window) #rows
		rangeY = range(point[0]-window,point[0]+window) #columns
		#print np.shape(rangeX)
		# windowpoints = np.array([rangeX,rangeY]) #rows,columns
		xx,yy = np.meshgrid(rangeX,rangeY)
		windowpoints = np.array([xx.flatten(),yy.flatten()]) #rows,columns
		#print point
		#print windowpoints
		#error = np.power(Leftwindowedimg[point[0],point[1]]-Rightwindowedimg[:,point[0]],2)
		
		# start window iteration at 0
		j = 0 # window number
		

		# set range of disparity values to those which lie within the image
		# Limit disparity range to -40 to + 40 to reduce iterations
		#lower = -point[1]+window
		#upper = Rightwindowedimg.shape[1]-point[1]-window
		lower = point[0]-slide_dist
		#lower = point[0]-20
		upper = point[0]+2
		if upper > rightimg.shape[1]+window-1:
			upper = rightimg.shape[1]+window-1
		if lower < window-1:
			lower = window-1
		d_range = range(lower-point[0],upper-point[0],skipPixel)
		#print d_range
		# initialize error matrix
		error = np.zeros(len(d_range))
		#print lower
		#print upper
		#print len(d_range)
		# iterate through the diaparity values

		Leftwindowedimg=Leftwindowedimg.astype('float32')
		Rightwindowedimg=Rightwindowedimg.astype('float32')
		#d_range=d_range.astype('float32')
		for d_temp in d_range:
			# Calculate sumsquared error between the feature point window and the particular disparity window
			error[j] = np.sum(np.square(np.subtract(Leftwindowedimg[windowpoints[0,:],windowpoints[1,:]],Rightwindowedimg[windowpoints[0,:],np.add(windowpoints[1,:],d_temp)])))
			#error = Leftwindowedimg[]-Rightwindowedimg[]
			temp=np.subtract(Leftwindowedimg[windowpoints[0,:],windowpoints[1,:]],Rightwindowedimg[windowpoints[0,:],np.add(windowpoints[1,:],d_temp)])
			
			# print('floating point error?')
			# print temp
			# print np.square(temp)

			j = j+1
		# the minimum error represents the disparity index which is a maximum
		#match = np.argmin(error)
		#print np.argmin(error)
		# print('errors')
		# print error
		
		
		d[i] = -d_range[np.argmin(error)]

		smolerror=error[np.argmin(error)]
		error[np.argmin(error)]=999999999999999999
		d2= error[np.argmin(error)]#-d_range[np.argmin(error)]
		# print 'smallet,2nd smallest'
		# print smolerror
		# print d2
		# Reject edge disparitys as NaN
		if d[i] <= 0:
			d[i] = 'NaN'
		if d[i] == d_range[0]:
			d[i] = 'NaN'
		#better be a unique thingy
		if smolerror> .95*d2:
			d[i] = 'NaN'
		# incriment point counter
		#print d[i]
		i = i+1
		

	Z = np.divide(f*B,d)
	# print Z
	return Z, d