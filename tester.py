#!/usr/bin/env python

import cv2 
import numpy as np
import stereoDepth as SD
from matplotlib import pyplot as plt
#from sklearn import linear_model, datasets

left = cv2.imread("images/aloeL.jpg")
right = cv2.imread("images/aloeR.jpg")
# left = cv2.imread("left_test.jpg")
# right = cv2.imread("right_test.jpg")
#left = cv2.imread("project_4_bags_and_cams/aloeL.jpg")
#right = cv2.imread("project_4_bags_and_cams/aloeR.jpg")
# left = cv2.imread("images/HelixLeft/frame0003.jpg")
# right = cv2.imread("images/HelixRight/frame0003.jpg")

leftGray = cv2.cvtColor(left,cv2.COLOR_BGR2GRAY)
rightGray = cv2.cvtColor(right,cv2.COLOR_BGR2GRAY)

# leftGray[430:450,530:560]=255

method = 2
if method == 1:
	sift = cv2.xfeatures2d.SIFT_create(700)
	kp, des = sift.detectAndCompute(leftGray,None)
	points = np.zeros((len(kp),2))
	ptnum = 0
	for p in kp:
		points[ptnum] = np.array(kp[ptnum].pt)
		ptnum= ptnum+1
	points = points.astype(int)
	#print points
else:
	count = 0
	subsample = 20
	#points = np.array([[325,233]])#
	points = np.zeros((np.floor(leftGray.shape[0]/subsample)*np.floor(leftGray.shape[1]/subsample),2))

	print leftGray.shape
	print points.shape
	for i in range(0,left.shape[1]-subsample+1,subsample):
		for j in range(0,left.shape[0]-subsample+1,subsample):
			points[count] = [i,j]
			count = count+1
	points = points.astype(int)
	print count


f = 202
B = 5
window = 4
skipPixel = 2
slide_dist = 150 #how far to the left do we look for the same shit
Z,d = SD.sterioDepth(points,leftGray,rightGray,f,B,window,skipPixel, slide_dist)


print d
nanMask = np.array(~np.isnan(d))
points = points[nanMask,:]
d = d[nanMask]
Z = Z[nanMask]
print np.median(d)
print np.mean(d)

depth_image=leftGray.copy()*0
dnormalized= np.divide(d,np.max(d))*255
#dnormalized = d
dnormalized= dnormalized.astype(int)
for i in range(points.shape[0]):
	# depth_image[points[i,1],points[i,0]]=dnormalized[i]
	cv2.circle(depth_image,(points[i,0],points[i,1]),5,dnormalized[i],-1)

cv2.imshow('depths?',depth_image)
#cv2.imshow('OG',leftGray)
cv2.imshow('OriginalL',leftGray)
cv2.imshow('OriginalR',rightGray)

# print Z
# print d

#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html#sklearn.linear_model.RANSACRegressor
# Robustly fit linear model with RANSAC algorithm
# ransac = linear_model.RANSACRegressor()
# ransac.fit(points, Z)
# inlier_mask = ransac.inlier_mask_
# outlier_mask = np.logical_not(inlier_mask)
# Z_filter =  Z[inlier_mask]


# dispimg = np.reshape(np.divide(d,np.max(d))*255,(leftGray.shape[0]//subsample,leftGray.shape[1]//subsample)).astype(int)
# depthimg = np.reshape(np.divide(Z,np.max(Z))*255,(leftGray.shape[0]//subsample,leftGray.shape[1]//subsample)).astype(int)
# mask = np.reshape(inlier_mask.astype('uint8'),(leftGray.shape[0]//subsample,leftGray.shape[1]//subsample))
# Zimg = cv2.bitwise_and(depthimg,depthimg,mask = mask) 


# #plt.imshow(dispimg,cmap='gray')
# plt.imshow(Zimg,cmap='gray')



# #cv2.imshow('disparity',dispimg)
# # cv2.imshow('depth',depthimg)
# cv2.imshow('Original',leftGray)

# plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()