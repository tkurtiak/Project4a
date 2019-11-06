import cv2 
import numpy as np
import stereoDepth as SD
from matplotlib import pyplot as plt
from sklearn import linear_model, datasets

left = cv2.imread("project_4_bags_and_cams/HelixLeft/frame0003.jpg")
right = cv2.imread("project_4_bags_and_cams/HelixRight/frame0003.jpg")
#left = cv2.imread("project_4_bags_and_cams/left_test.jpg")
#right = cv2.imread("project_4_bags_and_cams/right_test.jpg")


leftGray = cv2.cvtColor(left,cv2.COLOR_BGR2GRAY)
rightGray = cv2.cvtColor(right,cv2.COLOR_BGR2GRAY)


count = 0
subsample = 10 
#points = np.array([[325,233]])#
points = np.zeros(((leftGray.shape[0]//subsample)*(leftGray.shape[1]//subsample),2))
for i in range(0,left.shape[1]-1,subsample):
	for j in range(0,left.shape[0]-1,subsample):
		points[count] = [i,j]
		count = count+1
points = points.astype(int)

f = 202
B = 30
window = 20
skipPixel = 1
Z,d = SD.sterioDepth(points,leftGray,rightGray,f,B,window,skipPixel)
print Z
print d

#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html#sklearn.linear_model.RANSACRegressor
# Robustly fit linear model with RANSAC algorithm
ransac = linear_model.RANSACRegressor()
ransac.fit(points, Z)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
Z_filter =  Z[inlier_mask]


dispimg = np.reshape(np.divide(d,np.max(d))*255,(leftGray.shape[0]//subsample,leftGray.shape[1]//subsample)).astype(int)
depthimg = np.reshape(np.divide(Z,np.max(Z))*255,(leftGray.shape[0]//subsample,leftGray.shape[1]//subsample)).astype(int)
mask = np.reshape(inlier_mask.astype('uint8'),(leftGray.shape[0]//subsample,leftGray.shape[1]//subsample))
Zimg = cv2.bitwise_and(depthimg,depthimg,mask = mask) 

# cv2.imshow('disparity',dispimg)
# cv2.imshow('depth',depthimg)
# cv2.imshow('Original',leftGray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
plt.imshow(dispimg,cmap='gray')
plt.imshow(Zimg,cmap='gray')
plt.show()