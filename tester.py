import cv2 
import numpy as np
import stereoDepth as SD

left = cv2.imread("project_4_bags_and_cams/HelixLeft/frame0003.jpg")
right = cv2.imread("project_4_bags_and_cams/HelixRight/frame0003.jpg")

leftGray = cv2.cvtColor(left,cv2.COLOR_BGR2GRAY)
rightGray = cv2.cvtColor(right,cv2.COLOR_BGR2GRAY)


count = 0
subsample = 2 
points = np.zeros(((leftGray.shape[0]//subsample)*(leftGray.shape[1]//subsample),2))
for i in range(0,left.shape[0]-1,subsample):
	for j in range(0,left.shape[1]-1,subsample):
		points[count] = [i,j]
		count = count+1
points = points.astype(int)

f = 202
B = 200
window = 5
skipPixel = 2
Z,d = SD.sterioDepth(points,leftGray,rightGray,f,B,window,skipPixel)

dispimg = np.reshape(d,(leftGray.shape[0]//subsample,leftGray.shape[1]//subsample))
depthimg = np.reshape(Z,(leftGray.shape[0]//subsample,leftGray.shape[1]//subsample))


cv2.imshow('disparity',dispimg)
cv2.imshow('depth',depthimg)
cv2.imshow('Original',leftGray)
cv2.waitKey(0)
cv2.destroyAllWindows()