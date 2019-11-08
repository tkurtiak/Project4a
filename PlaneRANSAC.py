
#!/usr/bin/env python
import numpy as np


 

def PlaneRANSAC(points):
	#fits a plane to x,y,z points in an Nx3 array

	N=10 #set the number of iterations you want to run here

	bestscore=9999999999 #want a small score
	

	for j in range(N):
		#find 3 unique indicies
		#CAVEMAN STYLE
		#*grunts
		sample_indxs=np.random.randint(points.shape[0], size=3)
		unique=False
		while unique==False:
			if (sample_indxs[0]==sample_indxs[1]) or (sample_indxs[0]==sample_indxs[2]) or (sample_indxs[2]==sample_indxs[1]):
				sample_indxs=np.random.randint(points.shape[0], size=3)
			else:
				unique=True

		#equation of a plane Ax+By+Cz-D=0
		#find coefs A, B, C, D

		#normal vector is cross of two vectors in plane is also (A,B,C)
		normal=np.cross(points[sample_indxs[1],:] - points[sample_indxs[0],:],points[sample_indxs[2],:] - points[sample_indxs[0],:])
		D= -normal[0]*points[sample_indxs[0],0] -normal[1]*points[sample_indxs[0],1] -normal[2]*points[sample_indxs[0],2]

		dists=np.zeros(points.shape[0])
		for i in range(points.shape[0]):
			point=points[i,:]
			dists[i]= np.abs( normal[0]*point[0] + normal[1]*point[1] + normal[2]*point[2] + D) / np.sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2])

		score= np.median(dists)
		if score<bestscore:
			bestscore=score
			bestdists=dists
			bestnormal=normal
			bestD=D
	#skibidy
	indicies=np.squeeze(np.where(bestdists<= bestscore))
	return points[indicies,:], indicies, bestnormal, bestD

#TEST SHIT HERE IF YOU WANT BRO
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

points= np.random.randint(50, size=(100,3))

points[np.random.randint(points.shape[0], size=40),2]=25

points_close, indxs, normal, D = PlaneRANSAC(points)
print points_close.shape
print indxs

# plot the surface
plt3d = plt.figure().gca(projection='3d')

xx, yy = np.meshgrid(range(50), range(50))
zz = (-normal[0] * xx - normal[1] * yy - D) * 1. /normal[2]

plt3d.plot_surface(xx, yy, zz, alpha=0.2)

# Ensure that the next plot doesn't overwrite the first plot
ax = plt.gca()
ax.hold(True)
ax.scatter(points[:,0], points[:,1], points[:,2], color='red')
ax.scatter(points_close[:,0], points_close[:,1], points_close[:,2], color='green')
plt.show()

