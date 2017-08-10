import cv2
import numpy as np
from math import exp
from matplotlib import pyplot as plt

img = cv2.imread('lena.jpg',0)

def convolution(img, kernel):
	r=img.shape[0]
	c=img.shape[1]
	kr=kernel.shape[0]
	kc=kernel.shape[1]

	nimg = np.zeros((r,c),np.float32)

	for i in range(r):
		for j in range(c):
			v=0
			ti=i-kr/2
			tj=j-kc/2
			for k in range(kr):
				for l in range(kc):
					w=kernel[k][l]
					if(ti+k>=0 and ti+k<r and tj+l>=0 and tj+l<c):
						v+=(float(w)*img[ti+k][tj+l])
			nimg[i][j]=v
	
	return nimg

def sharpen_edge(img):
	n=5
	kernel = -np.ones((n,n),np.float32)/(n*n)
	kernel[n/2][n/2] = float((n*n)-1)/(n*n)
	return convolution(img,kernel)

def sobel_edge(img):
	x=np.zeros((3,3),np.float32)
	y=np.zeros((3,3),np.float32)
	x[0]=[1,0,-1]
	x[1]=[2,0,-2]
	x[2]=[1,0,-1]
	y[0]=[1,2,1]
	y[1]=[0,0,0]
	y[2]=[-1,-2,-1]
	Gx = convolution(img,x)
	Gy = convolution(img,y)
	G = np.power(np.add(np.multiply(Gx,Gx),np.multiply(Gy,Gy)),0.5)
	return G

def gauss_smoothing(img):
	n=5
	sigma=float(n)/5
	kernel=np.zeros((n,n),np.float32)
	mid=n/2
	for i in range(-mid,mid+1):
		for j in range(-mid,mid+1):
			kernel[i][j]=exp(-(i*i+j*j)/(2*sigma*sigma))

	return convolution(img,kernel)

def median_smoothing(img):
	n=5	
	r=img.shape[0]
	c=img.shape[1]
	nimg=np.zeros((r,c),np.float32)
	for i in range(r):
		for j in range(c):
			ls=[]
			for k in range(-(n/2),(n/2)+1):
				for l in range(-(n/2),(n/2)+1):
					if(i+k>=0 and i+k<r and j+l>=0 and j+l<c):
						ls.append(img[i+k][j+l])
			nimg[i][j]=np.median(np.array(ls))
	return nimg

edges1 = sharpen_edge(img)
edges2 = sobel_edge(img)
smoothing1 = gauss_smoothing(img)
smoothing2 = median_smoothing(img)

plt.subplot(231),plt.imshow(img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(232),plt.imshow(edges1, cmap='gray')
plt.title('Sharpen Edge'), plt.xticks([]), plt.yticks([])
plt.subplot(233),plt.imshow(edges2, cmap='gray')
plt.title('Sobel Edge'), plt.xticks([]), plt.yticks([])
plt.subplot(234),plt.imshow(smoothing1, cmap='gray')
plt.title('Gaussian Smoothing'), plt.xticks([]), plt.yticks([])
plt.subplot(235),plt.imshow(smoothing2, cmap='gray')
plt.title('Median Smoothing'), plt.xticks([]), plt.yticks([])

plt.show()
