import cv2
import numpy as np
from math import exp
from matplotlib import pyplot as plt

img = cv2.imread('lena.jpg',0)

def lowpass(img):
	radius = 50
	dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
	dft_shift = np.fft.fftshift(dft)
	r, c = img.shape
	cr, cc = r/2, c/2
	mask = np.zeros((r,c,2),np.uint8)
	mask[cr-radius:cr+radius, cc-radius:cc+radius] = 1
	fshift = dft_shift*mask
	f_ishift = np.fft.ifftshift(fshift)
	nimg = cv2.idft(f_ishift)
	nimg = cv2.magnitude(nimg[:,:,0],nimg[:,:,1])
	return nimg

def highpass(img):
	radius = 20
	dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
	dft_shift = np.fft.fftshift(dft)
	r, c = img.shape
	cr, cc = r/2, c/2
	mask = np.ones((r,c,2),np.uint8)
	mask[cr-radius:cr+radius, cc-radius:cc+radius] = 0
	fshift = dft_shift*mask
	f_ishift = np.fft.ifftshift(fshift)
	nimg = cv2.idft(f_ishift)
	nimg = cv2.magnitude(nimg[:,:,0],nimg[:,:,1])
	return nimg

edges = highpass(img)
smoothing = lowpass(img)

plt.subplot(131),plt.imshow(img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(edges, cmap='gray')
plt.title('Edge Detection'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(smoothing, cmap='gray')
plt.title('Noise Removal'), plt.xticks([]), plt.yticks([])

plt.show()
