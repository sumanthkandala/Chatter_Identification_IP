import cv2
import numpy as np
import scipy.misc
import scipy.ndimage.filters
from matplotlib import pyplot as plt
import math 
from skimage.feature import greycomatrix

def Ga_calculation(image):
	ga=0.0
	for i in range(image.shape[1]):
		ga+=abs(np.mean(image[:,i])-np.mean(image))
	return ga/image.shape[1]

############################################################

#Reading image
img = cv2.imread('chatter.jpg')
alpha = 1.3
beta = 0
threshold = 240
cv2.imshow('original',img)
Ga_original = Ga_calculation(img)
img = cv2.addWeighted(img,alpha,np.zeros(img.shape,img.dtype),0,beta)
bright_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,bright_img = cv2.threshold(bright_img,threshold,255,cv2.THRESH_BINARY)  
#bright_img = 255-bright_img
#Gaussian Blur
#img = cv2.GaussianBlur(img, (3, 3), 0)
#Converting the image to grays	cale
cv2.imwrite('grayscale.jpg',bright_img)
#Calculating the Ga value of the original image
print("Ga_original:",Ga_original)

############################################################
#Edge enhancement and its Ga
kernel = np.ones((3,3),np.float32)*(-1)
kernel[1,1] = 8
#img = cv2.Laplacian(img,cv2.CV_64F
enh_img1 = cv2.filter2D(bright_img,-1,kernel)
#enh_img = brighten(enh_img)
Ga_edge_enhanced=Ga_calculation(enh_img1)
# ShF = 100              #Sharpening factor!
# enh_img = enh_img*ShF/np.amax(enh_img) 
# enh_img = cv2.convertScaleAbs(enh_img)
cv2.imwrite('laplacian.jpg',enh_img1)
# Recalculate Ga after edge enhancement by Laplacian kernel
print("Ga_edge_enhanced", Ga_edge_enhanced)

#############################################################
# Magnification by the factor of 2 using cubic interpolation convolution
height, width = bright_img.shape[:2]
mag_img = cv2.resize(bright_img,(2*width, 2*height), interpolation=cv2.INTER_CUBIC)
enh_img = cv2.filter2D(mag_img,-1,kernel)
Ga_magnified = Ga_calculation(enh_img)
# ShF = 100 #Sharpening factor!
# enh_img = enh_img*ShF/np.amax(enh_img) 
# enh_img = cv2.convertScaleAbs(enh_img)
cv2.imwrite('laplacian_mag.jpg',enh_img)
print("Ga_magnified:",Ga_magnified)

#Variance of image
threshold = 240
ret,enh_img1 = cv2.threshold(enh_img1,threshold,255,cv2.THRESH_BINARY)  
variance = np.var(enh_img1,ddof = 1)/(enh_img1.size)
std_deviation = variance**0.5
print("std_deviation" ,std_deviation)
print("mean", np.mean(img))


#RMS Value
mean = np.mean(img)
rms = (img)**2
rms = float((rms.sum()/(rms.shape[0]*rms.shape[1]))**0.5)
print("rms",rms)

#Calculation of Optical Roughness parameter
orp = std_deviation/rms
print("orp",orp)

glcm_matrix = greycomatrix(enh_img, [1], [0], symmetric=True, normed=True)
#print(glcm_matrix)

#Energy Calcualation
energy = glcm_matrix**2
energy = float(energy.sum())
print("energy",energy)

#Entropy Calculation
entropy = 0.0
for i in range(glcm_matrix.shape[0]):
	for j in range(glcm_matrix.shape[1]):
		if(glcm_matrix[i][j]!=0):
			entropy += (glcm_matrix[i][j]*math.log(glcm_matrix[i][j]))/math.log(2)
print("entropy",entropy*-1)

cv2.waitKey(0)
cv2.destroyAllWindows()

#Statistical Analysis
plt.hist(enh_img.ravel(),256,[0,255]); 
plt.savefig('hist.png')
plt.show()