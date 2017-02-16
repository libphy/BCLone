import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
cwd = os.getcwd()
samplepath = '2016_12_01_13_31_12_937.jpg'
left = cwd+'/data/IMG/left_'+samplepath
right = cwd+'/data/IMG/right_'+samplepath
center = cwd+'/data/IMG/center_'+samplepath
imL = cv2.cvtColor(cv2.imread(left), cv2.COLOR_RGB2GRAY)
imR = cv2.cvtColor(cv2.imread(right), cv2.COLOR_RGB2GRAY)
imC = cv2.cvtColor(cv2.imread(center), cv2.COLOR_RGB2GRAY)

stereo = cv2.StereoBM_create(numDisparities=32, blockSize=19)
disparity = stereo.compute(imL,imR)
plt.imshow(disparity,'gray')
plt.show()

plt.figure()
plt.subplot(3,1,1)
plt.imshow(imL,cmap='gray')
plt.subplot(3,1,2)
plt.imshow(imC,cmap='gray')
plt.subplot(3,1,3)
plt.imshow(imR,cmap='gray')
plt.show()

plt.imshow((imL+imR)/2-imC,cmap='gray')
plt.imshow(abs(imL-imR),cmap='gray')
