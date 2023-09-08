'''
图像梯度 
https://docs.opencv.org/4.x/d5/d0f/tutorial_py_gradients.html
'''

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('shudu.png', cv.IMREAD_GRAYSCALE)

laplacian = cv.Laplacian(img,cv.CV_64F)
# 1,0 表示在x方向求梯度 0,1 表示在y方向求梯度, ksize是Sobel算子的大小
sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()