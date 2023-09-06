'''
形态变换
https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
'''

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('morphological.png')

# 侵蚀
kernel = np.ones((5, 5), np.uint8)
erosion = cv.erode(img, kernel, iterations=1)

# 膨胀
dilation = cv.dilate(img, kernel, iterations=1)

# Opening/开运算: 先侵蚀再膨胀, 消除背景中的噪声
opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

# Closing/闭运算 : 先膨胀再侵蚀, 消除前景中的噪声
closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

# Morphological Gradient: 膨胀图像减去侵蚀图像
gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)

# Top Hat: 输入图像减去开运算图像
tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)

# Black Hat: 闭运算图像减去输入图像
blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)

plt.subplot(181), plt.imshow(img), plt.title('Original')
plt.subplot(182), plt.imshow(erosion), plt.title('Erosion')
plt.subplot(183), plt.imshow(dilation), plt.title('Dilation')
plt.subplot(184), plt.imshow(opening), plt.title('Opening')
plt.subplot(185), plt.imshow(closing), plt.title('Closing')
plt.subplot(186), plt.imshow(gradient), plt.title('Gradient')
plt.subplot(187), plt.imshow(tophat), plt.title('Top hat')
plt.subplot(188), plt.imshow(blackhat), plt.title('Black hat')
plt.show()