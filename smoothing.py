import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

'''
图像平滑
https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
'''

img = cv.imread('messi.png')

# 2D卷积模糊 
kernel = np.ones((5, 5), np.float32) / 25 # 5x5的卷积核 /25 为了归一化，否则会变暗
dst1 = cv.filter2D(img, -1, kernel)

# 平均模糊 与2D卷积模糊类似，但是卷积核为归一化的均值
dst2 = cv.blur(img, (5, 5))

# 高斯模糊 与均值模糊类似，但是使用不同的卷积核，卷积核的数值满足高斯分布
dst3 = cv.GaussianBlur(img,(5,5),0)

# 中值模糊 用于去除椒盐噪声
dst4 = cv.medianBlur(img, 5)

# 双边滤波 保留边缘
dst5 = cv.bilateralFilter(img, 9, 75, 75) # 9为卷积核大小，75为颜色空间标准差，75为坐标空间标准差

plt.subplot(231), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(232), plt.imshow(dst1), plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.subplot(233), plt.imshow(dst2), plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.subplot(234), plt.imshow(dst3), plt.title('Gaussian Blurred')
plt.xticks([]), plt.yticks([])
plt.subplot(235), plt.imshow(dst4), plt.title('Median Blurred')
plt.xticks([]), plt.yticks([])
plt.subplot(236), plt.imshow(dst5), plt.title('Bilateral Filtering')
plt.xticks([]), plt.yticks([])
plt.show()




