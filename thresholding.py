'''
阈值处理
'''

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('keyboard.jpg')

pts1 = np.float32([[168,429],[1476,451],[35,840],[1637,826]])
pts2 = np.float32([[0,0],[1000,0],[0,300],[1000,300]])

# 视角转换
M = cv.getPerspectiveTransform(pts1,pts2)
dst = cv.warpPerspective(img,M,(1000,300))

# 画出4个点
pts3 = pts1.astype(int)
cv.polylines(img, [pts3], isClosed=True, color=(0, 255, 0), thickness=3)

# 灰度处理
th1 = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
th2 = th1

# 二值化阈值处理
ret,xh1 = cv.threshold(th2,55,255,cv.THRESH_BINARY)

# 自动阈值 阈值是邻域面积的平均值减去常数C。
xh2 = cv.adaptiveThreshold(th2,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,7,4)

# 自动阈值 阈值是邻域值的高斯加权和减去常数 C
xh3 = cv.adaptiveThreshold(th2,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,7,3)

# Otsu 阈值，指图像直方图确定最佳全局阈值。不用手动指定阈值
ret2,xh4 = cv.threshold(th2,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

# 高斯滤波后的 Otsu 阈值处理, 有效的处理噪点
blur = cv.GaussianBlur(th2,(7,7),0) # 7*7的高斯滤波，表示在模糊过程中，每个像素周围的邻域大小为 7x7。 标准差为0，表示由函数自动计算
ret3,xh5 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)


plt.subplot(331),plt.imshow(img),plt.title('Input') # 331 3行3列第一个
plt.subplot(332),plt.imshow(dst),plt.title('Output')
plt.subplot(333),plt.imshow(xh1),plt.title('THRESH_BINARY')
plt.subplot(334),plt.imshow(xh2),plt.title('ADAPTIVE_THRESH_MEAN_C')
plt.subplot(335),plt.imshow(xh3),plt.title('ADAPTIVE_THRESH_GAUSSIAN_C')
plt.subplot(336),plt.imshow(xh4),plt.title('THRESH_BINARY + THRESH_OTSU')
plt.subplot(337),plt.imshow(xh5),plt.title('BLUR + OTSU')
plt.subplot(338),plt.imshow(th2),plt.title('GRAY')

plt.show()