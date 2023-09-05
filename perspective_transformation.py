import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

'''
视角转换
'''
img = cv.imread('keyboard.jpg')

pts1 = np.float32([[168,429],[1476,451],[35,840],[1637,826]])
pts2 = np.float32([[0,0],[1000,0],[0,300],[1000,300]])

M = cv.getPerspectiveTransform(pts1,pts2)
dst = cv.warpPerspective(img,M,(1000,300))

# 画出4个点
pts1 = pts1.astype(int)
cv.polylines(img, [pts1], isClosed=True, color=(0, 255, 0), thickness=3)

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()