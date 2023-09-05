import numpy as np
import cv2 as cv

'''
几何变换
'''

img = cv.imread('messi.png')
rows, cols = img.shape[:2]

# 缩放
scaling = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC) # 比例缩放
# scaling = cv.resize(img, (300, 200), interpolation=cv.INTER_CUBIC) # 固定大小

# 在x轴方向上平移100个单位，y轴方向上平移50个单位

M1 = np.float32([[1, 0, 100], [0, 1, 50]])
translation = cv.warpAffine(img, M1, (cols, rows))

# 旋转
M2 = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),90,1)
rotation = cv.warpAffine(img,M2,(cols,rows))

cv.imshow('scaling', scaling)
cv.imshow('translation', translation)
cv.imshow('rotation', rotation)
cv.waitKey(0)

