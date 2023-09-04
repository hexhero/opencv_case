import numpy as np
import cv2 as cv

img1 = cv.imread('messi.png')
img2 = cv.imread('leo.png')
img2 = cv.flip(img2, 1)

r,c = 7,220

rows, cols, channels = img2.shape
roi = img1[r:rows+r, c:cols+c] # 图像1的感兴趣区域

img2gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY) # convert to gray
ret, mask_inv = cv.threshold(img2gray, 70, 255, cv.THRESH_BINARY) # 二值化 白色的部分为255 黑色的部分为0，阈值是70
mask = cv.bitwise_not(mask_inv) # 取反
img1_bg = cv.bitwise_and(roi, roi, mask=mask_inv) # 与操作
img2_fg = cv.bitwise_and(img1_bg, img2, mask=mask) # 与操作
dst = cv.add(img1_bg, img2_fg)
img1[r:rows+r, c:cols+c ] = dst

cv.imshow('mask', img1)
cv.waitKey(0)
