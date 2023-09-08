'''
作业 - 检测像素到区域的距离
'''
import numpy as np
import cv2 as cv

img = cv.imread('mask.png')
# img = cv.imread('star1.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 127, 255, 0)  # 二值化
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cnt = contours[0]

# 循环img像素
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        dist = cv.pointPolygonTest(cnt, (j, i), True)
        if dist > -2 and dist < 2:
            img[i, j] = [255, 255, 255]
        elif dist > 0:
            dist = abs(dist)
            img[i, j] = [0, 0, max(255-dist*10, 0)]
        elif dist < 0:
            dist = abs(dist)
            img[i, j] = [max(255-dist, 0), 0, 0]

cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()
