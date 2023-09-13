'''
SIFT（Scale-Invariant Feature Transform）是一种用于图像特征提取和匹配的算法。它由David Lowe在1999年提出，并在2004年发表了一篇经典的论文。SIFT算法的主要目标是在不同尺度和旋转条件下提取出具有独特性质的关键点，这些关键点在图像中具有良好的不变性。
'''
import numpy as np
import cv2 as cv

img = cv.imread('mofang.png')
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.SIFT_create()
kp = sift.detect(gray,None)
img=cv.drawKeypoints(gray,kp,img)

sift = cv.SIFT_create()
kp, des = sift.detectAndCompute(gray,None) # kp 是关键点列表

print(kp)

cv.imshow('dst',img)
cv.waitKey(0)
cv.destroyAllWindows()

