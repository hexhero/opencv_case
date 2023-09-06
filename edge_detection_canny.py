'''
Canny 边缘检测
https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
'''
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

cv.namedWindow('canny')

img = cv.imread('messi.png', cv.IMREAD_GRAYSCALE)

cv.createTrackbar('Min','canny',0,255,lambda x:None)
cv.createTrackbar('Max','canny',0,255,lambda x:None)

cv.imshow('original', img)

while 1:
    min = cv.getTrackbarPos('Min','canny')
    max = cv.getTrackbarPos('Max','canny')
    edges = cv.Canny(img,min,max) # 100是最小阈值，200是最大阈值
    cv.imshow('canny', edges)
    key = cv.waitKey(1)
    if key == 27:
        break
    
cv.destroyAllWindows()

