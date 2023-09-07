'''
图像金字塔
可以在不同分辨率下进行图像处理和分析。在图像处理中，常常会使用金字塔来进行特征提取、目标检测和图像融合等任务。
https://docs.opencv.org/4.x/dc/dff/tutorial_py_pyramids.html
'''

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('messi.png')
lower_reso = cv.pyrUp(img) # 向上采样，即增加图像的分辨率。它通过插值的方式在图像的每个像素之间插入新的像素，从而使图像的尺寸变大。
higher_reso  = cv.pyrDown(img) # 向下采样，即减小图像的分辨率。它通过对图像进行平滑并且采样，从而使图像的尺寸变小。下采样可以用于图像缩小或者在图像金字塔的下一层进行处理。

cv.imshow('original', img)
cv.imshow('lower_reso', lower_reso)
cv.imshow('higher_reso', higher_reso)

cv.waitKey(0)
cv.destroyAllWindows()
