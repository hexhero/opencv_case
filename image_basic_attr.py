import numpy as np
import cv2 as cv

img = cv.imread('messi.png')
cv.namedWindow('img')

# 操作单个像素
img.item(10,10,2) # 10,10,2 代表第10行，第10列，第2个通道的像素值 0:blue 1:green 2:red
img.itemset((10,10,2), 255) # 将第10行，第10列，第2个通道的像素值设置为255

'''
img.shape: 返回值是一个包含行数，列数，通道数的元组,
如果图像是灰度图像，则返回的元组仅包含行数和列数，因此这是检查加载的图像是灰度图像还是彩色图像的好方法。

'''
print("shape:", img.shape) 
print("size:",img.size) # 返回图像的像素数目

'''
img.dtype 在调试时非常重要，因为 OpenCV-Python 代码中的大量错误是由无效数据类型引起的。
'''
print("dtype",img.dtype) # 返回图像的数据类型 


