'''
OpenCV 反向投影

它基于图像的颜色直方图，通过将目标对象的颜色分布映射回输入图像，从而实现目标区域的提取。

https://docs.opencv.org/4.x/dc/df6/tutorial_py_histogram_backprojection.html

'''

import numpy as np
import cv2 as cv
img = cv.imread('messi.png')

y1,y2,x1,x2 = 610, 640, 35, 140
roi = img[y1:y2, x1:x2]
hsv = cv.cvtColor(roi,cv.COLOR_BGR2HSV)
target = img # 100:200 是高度，200:500 是宽度

assert target is not None, "file could not be read, check with os.path.exists()"
hsvt = cv.cvtColor(target,cv.COLOR_BGR2HSV)
# 计算对象直方图
roihist = cv.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
# 标准化直方图并应用反投影
cv.normalize(roihist,roihist,0,255,cv.NORM_MINMAX)
dst = cv.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)
# 现在用圆盘进行卷积
disc = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
cv.filter2D(dst,-1,disc,dst)

# 阈值和二进制 AND
ret,thresh = cv.threshold(dst,50,255,0)
thresh = cv.merge((thresh,thresh,thresh))
res = cv.bitwise_and(target,thresh)
# res = np.vstack((target,thresh,res))
cv.rectangle(target,(x1,y1),(x2,y2),(0,255,0),3)
res = np.hstack((target,thresh,res))

cv.imshow('res',res)
cv.waitKey(0)
cv.destroyAllWindows()