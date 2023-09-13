import numpy as np
import cv2 as cv

filename = 'mofang.png'
img = cv.imread(filename)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray = np.float32(gray)

'''
参数1：输入图像的灰度图像（单通道图像）。
参数2：角点检测中使用的窗口大小，表示窗口的大小为2x2像素。
参数3：Sobel算子的孔径大小，用于计算图像梯度。
参数4：角点响应函数中的自由参数k，控制角点检测的敏感度。
'''
dst = cv.cornerHarris(gray,4,3,0.09)  
#result is dilated for marking the corners, not important
dst = cv.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[255,0,0]
cv.imshow('dst',img)
if cv.waitKey(0) & 0xff == 27:
 cv.destroyAllWindows()