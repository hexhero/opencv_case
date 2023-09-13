'''
Shi-Tomasi 角点检测器， 与 Harris 角点检测器相比显示出了更好的结果。
'''

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('mofang.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
'''
参数一：输入的灰度图像，用于角点检测。
参数二：需要检测的角点数量，这里设置为 25。
参数三：角点的质量水平阈值，用于筛选角点。较小的值会选择更多的角点，但质量可能较低。
参数四：角点之间的最小距离。如果两个角点之间的距离小于这个值，其中一个角点将被丢弃。
'''
corners = cv.goodFeaturesToTrack(gray,50,0.08,25)
corners = np.intp(corners)
for i in corners:
    x,y = i.ravel()
    cv.circle(img,(x,y),3,255,-1)
plt.imshow(img),plt.show()