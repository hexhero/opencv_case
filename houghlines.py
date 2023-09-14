'''
霍夫线变换
'''

import cv2 as cv
import numpy as np

img = cv.imread(cv.samples.findFile('minesweeper.png'))
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray,50,150,apertureSize = 3)

# 霍夫变换
# lines = cv.HoughLines(edges,1,np.pi/180,130) # np.pi/180表示theta的精度，即霍夫空间中角度的步长，这里使用弧度制。 130 是阈值
# for line in lines:
#     rho,theta = line[0]
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))
#     cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)
 
# 概率霍夫变换  概率霍夫变换是我们看到的霍夫变换的优化。
# 参数1 (edges)：输入图像的边缘图像，通常是通过 Canny 边缘检测算法得到的。
# 参数2 (1)：距离分辨率，表示 Hough 变换中的距离分辨率。它决定了在 Hough 空间中每个像素点的距离单位。值为1表示像素级别的距离分辨率。
# 参数3 (np.pi/180)：角度分辨率，表示 Hough 变换中的角度分辨率。它决定了在 Hough 空间中每个角度的步长。值为 np.pi/180 表示以弧度为单位的角度分辨率，相当于每个角度步长为1度。
# 参数4 (100)：阈值，表示检测直线的累加器阈值。只有当某个直线在累加器中的投票数超过该阈值时，才会被认为是有效的直线。
# 参数5 (minLineLength=100)：最小线段长度，表示检测到的直线的最小长度。小于该长度的线段将被忽略。
# 参数6 (maxLineGap=10)：最大线段间隔，表示同一条直线上两个线段之间的最大间隔。如果两个线段的间隔超过该值，则它们将被认为是不同的直线。
lines = cv.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
print(len(lines))
for line in lines:
    x1,y1,x2,y2 = line[0]
    cv.line(img,(x1,y1),(x2,y2),(255,0,0),2)

cv.imshow('houghlines',img)
cv.waitKey(0)
cv.destroyAllWindows()
