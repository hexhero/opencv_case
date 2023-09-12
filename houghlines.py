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
# minLineLength=100：表示检测到的直线的最小长度。小于该长度的直线将被丢弃。
# maxLineGap=10：表示将同一直线上的断裂部分连接在一起的最大允许间隙。
# 函数返回一个包含检测到的直线的数组，每条直线由起点和终点的坐标表示。可以使用这些坐标来在图像上绘制检测到的直线。
lines = cv.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
print(len(lines))
for line in lines:
    x1,y1,x2,y2 = line[0]
    cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv.imshow('houghlines',img)
cv.waitKey(0)
cv.destroyAllWindows()
