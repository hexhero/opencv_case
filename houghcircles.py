'''
霍夫圆检测
https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html
'''
import numpy as np
import cv2 as cv
img = cv.imread('zhuoqiu.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

img = cv.medianBlur(img,5) # 中值滤波
cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)

'''
霍夫圆检测
参数解释如下：
    img：输入的灰度图像。
    cv.HOUGH_GRADIENT：表示使用 Hough 变换的一种方法来检测圆形。
    1：表示圆心之间的距离的分辨率，如果设为 1，表示使用与输入图像相同的分辨率。
    20：表示圆心之间的最小距离。如果检测到的两个圆的圆心之间的距离小于此值，则只保留其中一个圆。
    param1=50：是用于边缘检测的 Canny 算法的高阈值。
    param2=30：是用于确定圆心的累加器阈值。较小的值将导致更多的检测到的圆，但是可能会有错误的圆。
    minRadius=0：要检测的圆的最小半径。
    maxRadius=0：要检测的圆的最大半径。
'''
circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,20, param1=50,param2=50,minRadius=0,maxRadius=0)
circles = np.uint16(np.around(circles))

for i in circles[0,:]:
    # draw the outer circle
    cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    
cv.imshow('detected circles',cimg)
cv.waitKey(0)
cv.destroyAllWindows()

