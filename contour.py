'''
轮廓特征
https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
'''

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img2 = cv.imread('mask.png')
img = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
assert img is not None, "file could not be read, check with os.path.exists()"
ret,thresh = cv.threshold(img,127,255,0)

# 计算轮廓
contours,hierarchy = cv.findContours(thresh, 1, 2)
cnt = contours[0]

# 轮廓属性
M = cv.moments(cnt) # 计算轮廓的矩
print(M)

area = cv.contourArea(cnt) # 计算轮廓的面积
print(area)

perimeter = cv.arcLength(cnt,True) # 计算轮廓的周长
print(perimeter)

# 轮廓特征

# 宽高比
x,y,w,h = cv.boundingRect(cnt)
aspect_ratio = float(w)/h
print(aspect_ratio)

# 范围 轮廓面积与边界矩形面积的比
rect_area = w*h
extent = float(area)/rect_area
print(extent)

# 坚固性 坚固度是等高线面积与其凸包面积的比率。
hull = cv.convexHull(cnt)
hull_area = cv.contourArea(hull)
solidity = float(area)/hull_area
print(solidity)

# 等效直径 是面积与等值线面积相同的圆的直径。
equi_diameter = np.sqrt(4*area/np.pi)
print(equi_diameter)

# 定位 方向是对象定向的角度。以下方法还给出了长轴和短轴长度。
(x,y),(MA,ma),angle = cv.fitEllipse(cnt) 
print(x,y,MA,ma,angle)

# 轮廓近似
epsilon = 0.1*cv.arcLength(cnt,True) # 0.1是精度, 越小越精确，轮廓点越多，越接近原图
approx = cv.approxPolyDP(cnt,epsilon,True) # 近似多边形， True表示封闭
Approximation = cv.drawContours(img2.copy(), [approx], 0, (0,255,0), 3)

# 凸包
hull = cv.convexHull(cnt)
ConvexHull = cv.drawContours(img2.copy(),[hull],0,(0,255,0),2)

# 边界矩形
x,y,w,h = cv.boundingRect(cnt) # x,y是左上角坐标，w,h是宽高
StraightBoundingRectangle = cv.rectangle(img2.copy(),(x,y),(x+w,y+h),(0,255,0),2) # 2 是线宽

# 旋转矩形
rect = cv.minAreaRect(cnt)
box = cv.boxPoints(rect)
box = np.intp(box)
RotatedRectangle = cv.drawContours(img2.copy(),[box],0,(0,255,0),2)

# 最小闭圆
(x,y),radius = cv.minEnclosingCircle(cnt)
center = (int(x),int(y))
radius = int(radius)
MinimumEnclosingCircle = cv.circle(img2.copy(),center,radius,(0,255,0),2)

# 拟合椭圆
ellipse = cv.fitEllipse(cnt)
FittingEllipse = cv.ellipse(img2.copy(),ellipse,(0,255,0),2)

# 拟合直线
rows,cols = img.shape[:2]
vx,vy,x,y = cv.fitLine(cnt, cv.DIST_L2,0,0.01,0.01)
vx,vy,x,y = vx[0],vy[0],x[0],y[0]
lefty = int((-x*vy/vx) + y)
righty = int(((cols-x)*vy/vx)+y)
FittingLine = cv.line(img2.copy(),(cols-1,righty),(0,lefty),(0,255,0),2)

plt.subplot(241),plt.imshow(Approximation),plt.title('Approximation')
plt.xticks([]), plt.yticks([])
plt.subplot(242),plt.imshow(ConvexHull),plt.title('ConvexHull')
plt.xticks([]), plt.yticks([])
plt.subplot(243),plt.imshow(StraightBoundingRectangle),plt.title('StraightBoundingRectangle')
plt.xticks([]), plt.yticks([])
plt.subplot(244),plt.imshow(RotatedRectangle),plt.title('RotatedRectangle')
plt.xticks([]), plt.yticks([])
plt.subplot(245),plt.imshow(MinimumEnclosingCircle),plt.title('MinimumEnclosingCircle')
plt.xticks([]), plt.yticks([])
plt.subplot(246),plt.imshow(FittingEllipse),plt.title('FittingEllipse')
plt.xticks([]), plt.yticks([])
plt.subplot(247),plt.imshow(FittingLine),plt.title('FittingLine')
plt.xticks([]), plt.yticks([])
plt.subplot(248),plt.imshow(img2),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.show()
