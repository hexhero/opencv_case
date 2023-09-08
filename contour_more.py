'''
轮廓更多特征
https://docs.opencv.org/4.x/d5/d45/tutorial_py_contours_more_functions.html
'''

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img2 = cv.imread('star1.png')
img = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
assert img is not None, "file could not be read, check with os.path.exists()"
ret,thresh = cv.threshold(img,127,255,0)

# 计算轮廓
contours,hierarchy = cv.findContours(thresh, 1, 2)
cnt = contours[0]

# 凸性缺陷
hull = cv.convexHull(cnt,returnPoints = False)
defects = cv.convexityDefects(cnt,hull)

for i in range(defects.shape[0]):
    s,e,f,d = defects[i,0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    cv.line(img2,start,end,[0,255,0],2)
    cv.circle(img2,far,5,[0,0,255],-1)

# 测试点是否在轮廓内 
# 检测点(x,y)到轮廓的距离，如果点在轮廓外部，返回负值，如果在轮廓内部，返回正值，如果在轮廓上，返回零。
x,y = 50,50
dist = cv.pointPolygonTest(cnt,(x,y),True)
print('检测点是否在轮廓内', dist)
cv.circle(img2,(x,y),5,[255,255,0],-1) # -1 表示填充

# 匹配形状
star22 = cv.imread('star2.png')
star2 =  cv.cvtColor(star22, cv.COLOR_BGR2GRAY)
ret, thresh2 = cv.threshold(star2, 127, 255, 0)
contours2, hierarchy2 = cv.findContours(thresh2, 1, 2)
for cnt2 in contours2:  
    ret = cv.matchShapes(cnt,cnt2,1,0.0) # 值越小表示形状越相似。
    (x,y),(MA,ma),angle = cv.fitEllipse(cnt2) # 定位 x,y是中心点坐标 MA,ma是长短轴长度 angle是旋转角度
    cv.putText(star22, str(round(ret, 4)), (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    print('相似度', ret)
    if(ret < 0.1):
        rect = cv.minAreaRect(cnt2)
        box = cv.boxPoints(rect)
        box = np.intp(box)
        star22 = cv.drawContours(star22,[box],0,(0,255,0),2)

cv.imshow('star1',img2)
cv.imshow('star2', star22)
cv.waitKey(0)
cv.destroyAllWindows()
