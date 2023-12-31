import cv2 as cv
import numpy as np
import time

'''
颜色空间转换
'''

# fiags = [i for i in dir(cv) if i.startswith('COLOR_')]
# print(fiags)

cap = cv.VideoCapture('http://192.168.31.197:8080/video')

lower_yellow = np.array([0, 50, 50])
upper_yellow = np.array([3, 255, 255])


while 1:
    # time.sleep(0.1)
    _, frame = cap.read()
    if frame is None:
        break
    frame = cv.resize(frame, None, fx=0.5, fy=0.5 ,interpolation=cv.INTER_CUBIC)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_yellow, upper_yellow)
    
    # 消除噪声
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel) # 开运算: 先侵蚀再膨胀, 消除背景中的噪声
    mask = cv.dilate(mask, kernel, iterations=1) # 膨胀
    
    res = cv.bitwise_and(frame, frame, mask=mask) # 与运算
    
    # 计算mask轮廓
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x,y,w,h = cv.boundingRect(cnt)
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        print(cnt)
    
    cv.imshow('frame',frame)
    cv.imshow('mask',mask)
    cv.imshow('res',res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()