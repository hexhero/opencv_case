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
    _, frame = cap.read()
    if frame is None:
        break
    frame = cv.resize(frame, None, fx=0.5, fy=0.5 ,interpolation=cv.INTER_CUBIC)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    mask = cv.inRange(hsv, lower_yellow, upper_yellow)
    res = cv.bitwise_and(frame, frame, mask=mask)
    
    cv.imshow('frame',frame)
    cv.imshow('mask',mask)
    cv.imshow('res',res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()