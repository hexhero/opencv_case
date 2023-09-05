import cv2 as cv
import numpy as np
import time

'''
颜色空间转换
'''

# fiags = [i for i in dir(cv) if i.startswith('COLOR_')]
# print(fiags)

cap = cv.VideoCapture('ssstwitter.com_1693888084809.mp4')

lower_yellow = np.array([16, 160, 160])
upper_yellow = np.array([18, 250, 250])


while 1:
    _, frame = cap.read()
    if frame is None:
        break
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