import cv2 as cv
import numpy as np

img = cv.imread('messi.png')
cv.namedWindow('img')

#分离和合并通道
# b,g,r = cv.split(img) 
# img = cv.merge((b,g,r))

#Or
# b = img[:,:,0]

#设置所有红色通道的值为0


def event_r(x):
    img[:,:,2] = x

def event_g(x):
    img[:,:,1] = x
    
def event_b(x):
    img[:,:,0] = x

cv.createTrackbar('R','img',0,255,event_r)
cv.createTrackbar('G','img',0,255,event_g)
cv.createTrackbar('B','img',0,255,event_b)

while (1):
    cv.imshow('img', img)
    k = cv.waitKey(1)
    if k == 27:
        break

