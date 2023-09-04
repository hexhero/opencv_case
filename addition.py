import cv2 as cv
import numpy as np

'''
图片加法
'''
cv.namedWindow('image')

messi = cv.imread('messi.png')
leo = cv.imread('leo.png')

leo_shape = leo.shape

x,y = 8,210
a,b,c,d = x,x+leo_shape[0],y,y+leo_shape[1]

print(leo_shape)

dd = messi[a:b, c:d].copy()

# dst = cv.addWeighted(dd, 0.3, leo, 0.7, 0) # 0.7, 0.3 are the weights of the images . 0 is the gamma value
# messi[a:b, c:d] = dst

cv.createTrackbar('messi','image',1,10,lambda x: None)
cv.createTrackbar('leo','image',1,10,lambda x: None)

while 1:
    cv.imshow('image', messi)
    key = cv.waitKey(1)
    if key == 27:
        break
    h = cv.getTrackbarPos('messi','image')
    j = cv.getTrackbarPos('leo','image')
    dst = cv.addWeighted(dd, h*0.1, leo, j*0.1, 0) # 0.7, 0.3 are the weights of the images . 0 is the gamma value
    messi[a:b, c:d] = dst

cv.destroyAllWindows()