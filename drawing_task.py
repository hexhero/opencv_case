import numpy as np
import cv2 as cv

def nothing(x):
 pass

img = np.zeros((300,512,3), np.uint8)
cv.namedWindow('image')

cv.createTrackbar('R','image',0,255,nothing)
cv.createTrackbar('G','image',0,255,nothing)
cv.createTrackbar('B','image',0,255,nothing)
cv.createTrackbar('Radius','image',0,100,nothing)

def draw(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        r = cv.getTrackbarPos('R','image')
        g = cv.getTrackbarPos('G','image')
        b = cv.getTrackbarPos('B','image')
        radius = cv.getTrackbarPos('Radius','image')
        cv.circle(img, (x, y), radius, (b, g, r), -1)
        
cv.setMouseCallback('image', draw)

while (1):
    cv.imshow('image', img)
    k = cv.waitKey(1)
    if k == 27:
        break

cv.destroyAllWindows()
        
    