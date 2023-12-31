import cv2 as cv
import numpy as np

img = np.zeros((512, 512, 3), np.uint8)
cv.line(img, (0, 0), (511, 511), (255, 0, 0), 5)

cv.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)

cv.circle(img, (447, 63), 63, (0, 0, 255), -1)

pts = np.array([[50, 0], [25, 100], [100, 25], [
               0, 25], [75, 100], [50, 0]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv.polylines(img, [pts], True, (0, 255, 255))

font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img, 'OpenCV', (10, 500), font, 4, (255, 255, 255), 2, cv.LINE_AA)

cv.imshow('drawing', img)
cv.waitKey(0)

