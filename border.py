import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('messi.png')
BLUE = [255,0,0]
cv.copyMakeBorder(img, 10,10,10,10,cv.BORDER_REPLICATE)
reflect = cv.copyMakeBorder(img,10,10,10,10,cv.BORDER_REFLECT)
replicate = cv.copyMakeBorder(img,10,10,10,10,cv.BORDER_REPLICATE)
constant= cv.copyMakeBorder(img,10,10,10,10,cv.BORDER_CONSTANT,value=BLUE)

plt.subplot(231),plt.imshow(img,'gray'),plt.title('ORIGINAL')
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')

plt.show()

