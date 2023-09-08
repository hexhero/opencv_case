import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('messi.png')
BLUE = [255,0,0]
reflect = cv.copyMakeBorder(img,30,30,30,30,cv.BORDER_REFLECT) # 反射法
replicate = cv.copyMakeBorder(img,30,30,30,30,cv.BORDER_REPLICATE) # 复制法
constant= cv.copyMakeBorder(img,30,30,30,30,cv.BORDER_CONSTANT,value=BLUE) # 常量法

plt.subplot(221),plt.imshow(img,'gray'),plt.title('ORIGINAL')
plt.subplot(222),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
plt.subplot(223),plt.imshow(constant,'gray'),plt.title('CONSTANT')
plt.subplot(224),plt.imshow(reflect,'gray'),plt.title('REFLECT')
plt.show()

