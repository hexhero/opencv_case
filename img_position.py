import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('messi.png')

plt.imshow(img,'gray')
plt.show()