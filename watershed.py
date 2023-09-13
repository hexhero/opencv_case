'''
分水岭算法分隔图像
https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html

Watershed 是一种图像分割算法，用于将图像中的不同区域分离开来。它基于图像中的灰度或颜色信息，将图像视为地形图，将图像中的每个像素点都视为地形上的一个点。这个算法的主要思想是将图像看作是一个梯度的流域，其中梯度较大的区域被认为是物体的边界，而梯度较小的区域则被认为是物体的内部。

Watershed 算法的主要步骤如下：

预处理：首先对图像进行预处理，例如去噪、平滑化等操作，以便更好地提取图像的梯度信息。

计算梯度：使用梯度计算方法（如Sobel算子）计算图像的梯度，得到梯度图像。

寻找种子点：根据梯度图像寻找种子点，这些种子点将作为分割的起始点。

标记种子点：将种子点标记为不同的区域。

执行分水岭算法：从种子点开始，通过模拟水流从高处流向低处的过程，逐步扩展区域并分割图像。

后处理：根据需要对分割结果进行后处理，例如去除噪声、合并相似区域等操作。

Watershed 算法在图像分割领域具有广泛的应用，特别是在处理具有复杂边界和重叠区域的图像时表现出色。它可以用于目标检测、图像分析、医学图像处理等领域。
'''

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('water_coins.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)

# sure background area
sure_bg = cv.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv.watershed(img,markers)
img[markers == -1] = [255,0,0]

plt.subplot(331), plt.imshow(img, 'gray'), plt.title('original')
plt.subplot(332),plt.imshow(gray, 'gray'), plt.title('gray')
plt.subplot(333),plt.imshow(thresh, 'gray'), plt.title('thresh')
plt.subplot(334),plt.imshow(opening, 'gray'), plt.title('opening')
plt.subplot(335),plt.imshow(sure_bg, 'gray'), plt.title('sure_bg')
plt.subplot(336),plt.imshow(dist_transform, 'gray'), plt.title('dist_transform')
plt.subplot(337),plt.imshow(sure_fg, 'gray'), plt.title('sure_fg')
plt.subplot(338),plt.imshow(unknown, 'gray'), plt.title('unknown')
plt.subplot(339),plt.imshow(markers, 'gray'), plt.title('markers')
plt.show()