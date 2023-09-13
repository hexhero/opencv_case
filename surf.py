'''
123123
'''
import numpy as np
import cv2 as cv

print(__doc__)

img = cv.imread('mofang.png', cv.IMREAD_GRAYSCALE)

# 创建 SURF 对象。 您可以在此处或稍后指定参数。
# 这里我将Hessian Threshold设置为400
surf = cv.SURF_create(400)
# 直接查找关键点和描述符
kp, des = surf.detectAndCompute(img,None)
print(len(kp))

# 检查当前的 Hessian 阈值
print( surf.getHessianThreshold() )

# 我们将其设置为大约 50000。记住，它只是为了在图片中表示。
# 实际情况下，取值300-500比较好
surf.setHessianThreshold(50000)

# 再次计算关键点并检查其数量。
kp, des = surf.detectAndCompute(img,None)
print(len(kp))

img2 = cv.drawKeypoints(img,kp,None,(255,0,0),4)

cv.imshow('dst',img2)
cv.waitKey(0)
cv.destroyAllWindows()