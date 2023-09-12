'''
模板匹配
https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
'''

import cv2 as cv
import numpy as np

img_rgb = cv.imread('mario.png')
assert img_rgb is not None, "file could not be read, check with os.path.exists()"

img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
template = cv.imread('mario_coin.png', cv.IMREAD_GRAYSCALE)
assert template is not None, "file could not be read, check with os.path.exists()"

w, h = template.shape[::-1] # 模板图像的宽高

# 匹配单个结果
# 如果您使用 cv.TM_SQDIFF 作为比较方法，则最小值给出最佳匹配。
# res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
# min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
# bottom_right = (max_loc[0] + w, max_loc[1] + h)
# cv.rectangle(img_rgb, max_loc, bottom_right, (0, 255, 0), 2) # 颜色为BGR，所以红色为(0,0,255)

# 匹配多个结果
res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
threshold = 0.8 # 阈值
loc = np.where( res >= threshold) # 返回的是一个二维数组，第一个数组是行坐标，第二个数组是列坐标
for pt in zip(*loc[::-1]): # *loc[::-1] 表示把loc数组中的元素按照loc[::-1]的顺序传入zip函数
    cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 1)
    
cv.imshow('res.png',img_rgb)
cv.waitKey(0)
cv.destroyAllWindows()