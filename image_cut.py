import numpy as np
import cv2 as cv


img = cv.imread('messi.png')
cv.namedWindow('img')

ball = img[550:650, 450:550].copy()  # 从图像中提取一个区域 前面一组是行 后面一组是列
head = img[10:110, 250:350].copy()

img[10:110, 250:350] = ball
img[550:650, 450:550] = head

print(ball.shape, head.shape)

def get_positon(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print(x, y)

cv.setMouseCallback('img', get_positon)

cv.imshow('img', img)
cv.waitKey(0)
