import cv2
import numpy as np

# 定义 RGB 色值
r, g, b = 0, 255, 0

# 将 RGB 色值转换为 HSV 色值
rgb = np.uint8([[[b, g, r]]])
hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)

# 提取 HSV 色值
h, s, v = hsv[0][0]

print(f"RGB: ({r}, {g}, {b})")
print(f"HSV: ({h}, {s}, {v})")