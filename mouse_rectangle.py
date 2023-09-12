'''
鼠标框选
'''

import cv2
import numpy as np



class mouse_rectangle:
    # 定义全局变量
    drawing = False  # 是否开始绘制矩形
    ix, iy = -1, -1  # 矩形的起始坐标

    def __init__(self, result) -> None:
        self.result = result
        pass
    
    # 鼠标回调函数
    def draw_rectangle(self, event, x, y, flags, param):
        global ix, iy, drawing

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            self.result((ix, iy), (x, y))


if __name__ == '__main__':
    # 创建一个黑色图像窗口
    mr = mouse_rectangle(result = lambda a, b:  cv2.rectangle(img, a, b, (0, 255, 0), 2))
    img = np.zeros((512, 512, 3), np.uint8)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mr.draw_rectangle)

    while True:
        cv2.imshow('image', img)
        #
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
