# OpenCV Case

- [图片混合](addition.py) 
- [图片计算](bitwise.py) 取反，与，或，异或 遮罩
- [图片边框](border.py) 
- [图片通道](channels.py) 
- [绘图](drawing.py) 
- [基础属性](image_basic_attr.py) 尺寸，通道，数据类型
- [剪切](image_cut.py) 
- [定位工具](img_position.py) 
- [鼠标事件](mouse.py)
- [播放视频](play_video.py)
- [滑动按钮](track_bar.py)
- [* 性能优化](https://docs.opencv.org/4.x/dc/d71/tutorial_py_optimization.html)
- [颜色空间](color_space.py) 提取颜色阈值
- [几何变换](transformations.py) 缩放，平移，旋转
- [视角转换](perspective_transformation.py) 透视变换
- [阈值处理](thresholding.py) 二值化
- [图像平滑/模糊](smoothing.py)
- [形态变换](morphological.py) 侵蚀 膨胀 开运算 闭运算
- [图像梯度](gradients.py)  
- [Canny边缘检测](edge_detection_canny.py) 图片边缘，轮廓
- [图像金字塔/分辨率采样](pyramids.py) 多分辨率采样
- [轮廓特征](contour.py) 轮廓面积，周长，近似，凸包，凸缺陷
- [更多轮廓特征-形状匹配](contour_more.py)
- [检测点是否在轮廓中-作业](contour_task.py)
- [直方图绘制&计算](histogram1.ipynb) 统计像素值分布
- [直方图均衡(对比度)](histogram2.ipynb) 校正对比度
- [直方图反投影](histogram_backprojection.py) (含有拼接图片的内容) 它基于图像的颜色直方图，通过将目标对象的颜色分布映射回输入图像，从而实现目标区域的提取。
- [傅里叶变换](fourier_transform.py)
- [模板匹配](template_matching.py)
- [霍夫线检测](houghlines.py)
- [霍夫圆检测](houghcircles.py)
- [分水岭算法分隔图像](watershed.py)
- [GrabCut 交互式前景提取](grabcut.py)
- 特征检测
    - [Harris角点检测](harris_corner_detection.py)
    - [Shi-Tomasi 角点检测](shi_tomasi_corner_detection.py)
    - [SIFT 尺度不变特征变换](sift.py) 当图像比例发生变化时，哈里斯角探测器不够好。Lowe开发了一种突破性的方法来寻找尺度不变的特征，它被称为
    - [SURF 加速稳健特征](surf.py) SIFT真的很好，但不够快，所以人们想出了一个叫做SURF的加速版本。
    - [特征匹配](feature_matching.py)