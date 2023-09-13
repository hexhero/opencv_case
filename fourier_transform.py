'''
傅里叶变换在图像处理中有以下几个常见的应用：

频域滤波：通过将图像转换到频域，可以使用滤波器来增强或抑制特定频率的信息。例如，可以使用低通滤波器来平滑图像并去除高频噪声，或者使用高通滤波器来增强图像的边缘信息。

图像去噪：通过在频域中滤除高频噪声，可以减少图像中的噪声。这可以通过将图像转换到频域，将高频噪声设置为零，然后将图像转换回时域来实现。

特征提取：傅里叶变换可以帮助提取图像中的频率特征，如纹理、周期性模式等。通过分析频域中的幅度谱和相位谱，可以得到关于图像中不同频率分量的信息。

图像压缩：傅里叶变换在图像压缩中起着重要的作用。通过将图像转换到频域，可以利用频域的特性来减少图像的数据量，从而实现图像的压缩。
'''

# import cv2 as cv
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv.imread('messi.png', cv.IMREAD_GRAYSCALE)
# assert img is not None, "file could not be read, check with os.path.exists()"
# f = np.fft.fft2(img)
# fshift = np.fft.fftshift(f)
# magnitude_spectrum = 20*np.log(np.abs(fshift))

# rows, cols = img.shape
# crow,ccol = rows//2 , cols//2
# fshift[crow-30:crow+31, ccol-30:ccol+31] = 0
# f_ishift = np.fft.ifftshift(fshift)
# img_back = np.fft.ifft2(f_ishift)
# img_back = np.real(img_back)

# plt.subplot(131),plt.imshow(img, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(132),plt.imshow(magnitude_spectrum, cmap = 'gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.subplot(133),plt.imshow(img_back)
# plt.title('Magnitude Spectrum2'), plt.xticks([]), plt.yticks([])
# plt.show()

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('messi.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

dft = cv.dft(np.float32(img),flags = cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(cv.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

rows, cols = img.shape
crow,ccol = rows/2 , cols/2
# create a mask first, center square is 1, remaining all zeros
mask = np.zeros((rows,cols,2),np.uint8)
mask[int(crow)-30:int(crow)+30, int(ccol)-30:int(ccol)+30] = 1
# apply mask and inverse DFT
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv.idft(f_ishift)
img_back = cv.magnitude(img_back[:,:,0],img_back[:,:,1])

plt.subplot(131),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(img_back, cmap = 'gray')
plt.title('Magnitude Spectrum2'), plt.xticks([]), plt.yticks([])
plt.show()