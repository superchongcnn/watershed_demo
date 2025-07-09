import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os

def watershed():
    root = os.getcwd() # 获取当前工作目录
    
    # 构建图像文件的完整路径
    # 确保 'ct.jpg' 文件在脚本运行的同一目录下，
    # 或者如果它在子文件夹中，比如 'demoImages'，则路径应为 os.path.join('demoImages', 'ct.jpg')
    imgPath = os.path.join('tute2.jpg') 
    
    # 1. **首先读取图像**
    img_original = cv.imread(imgPath) 
    
    # **关键检查：确保图片成功加载**
    if img_original is None:
        print(f"错误：无法加载图像。请检查路径是否正确：{imgPath}")
        print("请确认 'ct.jpg' 文件是否存在于脚本运行的目录中。")
        return # 如果图片加载失败，则退出函数，避免后续错误

    # 2. **然后对读取到的图像进行裁剪**
    # 注意：新的裁剪范围是 [10:700, 10:900]，请确保这个范围适合你的CT图像尺寸
  #  img_cropped = img_original[1300:2100, 10:900] 
    img_cropped = img_original[610:1300, 10:900] 
    
    # 检查裁剪后的图像是否有效（避免裁剪出空图像）
    if img_cropped.shape[0] == 0 or img_cropped.shape[1] == 0:
        print("错误：裁剪后的图像为空。请检查裁剪范围是否超出了原始图像尺寸。")
        print(f"原始图像尺寸：{img_original.shape}")
        return

    # 将裁剪后的图像从BGR颜色空间转换为RGB颜色空间，用于Matplotlib显示
    imgRGB = cv.cvtColor(img_cropped, cv.COLOR_BGR2RGB)
    # 将裁剪后的图像转换为灰度图像，用于后续的分水岭处理
    img_gray = cv.cvtColor(img_cropped, cv.COLOR_BGR2GRAY)

    # 创建第一个Matplotlib图形窗口，用于展示中间处理结果
    plt.figure(figsize=(15, 10)) # 调整图表大小以便更好地显示所有子图
    
    plt.subplot(231)
    plt.imshow(img_gray, cmap='gray') # 显示裁剪后的灰度图像
    plt.title('1. Cropped Gray Image')

    plt.subplot(232)
    # 对灰度图像进行反向二值化处理 (THRESH_BINARY_INV)
    # 阈值120可能需要根据CT图像的亮度/对比度进行调整，以更好地分离肺部区域
    _, imgThreshold = cv.threshold(img_gray, 120, 255, cv.THRESH_BINARY_INV)
    plt.imshow(imgThreshold, cmap='gray') # 显示二值化后的图像
    plt.title('2. Binary Threshold (INV)')

    plt.subplot(233)
    kernel = np.ones((3,3), np.uint8) # 定义一个3x3的核
    # 对二值化图像进行膨胀操作，连接前景区域或填充小孔
    imgDilate = cv.morphologyEx(imgThreshold, cv.MORPH_DILATE, kernel)
    plt.imshow(imgDilate, cmap='gray') # 显示膨胀后的图像
    plt.title('3. Dilated Image')

    plt.subplot(234)
    # 计算距离变换，用于找到前景物体的中心区域
    distTrans = cv.distanceTransform(imgDilate, cv.DIST_L2, 5)
    plt.imshow(distTrans, cmap='jet') # 用jet颜色映射显示距离变换结果
    plt.title('4. Distance Transform')

    plt.subplot(235)
    # 对距离变换结果进行二值化，提取前景标记
    # 阈值15可能需要根据具体图像和预期分割对象的大小进行调整
    _, distThresh = cv.threshold(distTrans, 15, 255, cv.THRESH_BINARY)
    plt.imshow(distThresh, cmap='gray') # 显示前景标记
    plt.title('5. Distance Threshold (Markers)')

    plt.subplot(236)
    # 将标记图像转换为uint8类型
    distThresh_uint8 = np.uint8(distThresh) # 使用新的变量名，避免混淆
    # 查找连通组件，为每个独立区域分配一个唯一的标签
    _, labels = cv.connectedComponents(distThresh_uint8)
    plt.imshow(labels, cmap='nipy_spectral') # 显示连通组件标签
    plt.title('6. Connected Components (Pre-Watershed Markers)')
    plt.tight_layout() # 调整子图布局，避免重叠
    
    # 创建第二个Matplotlib图形窗口，用于展示分水岭算法的最终结果
    plt.figure(figsize=(12, 6))
    
    plt.subplot(121)
    # 执行分水岭算法
    # 注意：cv.watershed 需要一个3通道的彩色图像作为第一个参数（imgRGB）
    # 和一个标记图像作为第二个参数（labels）
    # 分水岭算法会修改传入的labels数组，将其分割为不同的区域和边界
    labels_watershed = cv.watershed(imgRGB, labels.copy()) # 使用labels的副本，避免修改原始labels
    plt.imshow(labels_watershed, cmap='nipy_spectral')
    plt.title('Watershed Segmentation Result')

    plt.subplot(122)
    # 将分水岭线标记为红色
    # 分水岭算法会将分水岭线标记为-1
    imgRGB_with_lines = imgRGB.copy() # 创建imgRGB的副本，避免修改原始图像
    imgRGB_with_lines[labels_watershed == -1] = [255, 0, 0] # 将分水岭线像素设置为红色
    plt.imshow(imgRGB_with_lines)
    plt.title('Original Image with Watershed Lines')
    plt.tight_layout()

    plt.show() # 显示所有图形窗口

if __name__ == '__main__':
    watershed()