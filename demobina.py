import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os

def watershed_segmentation_for_bina_png():
    """
    对本地图像 'bina.png' 执行分水岭图像分割，
    并可视化每个关键步骤，特别旨在分离相交的物体。
    """
    
    # --- 1. 图像加载与预处理 ---
    
    # 构建图像文件的完整路径
    # 确保 'bina.png' 文件与脚本在同一目录下
    image_path = os.path.join('bina.png') 
    
    # 以灰度模式读取图像
    # 对于通常为二值图的 'bina.png'，以灰度读取是合适的
    img_original_gray = cv.imread(image_path, cv.IMREAD_GRAYSCALE) 
    
    # 关键检查：确保图片成功加载
    if img_original_gray is None:
        print(f"错误：无法加载图像。请检查路径是否正确：{image_path}")
        print("请确认 'bina.png' 文件是否存在于脚本运行的目录中。")
        print("重要提示：分水岭算法需要图像中包含明确的前景对象和背景才能进行有意义的分割。")
        print("如果 'bina.png' 是纯色图像（例如，完全灰色），将无法分割。")
        return

    # 为了分水岭算法的最终显示（需要3通道图像来绘制彩色分割线），
    # 将原始灰度图像转换为BGR（OpenCV的默认颜色顺序）
    img_display_color = cv.cvtColor(img_original_gray, cv.COLOR_GRAY2BGR)

    # 假设 'bina.png' 已经是二值图像（只有黑白两色），
    # 并且白色代表前景，黑色代表背景。
    # 如果你的 'bina.png' 是黑色前景，白色背景，请取消注释下一行：
    # img_binary = cv.bitwise_not(img_original_gray)
    img_binary = img_original_gray 
    
    # --- 2. 分水岭算法核心预处理步骤 ---
    # 这些步骤旨在从二值图像中提取出用于指导分水岭算法的“标记”。

    # 2.1. 噪声去除 / 平滑对象 (形态学开运算)
    # 创建一个3x3的核，用于形态学操作
    kernel = np.ones((3,3), np.uint8)
    # 开运算 (Open): 先腐蚀后膨胀。
    # 目的：去除小的白色噪声点，并平滑物体边界。
    # iterations=2 可以执行两次开运算，增强效果。
    opening = cv.morphologyEx(img_binary, cv.MORPH_OPEN, kernel, iterations = 2)

    # 2.2. 确定背景区域 (Sure Background)
    # 膨胀操作：将前景区域（白色部分）膨胀。
    # 目的：创建一片我们“确定”是背景的区域。
    # 迭代次数越多，确定的背景区域越大。
    sure_bg = cv.dilate(opening, kernel, iterations=3)

    # 2.3. 确定前景区域 (Sure Foreground)
    # 距离变换 (Distance Transform): 计算每个前景像素到最近的背景像素的距离。
    # 像素值越大，表示该像素离背景越远，因此越可能是前景物体的中心。
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5) # L2距离，使用5x5掩码

    # 阈值化距离变换结果，获取确定前景区域。
    # 这是分离相交物体的关键步骤。
    # 提高阈值 (例如 0.85 或 0.9) 可以确保只有物体最核心的区域被标记为前景，
    # 从而在相交处分开不同的物体。
    if dist_transform.max() == 0:
        print("警告：距离变换结果全为0，可能图像中没有足够大的前景对象。")
        print("请检查 'bina.png' 是否包含期望分割的对象，或调整预处理参数。")
        sure_fg = np.zeros_like(opening, dtype=np.uint8) # 创建一个全0的sure_fg
    else:
        # 尝试一个较高的阈值，例如 0.85 或 0.9
        # 如果相交圆仍未分开，请尝试增加此值，例如 0.9 或 0.95
        threshold_ratio = 0.99 
        _, sure_fg = cv.threshold(dist_transform, threshold_ratio * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg) # 转换为uint8类型

    # 2.4. 识别未知区域 (Unknown Region)
    # 未知区域 = 确定背景 - 确定前景
    # 这些是分水岭算法需要“决定”边界的区域。
    unknown = cv.subtract(sure_bg, sure_fg)

    # 2.5. 生成标记 (Marker Labelling)
    # 对确定前景区域进行连通组件分析，为每个独立的前景对象分配一个唯一的标签。
    # 'markers' 数组将包含每个像素所属的连通组件的ID。
    _, markers = cv.connectedComponents(sure_fg)

    # 调试信息：打印标记数量
    num_foreground_markers = np.max(markers)
    print(f"检测到的前景标记数量 (Sure Foreground): {num_foreground_markers}")
    if num_foreground_markers <= 1:
        print("警告：前景标记数量不足，可能无法将相交的圆分开。")
        print("请尝试：")
        print("  a) 调整距离变换阈值 (例如将 'threshold_ratio' 提高到 0.9 或更高)。")
        print("  b) 确保 'bina.png' 中的对象没有完全融合，存在可分离的“颈部”或“谷地”。")

    # 分水岭算法约定：
    # - 未知区域 (unknown) 的像素标记为 0。
    # - 确定背景 (sure_bg) 标记为 1。
    # - 确定前景 (sure_fg) 标记从 2 开始（markers + 1）。
    markers = markers + 1 # 将所有前景标记 ID 加 1，避免与背景 0/1 冲突
    markers[unknown == 255] = 0 # 将未知区域的像素标记为 0

    # --- 3. 可视化中间处理结果 ---
    plt.figure(figsize=(18, 12)) # 设置图表大小
    
    plt.subplot(241)
    plt.imshow(img_original_gray, cmap='gray')
    plt.title('1. Original bina.png (Grayscale)')
    plt.axis('off') # 关闭坐标轴

    plt.subplot(242)
    plt.imshow(img_binary, cmap='gray')
    plt.title('2. Initial Binary Image')
    plt.axis('off')

    plt.subplot(243)
    plt.imshow(opening, cmap='gray')
    plt.title('3. Morphological Opening (Noise Removed)')
    plt.axis('off')

    plt.subplot(244)
    plt.imshow(sure_bg, cmap='gray')
    plt.title('4. Sure Background Area (Dilated)')
    plt.axis('off')

    plt.subplot(245)
    plt.imshow(dist_transform, cmap='jet') # 使用 'jet' 颜色映射显示距离梯度
    plt.title('5. Distance Transform')
    plt.axis('off')

    plt.subplot(246)
    plt.imshow(sure_fg, cmap='gray')
    plt.title('6. Sure Foreground (Markers) - Crucial for separation')
    plt.axis('off')

    plt.subplot(247)
    plt.imshow(unknown, cmap='gray')
    plt.title('7. Unknown Region (Boundary)')
    plt.axis('off')

    plt.subplot(248)
    plt.imshow(markers, cmap='nipy_spectral') # 使用 'nipy_spectral' 颜色映射显示不同的标记ID
    plt.title('8. Final Markers for Watershed (Input)')
    plt.axis('off')
    plt.tight_layout() # 自动调整子图布局，避免重叠
    
    # --- 4. 执行分水岭算法 ---
    # 分水岭算法接受一个3通道的彩色图像和标记图。
    # 它会修改传入的 'markers' 数组，并返回修改后的数组，其中包含了最终的分割标签。
    # 需要将 markers 转换为 np.int32 类型
    markers = np.int32(markers) 
    
    # 执行分水岭
    # 注意：cv.watershed 会修改 markers 数组，我们用其返回值作为最终标签
    labels_watershed = cv.watershed(img_display_color.copy(), markers)

    # --- 5. 可视化最终分割结果 ---
    plt.figure(figsize=(14, 7))
    
    plt.subplot(121)
    # 分水岭结果：每个分割区域由一个唯一的标签表示。
    # 使用 'nipy_spectral' 颜色映射来区分不同的标签（区域）。
    plt.imshow(labels_watershed, cmap='nipy_spectral') 
    plt.title('Watershed Segmentation Result (Labels)')
    plt.axis('off')

    plt.subplot(122)
    # 将分水岭线叠加在原始彩色图像上。
    # 分水岭算法会将分割边界（分水岭线）标记为 -1。
    img_result_with_lines = img_display_color.copy() 
    img_result_with_lines[labels_watershed == -1] = [255, 0, 0] # 将边界像素设置为红色
    plt.imshow(img_result_with_lines)
    plt.title('Original Image with Watershed Lines')
    plt.axis('off')
    plt.tight_layout()

    plt.show() # 显示所有图形窗口

if __name__ == '__main__':
    watershed_segmentation_for_bina_png()