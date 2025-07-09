import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
# from scipy.ndimage import label, generate_binary_structure # 实际上这里用不到 label
from skimage.feature import peak_local_max # 确保你已安装 pip install scikit-image

def watershed_solution_two_local_maxima_fixed():
    """
    方案二 (修正版)：利用距离变换的局部最大值作为前景标记来分离相交物体。
    核心是确保 'skimage' 库已正确安装。
    """
    print("--- 正在运行方案二 (修正版)：局部最大值作为标记 ---")

    # --- 1. 图像加载与预处理 ---
    image_path = os.path.join('bina.png')
    img_original_gray = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    if img_original_gray is None:
        print(f"错误：无法加载图像。请检查路径是否正确：{image_path}")
        return

    img_display_color = cv.cvtColor(img_original_gray, cv.COLOR_GRAY2BGR)
    img_binary = img_original_gray # 假设白色前景，黑色背景

    # --- 2. 分水岭算法核心预处理步骤 ---

    # 2.1. 噪声去除 / 平滑对象 (形态学开运算)
    kernel = np.ones((3,3), np.uint8)
    opening = cv.morphologyEx(img_binary, cv.MORPH_OPEN, kernel, iterations = 2)

    # 2.2. 确定背景区域 (Sure Background)
    sure_bg = cv.dilate(opening, kernel, iterations=3)

    # 2.3. 距离变换
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)

    # **2.4. 寻找局部最大值作为前景标记 (关键)**
    # 将距离变换结果归一化到 0-1 范围，方便设置阈值
    dist_transform_norm = cv.normalize(dist_transform, None, 0, 1.0, cv.NORM_MINMAX)

    # 找到局部最大值的坐标。这两个参数需要根据你的圆的大小和相交情况来调整。
    # min_distance: 局部最大值点之间的最小距离。如果太小，一个圆可能被分成多个标记。
    #               如果太大，两个相邻的圆可能只找到一个标记。
    # threshold_abs: 绝对阈值，只有大于此值的点才会被认为是局部最大值。
    
    # 根据你提供的图像，圆的直径可能在200-300像素左右。
    # 那么 min_distance 应该小于其直径，但足以分隔两个圆心。
    # 尝试 min_distance = 100 到 150 之间。
    # threshold_abs 保持 0.3-0.5 之间。
    coordinates = peak_local_max(dist_transform_norm, 
                                 min_distance=100, # 调整这个值！
                                 threshold_abs=0.4) # 调整这个值！

    # 创建 sure_fg 标记图像
    sure_fg = np.zeros(dist_transform.shape, dtype=np.uint8)
    # 将找到的局部最大值点设置为 255 (白色)
    for coords in coordinates:
        sure_fg[coords[0], coords[1]] = 255

    # 如果需要，对 sure_fg 进行小范围的膨胀，使标记更明显（可选）
    sure_fg = cv.dilate(sure_fg, np.ones((3,3), np.uint8), iterations=1)

    # 2.5. 识别未知区域 (Unknown Region)
    unknown = cv.subtract(sure_bg, sure_fg)

    # 2.6. 生成标记 (Marker Labelling)
    _, markers = cv.connectedComponents(sure_fg)

    num_foreground_markers = np.max(markers)
    print(f"方案二 (修正版)：检测到的前景标记数量 (Sure Foreground): {num_foreground_markers}")
    if num_foreground_markers <= 1:
        print("方案二 (修正版) 警告：前景标记数量不足，相交的圆可能仍未分开。")
        print("请尝试：a) 调整 peak_local_max 的 'min_distance' 或 'threshold_abs' 参数。")
        print("         b) 检查 'bina.png' 是否包含期望的对象。")

    markers = markers + 1
    markers[unknown == 255] = 0

    # --- 3. 可视化中间处理结果 ---
    plt.figure(figsize=(20, 12)) 
    
    plt.subplot(251)
    plt.imshow(img_original_gray, cmap='gray')
    plt.title('1. Original bina.png (Grayscale)')
    plt.axis('off')

    plt.subplot(252)
    plt.imshow(img_binary, cmap='gray')
    plt.title('2. Initial Binary Image')
    plt.axis('off')

    plt.subplot(253)
    plt.imshow(opening, cmap='gray')
    plt.title('3. Morphological Opening (Noise Removed)')
    plt.axis('off')

    plt.subplot(254)
    plt.imshow(sure_bg, cmap='gray')
    plt.title('4. Sure Background Area (Dilated)')
    plt.axis('off')
    
    plt.subplot(255)
    plt.imshow(dist_transform, cmap='jet')
    plt.title('5. Distance Transform')
    plt.axis('off')

    plt.subplot(256)
    plt.imshow(sure_fg, cmap='gray')
    plt.title('6. Sure Foreground (Markers) - Local Maxima') # 观察这里是否分开
    plt.axis('off')

    plt.subplot(257)
    plt.imshow(unknown, cmap='gray')
    plt.title('7. Unknown Region (Boundary)')
    plt.axis('off')

    plt.subplot(258)
    plt.imshow(markers, cmap='nipy_spectral')
    plt.title('8. Final Markers for Watershed')
    plt.axis('off')
    plt.tight_layout()

    # --- 4. 执行分水岭算法 ---
    markers = np.int32(markers)
    labels_watershed = cv.watershed(img_display_color.copy(), markers)

    # --- 5. 可视化最终分割结果 ---
    plt.figure(figsize=(14, 7))
    
    plt.subplot(121)
    plt.imshow(labels_watershed, cmap='nipy_spectral')
    plt.title('Watershed Segmentation Result (Labels)')
    plt.axis('off')

    plt.subplot(122)
    img_result_with_lines = img_display_color.copy()
    img_result_with_lines[labels_watershed == -1] = [255, 0, 0]
    plt.imshow(img_result_with_lines)
    plt.title('Original Image with Watershed Lines')
    plt.axis('off')
    plt.tight_layout()

    plt.show()