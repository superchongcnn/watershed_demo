import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.feature import peak_local_max

def watershed_segment_two_parts():
    """
    优化分水岭分割重叠圆，旨在将前景分为两部分。
    核心：通过精细调整 peak_local_max 参数来生成两个独立的“前景标记”。
    """
    print("--- 正在运行分水岭分割两部分前景 ---")

    # 假设 'bina.png' 存在于脚本相同目录下
    image_path = os.path.join('tute2.jpg')
    img_original_gray1 = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    img_original_gray = img_original_gray1[10:700, 10:900] 

    if img_original_gray is None:
        print(f"错误：无法加载图像。请检查路径是否正确：{image_path}")
        print("请确保 'bina.png' 文件确实存在并可读。")
        return

    # 将灰度图转换为BGR，用于分水岭函数的彩色输入
    img_display_color = cv.cvtColor(img_original_gray, cv.COLOR_GRAY2BGR)
    # 假设输入bina.png已经是二值图（白色前景，黑色背景）
    img_binary = img_original_gray 

    # --- 1. 形态学开运算 (去除噪声，平滑边界) ---
    kernel_open = np.ones((3,3), np.uint8)
    opening = cv.morphologyEx(img_binary, cv.MORPH_OPEN, kernel_open, iterations = 2)

    # --- 2. (关键) 轻微腐蚀以帮助分离重叠区域 ---
    # 调整这里！使用一个有效的小核进行腐蚀，以在距离变换图上创建更清晰的“山谷”。
    # 目的：让两个圆的连接处变得更细，甚至断开，从而帮助分水岭生成直线。
    kernel_pre_erode = np.ones((5,5), np.uint8) # <<<<<<<<<< 尝试 (3,3), (5,5), (7,7)
    iterations_pre_erode = 1 # <<<<<<<<<< 尝试 1 或 2 次迭代
    pre_eroded_for_dt = cv.erode(opening, kernel_pre_erode, iterations=iterations_pre_erode)
    
    # 确保腐蚀后图像不为空，否则后续步骤会出错
    if np.max(pre_eroded_for_dt) == 0:
        print("致命错误：预腐蚀操作过度，所有前景物体在图像中消失。")
        print("请尝试减小 'kernel_pre_erode' 大小或 'iterations_pre_erode' 次数。")
        plt.figure(figsize=(6,6))
        plt.imshow(pre_eroded_for_dt, cmap='gray')
        plt.title('Error: Pre-Eroded image is empty!')
        plt.axis('off')
        plt.show() 
        return
        
    # --- 3. 距离变换 ---
    # 对预腐蚀后的图像进行距离变换，圆心处的值最高
    dist_transform = cv.distanceTransform(pre_eroded_for_dt, cv.DIST_L2, 5) 
    
    # 将距离变换结果归一化到 0-1 范围，方便设置 threshold_abs
    dist_transform_norm = cv.normalize(dist_transform, None, 0, 1.0, cv.NORM_MINMAX)

    # --- 4. 寻找局部最大值作为前景标记 (*** 最重要的调整区域 ***) ---
    # 这是实现分割两部分的关键！需要找到2个独立的峰值。
    # 您需要根据 'bina.png' 中圆的实际大小和重叠程度进行多次尝试和调整。

    # current_min_distance：两个峰值（圆心）之间的最小距离。
    # 根据之前的经验，60 可能是个不错的起点，但如果预腐蚀增强了，可能需要重新微调。
    current_min_distance = 60 # <<<<<<<<<< 在这里反复调整！从 40, 50, 60, 80 向上尝试
                               # 每次调整后，运行代码，检查第6张图和第9张图。

    # current_threshold_abs：距离变换值必须高于此阈值才能被认为是峰值。
    # 通常在 0.3 到 0.5 之间。
    current_threshold_abs = 0.35 # <<<<<<<<<< 在这里反复调整！从 0.3, 0.35, 0.4, 0.45 尝试

    coordinates = peak_local_max(dist_transform_norm, 
                                 min_distance=current_min_distance, 
                                 threshold_abs=current_threshold_abs)

    # 创建 sure_fg (确定前景) 图像，只包含找到的标记点
    sure_fg = np.zeros(dist_transform.shape, dtype=np.uint8)
    if len(coordinates) > 0:
        for coords in coordinates:
            sure_fg[coords[0], coords[1]] = 255
    else:
        print("警告：peak_local_max 未找到任何局部最大值作为前景标记。")
        print("请尝试减小 'current_min_distance' 或 'current_threshold_abs' 参数。")
    
    # (可选) 对 sure_fg 进行小范围膨胀，使标记更明显。
    # 仅当标记太小导致 connectedComponents 识别困难时才需要。
    # 但要非常小心，避免膨胀后标记再次连接。
    sure_fg = cv.dilate(sure_fg, np.ones((3,3), np.uint8), iterations=1)

    # --- 5. 确定背景 (Sure Background) ---
    # 通常通过膨胀原始开运算图像来获得
    sure_bg = cv.dilate(opening, kernel_open, iterations=3) 

    # --- 6. 未知区域 ---
    # 未知区域 = 确定背景 - 确定前景
    unknown = cv.subtract(sure_bg, sure_fg)

    # --- 7. 生成标记 (Marker Labelling) ---
    # connectedComponents 为每个独立的确定前景区域分配一个唯一的标签
    _, markers = cv.connectedComponents(sure_fg)
    num_foreground_markers = np.max(markers) # 获取前景标记的数量

    # 标记调整：背景通常设为1，未知区域设为0
    markers = markers + 1
    markers[unknown == 255] = 0

    # --- 8. 执行分水岭算法 ---
    markers = np.int32(markers) # markers 必须是 int32 类型
    
    # 检查标记数量，如果不是两个，很可能分割不正确
    if num_foreground_markers != 2: 
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"!!!!!! 警告：检测到的前景标记数量为 {num_foreground_markers}，不等于期望的2个。!!!!!!")
        print("!!!!!! 请务必调整 peak_local_max 参数，直到找到2个独立的标记！      !!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    
    labels_watershed = cv.watershed(img_display_color.copy(), markers)

    # --- 打印诊断信息 (在图表绘制前打印，更早看到) ---
    print(f"1. 原始灰度图像尺寸: {img_original_gray.shape}, 像素范围: [{np.min(img_original_gray)}, {np.max(img_original_gray)}]")
    print(f"2. 二值图像尺寸: {img_binary.shape}, 像素范围: [{np.min(img_binary)}, {np.max(img_binary)}]")
    print(f"3. 开运算后图像尺寸: {opening.shape}, 像素范围: [{np.min(opening)}, {np.max(opening)}]")
    print(f"4. 预腐蚀后图像尺寸: {pre_eroded_for_dt.shape}, 像素范围: [{np.min(pre_eroded_for_dt)}, {np.max(pre_eroded_for_dt)}]")
    print(f"5. 距离变换后图像尺寸: {dist_transform.shape}, 像素范围: [{np.min(dist_transform)}, {np.max(dist_transform)}]")
    print(f"6. 确定前景图像尺寸: {sure_fg.shape}, 像素范围: [{np.min(sure_fg)}, {np.max(sure_fg)}]")
    print(f"7. 确定背景图像尺寸: {sure_bg.shape}, 像素范围: [{np.min(sure_bg)}, {np.max(sure_bg)}]")
    print(f"8. 未知区域图像尺寸: {unknown.shape}, 像素范围: [{np.min(unknown)}, {np.max(unknown)}]")
    print(f"9. 初始标记的唯一值 (在Sure Foreground上): {np.unique(markers[sure_fg == 255])}")
    print(f"   **检测到的前景标记数量: {num_foreground_markers}**")
    print(f"10. 最终标记的唯一值 (输入分水岭): {np.unique(labels_watershed)}")
    print(f"11. 分水岭结果标签的唯一值: {np.unique(labels_watershed)}")

    # --- 图表绘制 ---

    # 第一组图：前8张图，按2*4格式排列在一个窗口
    plt.figure(figsize=(20, 10)) 
    
    plt.subplot(2, 4, 1) 
    plt.imshow(img_original_gray, cmap='gray')
    plt.title('1. Original bina.png (Grayscale)')
    plt.axis('off')

    plt.subplot(2, 4, 2) 
    plt.imshow(img_binary, cmap='gray')
    plt.title('2. Initial Binary Image')
    plt.axis('off')

    plt.subplot(2, 4, 3) 
    plt.imshow(opening, cmap='gray')
    plt.title('3. Morphological Opening (Noise Removed)')
    plt.axis('off')

    plt.subplot(2, 4, 4) 
    plt.imshow(pre_eroded_for_dt, cmap='gray')
    plt.title(f'4. Pre-Eroded (K:{kernel_pre_erode[0][0]}, I:{iterations_pre_erode})')
    plt.axis('off')
    
    plt.subplot(2, 4, 5) 
    plt.imshow(dist_transform, cmap='jet')
    plt.title('5. Distance Transform (on Pre-Eroded)')
    plt.axis('off')

    # 关键的第6张图：确定前景 (标记)
    plt.subplot(2, 4, 6) 
    plt.imshow(sure_fg, cmap='gray')
    # 确保f-string语法正确，避免字符串未终止错误
    plt.title(f'6. Sure Foreground (Markers)\nmin_dist:{current_min_distance}, thresh_abs:{current_threshold_abs}')
    plt.axis('off')

    plt.subplot(2, 4, 7) 
    plt.imshow(sure_bg, cmap='gray')
    plt.title('7. Sure Background Area (Dilated)')
    plt.axis('off')

    plt.subplot(2, 4, 8) 
    plt.imshow(unknown, cmap='gray')
    plt.title('8. Unknown Region (Boundary)')
    plt.axis('off')

    plt.tight_layout()
    plt.show(block=False) 

    # 第二组图：剩余的图 (最终标记和结果) 在另一个窗口
    plt.figure(figsize=(14, 7)) 
    
    plt.subplot(1, 2, 1) 
    plt.imshow(markers, cmap='nipy_spectral')
    plt.title(f'9. Final Markers ({num_foreground_markers} found)')
    plt.axis('off')

    plt.subplot(1, 2, 2) 
    plt.imshow(labels_watershed, cmap='nipy_spectral')
    plt.title('10. Watershed Segmentation Result (Labels)')
    plt.axis('off')

    plt.tight_layout()
    plt.show(block=False) 

    # 绘制最终结果 (带分水岭线) 在第三个窗口
    plt.figure(figsize=(7, 7)) 
    img_result_with_lines = img_display_color.copy()
    img_result_with_lines[labels_watershed == -1] = [255, 0, 0] # 红色分水岭线
    plt.imshow(img_result_with_lines)
    plt.title('11. Original Image with Watershed Lines')
    plt.axis('off')
    plt.tight_layout()
    
    print("--- 所有图表已尝试显示。请关闭所有图表窗口继续。---")
    plt.show(block=True) 

if __name__ == '__main__':
    watershed_segment_two_parts()