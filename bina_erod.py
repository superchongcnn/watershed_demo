import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.feature import peak_local_max # 再次提醒：确保 pip install scikit-image

def debug_watershed_pipeline():
    """
    调试分水岭流程，在关键步骤后显示图像并打印诊断信息。
    将前8张图在一个窗口显示，按2*4格式排列，剩余图在另一个窗口显示。
    """
    print("--- 开始调试分水岭管道 ---")

    image_path = os.path.join('bina.png')
    img_original_gray = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    if img_original_gray is None:
        print(f"错误：无法加载图像。请检查路径是否正确：{image_path}")
        print("请确保 'bina.png' 文件确实存在并可读。")
        return

    img_display_color = cv.cvtColor(img_original_gray, cv.COLOR_GRAY2BGR)
    img_binary = img_original_gray 

    # --- 1. 形态学开运算 (Noise Removed) ---
    kernel_open = np.ones((3,3), np.uint8)
    opening = cv.morphologyEx(img_binary, cv.MORPH_OPEN, kernel_open, iterations = 2)

    # --- 2. 腐蚀分离 (如果使用方案一，这里是强制腐蚀的部分) ---
    kernel_erode = np.ones((4,4), np.uint8) 
    iterations_erode = 2 
    eroded_for_separation = cv.erode(opening, kernel_erode, iterations=iterations_erode)
    
    if np.max(eroded_for_separation) == 0:
        print("致命错误：强制腐蚀操作过度，所有前景物体在图像中消失。")
        print("请检查 'bina.png' 是否为空，或尝试减少 'kernel_erode' 大小和 'iterations_erode' 次数。")
        plt.figure(figsize=(6,6))
        plt.imshow(eroded_for_separation, cmap='gray')
        plt.title('Eroded image is empty - ADJUST EROSION PARAMS!')
        plt.axis('off')
        plt.show() 
        return
    
    # --- 3. 距离变换 ---
    dist_transform = cv.distanceTransform(eroded_for_separation, cv.DIST_L2, 5) # 对腐蚀后的图像进行距离变换

    # --- 4. 确定前景 (Sure Foreground) ---
    threshold_ratio = 0.6 
    if dist_transform.max() == 0:
        sure_fg = np.zeros_like(eroded_for_separation, dtype=np.uint8)
        print("警告：距离变换最大值为0 (max distance is 0)，无法生成前景标记。请检查腐蚀结果或原始图像。")
    else:
        _, sure_fg = cv.threshold(dist_transform, threshold_ratio * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
    
    # --- 5. 确定背景 (Sure Background) ---
    sure_bg = cv.dilate(opening, kernel_open, iterations=3) 

    # --- 6. 未知区域 ---
    unknown = cv.subtract(sure_bg, sure_fg)

    # --- 7. 标记生成 ---
    _, markers = cv.connectedComponents(sure_fg)
    num_foreground_markers = np.max(markers)

    markers = markers + 1
    markers[unknown == 255] = 0

    # --- 8. 执行分水岭算法 ---
    markers = np.int32(markers)
    labels_watershed = cv.watershed(img_display_color.copy(), markers)

    # --- 打印诊断信息 ---
    print(f"1. 原始灰度图像尺寸: {img_original_gray.shape}, 像素范围: [{np.min(img_original_gray)}, {np.max(img_original_gray)}]")
    print(f"2. 二值图像尺寸: {img_binary.shape}, 像素范围: [{np.min(img_binary)}, {np.max(img_binary)}]")
    print(f"3. 开运算后图像尺寸: {opening.shape}, 像素范围: [{np.min(opening)}, {np.max(opening)}]")
    print(f"4. 腐蚀后图像尺寸: {eroded_for_separation.shape}, 像素范围: [{np.min(eroded_for_separation)}, {np.max(eroded_for_separation)}]")
    print(f"5. 距离变换后图像尺寸: {dist_transform.shape}, 像素范围: [{np.min(dist_transform)}, {np.max(dist_transform)}]")
    print(f"6. 确定前景图像尺寸: {sure_fg.shape}, 像素范围: [{np.min(sure_fg)}, {np.max(sure_fg)}]")
    print(f"7. 确定背景图像尺寸: {sure_bg.shape}, 像素范围: [{np.min(sure_bg)}, {np.max(sure_bg)}]")
    print(f"8. 未知区域图像尺寸: {unknown.shape}, 像素范围: [{np.min(unknown)}, {np.max(unknown)}]")
    print(f"9. 初始标记的唯一值 (在Sure Foreground上): {np.unique(markers[sure_fg == 255])}")
    print(f"   检测到的前景标记数量: {num_foreground_markers}")
    print(f"10. 最终标记的唯一值 (输入分水岭): {np.unique(markers)}")
    print(f"11. 分水岭结果标签的唯一值: {np.unique(labels_watershed)}")

    if num_foreground_markers <= 1:
        print("警告：前景标记数量不足，相交的圆可能仍未分开。")
        print("请尝试：a) 微调腐蚀的 'kernel_erode' 和 'iterations_erode'。")
        print("         b) 降低 'threshold_ratio' (例如 0.5)。")
        print("         c) 考虑使用方案二 (局部最大值) 或检查 'bina.png'。")
    
    # --- 第一组图：前8张图，按2*4格式排列在一个窗口 ---
    plt.figure(figsize=(20, 10)) # 适当调整窗口大小以便容纳8张图 (2行4列)
    
    # 图 1: 原始图像
    plt.subplot(2, 4, 1) # 2行4列的第1个
    plt.imshow(img_original_gray, cmap='gray')
    plt.title('1. Original bina.png (Grayscale)')
    plt.axis('off')

    # 图 2: 二值图像
    plt.subplot(2, 4, 2) # 2行4列的第2个
    plt.imshow(img_binary, cmap='gray')
    plt.title('2. Initial Binary Image')
    plt.axis('off')

    # 图 3: 形态学开运算
    plt.subplot(2, 4, 3) # 2行4列的第3个
    plt.imshow(opening, cmap='gray')
    plt.title('3. Morphological Opening (Noise Removed)')
    plt.axis('off')

    # 图 4: 强制腐蚀图像
    plt.subplot(2, 4, 4) # 2行4列的第4个
    plt.imshow(eroded_for_separation, cmap='gray')
    plt.title(f'4. Force Eroded (K:{kernel_erode[0][0]}, I:{iterations_erode})')
    plt.axis('off')
    
    # 图 5: 距离变换
    plt.subplot(2, 4, 5) # 2行4列的第5个
    plt.imshow(dist_transform, cmap='jet')
    plt.title('5. Distance Transform (on Eroded)')
    plt.axis('off')

    # 图 6: 确定前景 (标记)
    plt.subplot(2, 4, 6) # 2行4列的第6个
    plt.imshow(sure_fg, cmap='gray')
    plt.title(f'6. Sure Foreground (Markers) - Thresh:{threshold_ratio}')
    plt.axis('off')

    # 图 7: 确定背景
    plt.subplot(2, 4, 7) # 2行4列的第7个
    plt.imshow(sure_bg, cmap='gray')
    plt.title('7. Sure Background Area (Dilated)')
    plt.axis('off')

    # 图 8: 未知区域
    plt.subplot(2, 4, 8) # 2行4列的第8个
    plt.imshow(unknown, cmap='gray')
    plt.title('8. Unknown Region (Boundary)')
    plt.axis('off')

    plt.tight_layout()
    plt.show(block=False) # 非阻塞显示第一组图

    # --- 第二组图：剩余的图 (标记和最终结果) 在另一个窗口 ---
    plt.figure(figsize=(14, 7)) # 调整窗口大小
    
    # 图 9: 最终标记 (输入分水岭)
    plt.subplot(1, 2, 1) # 1行2列的第一个
    plt.imshow(markers, cmap='nipy_spectral')
    plt.title(f'9. Final Markers ({num_foreground_markers} found)')
    plt.axis('off')

    # 图 10: 分水岭结果标签
    plt.subplot(1, 2, 2) # 1行2列的第二个
    plt.imshow(labels_watershed, cmap='nipy_spectral')
    plt.title('10. Watershed Segmentation Result (Labels)')
    plt.axis('off')

    plt.tight_layout()
    plt.show(block=False) # 非阻塞显示第二组图

    # --- 绘制最终结果 (带分水岭线) 在第三个窗口 ---
    plt.figure(figsize=(7, 7)) # 单图窗口
    img_result_with_lines = img_display_color.copy()
    img_result_with_lines[labels_watershed == -1] = [255, 0, 0] # 红色分水岭线
    plt.imshow(img_result_with_lines)
    plt.title('11. Original Image with Watershed Lines')
    plt.axis('off')
    plt.tight_layout()
    
    print("--- 所有图表已尝试显示。请关闭所有图表窗口继续。---")
    plt.show(block=True) # 最后一个 show 使用 block=True 阻塞程序，直到所有图表关闭

if __name__ == '__main__':
    debug_watershed_pipeline()