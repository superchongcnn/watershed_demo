import numpy as np
import matplotlib.pyplot as plt

def generate_vertical_grayscale_bar(width=50, height=256):
    """
    生成一个灰度值从0到255的纵向灰度条。

    Args:
        width (int): 灰度条的宽度（像素）。
        height (int): 灰度条的高度（像素）。默认256，对应0-255的灰度级。

    Returns:
        numpy.ndarray: 包含灰度条图像的二维NumPy数组。
    """
    # 创建一个空的二维数组，高度为256（对应0-255），宽度为指定宽度
    # 数据类型设置为uint8，因为灰度值通常是0-255
    grayscale_bar = np.zeros((height, width), dtype=np.uint8)

    # 填充灰度值
    # 从上到下，每一行代表一个灰度级
    # 灰度值从0（黑色）到255（白色）
    for i in range(height):
        grayscale_bar[i, :] = i  # 将当前行的所有像素设置为灰度值 i

    return grayscale_bar

if __name__ == '__main__':
    # 生成灰度条图像
    bar_image = generate_vertical_grayscale_bar()

    # 使用Matplotlib显示灰度条
    plt.figure(figsize=(2, 6)) # 设置图表大小，使灰度条看起来更细长
    plt.imshow(bar_image, cmap='gray', origin='upper') # 'gray'颜色映射，'upper'表示原点在左上角
    
    # 添加标题和坐标轴标签
    plt.title('Vertical Grayscale Bar (0-255)')
    plt.xlabel('Width')
    plt.ylabel('Grayscale Value (0-255)')
    
    # 设置Y轴刻度，对应灰度值
    plt.yticks(np.arange(0, 256, 32)) # 每32个灰度值一个刻度

    # 隐藏X轴刻度（因为宽度没有实际意义）
    plt.xticks([]) 

    plt.colorbar(label='Grayscale Value') # 添加颜色条，更直观显示灰度值
    plt.tight_layout() # 调整布局，防止标签重叠
    plt.show()

    # 也可以保存为图片文件
    # plt.imsave('vertical_grayscale_bar.png', bar_image, cmap='gray')
    # print("灰度条已保存为 vertical_grayscale_bar.png")