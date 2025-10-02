import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def harris_corner_detection(image, k=0.04, threshold=0.01, sigma=1.0):
    """
    Harris角点检测算法实现
    
    参数:
        image: 输入灰度图像
        k: Harris响应函数中的常数（通常0.04-0.06）
        threshold: 响应值阈值（用于筛选角点，通常0.01-0.1）
        sigma: 高斯滤波的标准差（用于平滑图像，减少噪声影响）
    
    返回:
        corners: 角点坐标列表[(x1,y1), (x2,y2), ...]
        response: Harris响应值矩阵
    """
    # 步骤1: 高斯平滑图像（减少噪声）
    def gaussian_kernel(sigma):
        kernel_size = max(3, int(6 * sigma + 1))
        if kernel_size % 2 == 0:
            kernel_size += 1
        size = kernel_size // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        return kernel / np.sum(kernel)  # 归一化
    
    g_kernel = gaussian_kernel(sigma)
    blurred = cv2.filter2D(image, -1, g_kernel)  # 使用OpenCV的滤波函数提高效率
    
    # 步骤2: 计算x和y方向的梯度（使用Sobel算子）
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    Ix = cv2.filter2D(blurred, -1, sobel_x)  # x方向梯度
    Iy = cv2.filter2D(blurred, -1, sobel_y)  # y方向梯度
    
    # 步骤3: 计算梯度的协方差矩阵元素
    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    Ixy = Ix * Iy
    
    # 对协方差矩阵元素进行高斯平滑（局部区域整合）
    Ix2_smoothed = cv2.filter2D(Ix2, -1, g_kernel)
    Iy2_smoothed = cv2.filter2D(Iy2, -1, g_kernel)
    Ixy_smoothed = cv2.filter2D(Ixy, -1, g_kernel)
    
    # 步骤4: 计算Harris响应值 R = det(M) - k*(trace(M))^2
    det_M = Ix2_smoothed * Iy2_smoothed - Ixy_smoothed ** 2  # 行列式
    trace_M = Ix2_smoothed + Iy2_smoothed  # 迹
    response = det_M - k * (trace_M ** 2)
    
    # 步骤5: 阈值筛选（保留响应值高的点）
    max_response = np.max(response)
    if max_response == 0:
        return [], response
    corner_mask = response > threshold * max_response  # 阈值化
    
    # 步骤6: 非极大值抑制（3x3邻域内保留最大值）
    corners = []
    h, w = response.shape
    for i in range(1, h-1):
        for j in range(1, w-1):
            if corner_mask[i, j]:
                # 检查3x3邻域是否为最大值
                if response[i, j] == np.max(response[i-1:i+2, j-1:j+2]):
                    corners.append((j, i))  # (x,y)坐标
    
    return corners, response

def draw_corners(image, corners, color=(0, 255, 0), radius=3, thickness=2):
    """在图像上绘制角点"""
    image_with_corners = image.copy()
    for (x, y) in corners:
        cv2.circle(image_with_corners, (x, y), radius, color, thickness)
    return image_with_corners

def save_image(image, save_path, filename):
    """保存图像"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    full_path = os.path.join(save_path, filename)
    cv2.imwrite(full_path, image)
    print(f"图像已保存至: {full_path}")

# 测试代码
if __name__ == "__main__":
    # 配置参数
    input_image_path = "..\\data\\beauty.jpg"  # 测试图像路径
    save_dir = "..\\result"
    k = 0.05  # Harris参数（推荐0.04-0.06）
    threshold = 0.02  # 响应值阈值（越小检测到的角点越多）
    sigma = 1.0  # 高斯平滑参数
    
    # 读取图像并转换为灰度图
    image = cv2.imread(input_image_path)
    if image is None:
        raise ValueError(f"无法读取图像，请检查路径: {input_image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # 执行Harris角点检测
    corners, response = harris_corner_detection(gray, k=k, threshold=threshold, sigma=sigma)
    print(f"检测到 {len(corners)} 个角点")
    
    # 绘制角点
    image_with_corners = draw_corners(image, corners)
    
    # 保存结果
    save_image(image, save_dir, "original.jpg")
    save_image(cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR), save_dir, "gray.jpg")
    save_image(image_with_corners, save_dir, "harris_corners.jpg")
    
    # 显示结果
    plt.figure(figsize=(15, 10))
    
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("original_image")
    plt.axis("off")
    
    plt.subplot(132)
    plt.imshow(gray, cmap="gray")
    plt.title("gray_image")
    plt.axis("off")
    
    plt.subplot(133)
    plt.imshow(cv2.cvtColor(image_with_corners, cv2.COLOR_BGR2RGB))
    plt.title(f"Harris corner detection (total {len(corners)} corners)")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()