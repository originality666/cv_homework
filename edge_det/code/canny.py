import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def canny_edge_detection(image, low_threshold, high_threshold, sigma=1.0):
    """实现Canny边缘检测算法"""
    # 步骤1: 高斯滤波降噪
    def gaussian_kernel(size, sigma):
        size = int(size // 2)
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2))) * normal
        return g
    
    kernel_size = int(2 * np.ceil(2 * sigma) + 1)
    g_kernel = gaussian_kernel(kernel_size, sigma)
    
    def convolve(image, kernel):
        kernel = np.flipud(np.fliplr(kernel))
        kernel_h, kernel_w = kernel.shape
        image_h, image_w = image.shape
        pad_h, pad_w = kernel_h // 2, kernel_w // 2
        
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        output = np.zeros_like(image)
        
        for i in range(image_h):
            for j in range(image_w):
                output[i, j] = np.sum(padded_image[i:i+kernel_h, j:j+kernel_w] * kernel)
        
        return output
    
    blurred = convolve(image, g_kernel)
    
    # 步骤2: 计算梯度强度和方向
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    grad_x = convolve(blurred, sobel_x)
    grad_y = convolve(blurred, sobel_y)
    
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    grad_magnitude = (grad_magnitude / np.max(grad_magnitude)) * 255
    
    grad_direction = np.arctan2(grad_y, grad_x)
    grad_direction = np.rad2deg(grad_direction)
    grad_direction[grad_direction < 0] += 180
    
    # 步骤3: 非极大值抑制
    image_h, image_w = grad_magnitude.shape
    non_max_suppression = np.zeros((image_h, image_w), dtype=np.uint8)
    
    for i in range(1, image_h - 1):
        for j in range(1, image_w - 1):
            angle = grad_direction[i, j]
            
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q = grad_magnitude[i, j+1]
                r = grad_magnitude[i, j-1]
            elif 22.5 <= angle < 67.5:
                q = grad_magnitude[i+1, j+1]
                r = grad_magnitude[i-1, j-1]
            elif 67.5 <= angle < 112.5:
                q = grad_magnitude[i+1, j]
                r = grad_magnitude[i-1, j]
            else:
                q = grad_magnitude[i+1, j-1]
                r = grad_magnitude[i-1, j+1]
            
            if grad_magnitude[i, j] >= q and grad_magnitude[i, j] >= r:
                non_max_suppression[i, j] = grad_magnitude[i, j]
    
    # 步骤4 & 5: 双阈值检测和边缘连接
    edges = np.zeros_like(non_max_suppression)
    strong = 255
    weak = 100
    
    strong_i, strong_j = np.where(non_max_suppression >= high_threshold)
    edges[strong_i, strong_j] = strong
    
    weak_i, weak_j = np.where((non_max_suppression <= high_threshold) & 
                             (non_max_suppression >= low_threshold))
    edges[weak_i, weak_j] = weak
    
    for i in range(1, image_h - 1):
        for j in range(1, image_w - 1):
            if edges[i, j] == weak:
                if np.any(edges[i-1:i+2, j-1:j+2] == strong):
                    edges[i, j] = strong
                else:
                    edges[i, j] = 0
    
    return edges

def save_image(image, save_path, filename):
    """保存图像到指定路径"""
    # 确保保存目录存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 构建完整保存路径
    full_path = os.path.join(save_path, filename)
    
    # 处理不同类型的图像（彩色/灰度）
    if len(image.shape) == 3:
        # 彩色图像需要转换为BGR格式才能正确保存
        cv2.imwrite(full_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    else:
        # 灰度图像直接保存
        cv2.imwrite(full_path, image)
    
    print(f"图像已保存至: {full_path}")

# 测试代码
if __name__ == "__main__":
    # 配置参数
    input_image_path = "..\\data\\beauty.jpg"  # 输入图像路径
    save_directory = "..\\result"    # 保存结果的目录
    low_threshold = 50                  # 低阈值
    high_threshold = 200                # 高阈值
    sigma = 2                         # 高斯滤波标准差
    
    # 读取图像
    image = cv2.imread(input_image_path)
    if image is None:
        raise ValueError(f"无法读取图像，请检查路径: {input_image_path}")
    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 执行边缘检测
    custom_edges = canny_edge_detection(gray, low_threshold, high_threshold, sigma)
    opencv_edges = cv2.Canny(gray, low_threshold, high_threshold)
    
    # 保存图像
    save_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), save_directory, "original_image.jpg")
    save_image(gray, save_directory, "gray_image.jpg")
    save_image(custom_edges, save_directory, "custom_canny.jpg")
    save_image(opencv_edges, save_directory, "opencv_canny.jpg")
    
    # 显示结果
    plt.figure(figsize=(15, 10))
    
    plt.subplot(221)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("original_image")
    plt.axis("off")
    
    plt.subplot(222)
    plt.imshow(gray, cmap="gray")
    plt.title("gray_image")
    plt.axis("off")
    
    plt.subplot(223)
    plt.imshow(custom_edges, cmap="gray")
    plt.title("custom_canny reuslt")
    plt.axis("off")
    
    plt.subplot(224)
    plt.imshow(opencv_edges, cmap="gray")
    plt.title("opencv_canny result")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()