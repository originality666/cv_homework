import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

def mean_shift_segmentation(image_path, quantile=0.3, n_samples=500):
    """
    使用均值移动算法对图像进行分割
    
    参数:
        image_path: 图像路径
        quantile: 用于估计带宽的分位数,值越大带宽越大s
        n_samples: 用于估计带宽的样本数
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("无法读取图像，请检查路径是否正确")
    
    # 转换为RGB格式（OpenCV默认读取为BGR）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 获取图像尺寸
    h, w, c = image_rgb.shape
    
    # 将图像数据转换为二维数组 (像素数, 3)
    flat_image = image_rgb.reshape(h * w, c)
    
    # 估计带宽
    print("正在估计带宽...")
    bandwidth = estimate_bandwidth(flat_image, quantile=quantile, n_samples=n_samples)
    print(f"估计的带宽值: {bandwidth}")
    
    # 应用均值移动算法
    print("正在进行均值移动聚类...")
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(flat_image)
    
    # 获取标签和聚类中心
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    
    # 计算聚类数量
    n_clusters = len(np.unique(labels))
    print(f"聚类数量: {n_clusters}")
    
    # 用聚类中心的值替换每个像素
    segmented_image = cluster_centers[labels].reshape(h, w, c)
    segmented_image = segmented_image.astype(np.uint8)  # 转换为整数类型
    
    # 显示结果
    plt.figure(figsize=(15, 10))
    
    plt.subplot(121)
    plt.imshow(image_rgb)
    plt.title('org')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(segmented_image)
    plt.title(f'result(num of cluster:{n_clusters})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 保存结果
    result_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('..\\result\\segmented_result.jpg', result_bgr)
    print("分割结果已保存为 'segmented_result.jpg'")
    
    return segmented_image

if __name__ == "__main__":
    # 请将下面的路径替换为你自己拍摄的图像路径
    image_path = "..\\data\\img2.jpg"  # 替换为你的图像路径
    
    # 可以调整这些参数来获得更好的分割效果
    try:
        mean_shift_segmentation(
            image_path=image_path,
            quantile=0.1,  # 分位数，较小的值会产生更多聚类
            n_samples=1000  # 用于估计带宽的样本数
        )
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")