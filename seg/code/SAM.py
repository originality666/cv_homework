import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def simple_panoptic_segmentation(image_path):
    
    # 初始化SAM
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
    # 创建自动掩码生成器
    mask_generator = SamAutomaticMaskGenerator(sam)
    
    # 加载图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 生成掩码
    masks = mask_generator.generate(image)
    print(f"检测到 {len(masks)} 个分割区域")
    
    # 创建全景图
    h, w = image.shape[:2]
    panoptic_result = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 为每个掩码分配颜色
    for i, mask_data in enumerate(masks):
        mask = mask_data['segmentation']
        # 生成随机但可重复的颜色
        np.random.seed(i)
        color = np.random.randint(0, 256, 3)
        panoptic_result[mask] = color
    
    # 可视化
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('org')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(panoptic_result)
    plt.title('result')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(image)
    for mask_data in masks:
        mask = mask_data['segmentation']
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            plt.plot(contour[:, 0, 0], contour[:, 0, 1], 'r-', linewidth=1)
    plt.title('boundary')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return panoptic_result, masks

# 使用示例
if __name__ == "__main__":
    image_path = "..\\data\\img2.jpg"  # 替换为您的图像路径
    panoptic_result, masks = simple_panoptic_segmentation(image_path)
    
    # 保存结果
    cv2.imwrite('..\\result\\SAM_result.jpg', cv2.cvtColor(panoptic_result, cv2.COLOR_RGB2BGR))
    print("done!")