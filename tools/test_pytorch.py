import os
import sys  
file_path = os.path.abspath(__file__)
infer_dir = os.path.dirname(file_path)
package_dir = os.path.dirname(infer_dir)
sys.path.append(package_dir)

import cv2
import torch
import numpy as np
from cannydet.torch import CannyDetector

lower = 2.5  # 最小阈值
upper = 5  # 最大阈值

img_path = 'test_imgs/1-1.jpg'  # 指定测试图像路径
res_path = 'test.png'  # 指定结果保存路径

img = cv2.imread(img_path, 1)  # 读取彩色图像
img = np.transpose(img, [2, 1, 0]) / 255.0 # 转置 + 归一化
img_tensor = torch.from_numpy(img[None, ...]).float() # 转换为 Tensor

canny = CannyDetector(device='cpu') # 初始化 Canny 检测器，可设置其运行的设备

edge = canny(img_tensor, lower, upper)  # Canny 图像边缘检测
edge = np.squeeze(edge.cpu().numpy()) # 去除 Batch dim
edge = np.transpose(edge, [1, 0]) # 图像转置
edge = (edge * 255).astype(np.uint8)  # 反归一化

cv2.imshow('result', edge)  # 结果预览
cv2.waitKey(0)  # 等待响应
cv2.imwrite(res_path, edge)  # 结果保存
