import cv2

lower = 30  # 最小阈值
upper = 70  # 最大阈值

img_path = 'test_imgs/1-1.jpg'  # 指定测试图像路径
res_path = 'test.png'  # 指定结果保存路径

gray = cv2.imread(img_path, 0)  # 读取灰度图像
edge = cv2.Canny(gray, lower, upper) # Canny 图像边缘检测

cv2.imshow('result', edge)  # 结果预览
cv2.waitKey(0)  # 等待响应
cv2.imwrite(res_path, edge)  # 结果保存
