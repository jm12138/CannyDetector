import torch
import torch.nn as nn

import math
import numpy as np

from scipy.signal import gaussian


class CannyDetector(nn.Module):
    def __init__(self, filter_size=5, std=1.0, device='cpu'):
        super(CannyDetector, self).__init__()
        self.device = device
        # 高斯滤波器
        generated_filters = gaussian(filter_size,std=std).reshape([1,filter_size])

        self.gaussian_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,filter_size), padding=(0,filter_size//2), bias=False)
        self.gaussian_filter_horizontal.weight.data.copy_(torch.from_numpy(generated_filters[None, None, ...].astype('float32')))
        self.gaussian_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size,1), padding=(filter_size//2,0), bias=False)
        self.gaussian_filter_vertical.weight.data.copy_(torch.from_numpy(generated_filters.T[None, None, ...].astype('float32')))


        # Sobel 滤波器
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])

        self.sobel_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2, bias=False)
        self.sobel_filter_horizontal.weight.data.copy_(torch.from_numpy(sobel_filter[None, None, ...].astype('float32')))
        self.sobel_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2, bias=False)
        self.sobel_filter_vertical.weight.data.copy_(torch.from_numpy(sobel_filter.T[None, None, ...].astype('float32')))

        # 定向滤波器
        filter_0 = np.array([[ 0, 0, 0],
                             [ 0, 1, -1],
                             [ 0, 0, 0]])

        filter_45 = np.array([[0, 0, 0],
                              [ 0, 1, 0],
                              [ 0, 0, -1]])

        filter_90 = np.array([[ 0, 0, 0],
                              [ 0, 1, 0],
                              [ 0,-1, 0]])

        filter_135 = np.array([[ 0, 0, 0],
                               [ 0, 1, 0],
                               [-1, 0, 0]])

        filter_180 = np.array([[ 0, 0, 0],
                               [-1, 1, 0],
                               [ 0, 0, 0]])

        filter_225 = np.array([[-1, 0, 0],
                               [ 0, 1, 0],
                               [ 0, 0, 0]])

        filter_270 = np.array([[ 0,-1, 0],
                               [ 0, 1, 0],
                               [ 0, 0, 0]])

        filter_315 = np.array([[ 0, 0, -1],
                               [ 0, 1, 0],
                               [ 0, 0, 0]])

        all_filters = np.stack([filter_0, filter_45, filter_90, filter_135, filter_180, filter_225, filter_270, filter_315])

        self.directional_filter = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=filter_0.shape, padding=filter_0.shape[-1] // 2, bias=False)
        self.directional_filter.weight.data.copy_(torch.from_numpy(all_filters[:, None, ...].astype('float32')))


        # 连通滤波器
        connect_filter = np.array([[1, 1, 1],
                                 [1, 0, 1],
                                 [1, 1, 1]])

        self.connect_filter = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=connect_filter.shape, padding=connect_filter.shape[0]//2, bias=False)
        self.connect_filter.weight.data.copy_(torch.from_numpy(connect_filter[None, None, ...].astype('float32')))

    @torch.no_grad()
    def forward(self, img, threshold1=10.0, threshold2=100.0):
        # 拆分图像通道
        img_r = img[:,0:1] # red channel
        img_g = img[:,1:2] # green channel
        img_b = img[:,2:3] # blue channel

        # Step1: 应用高斯滤波进行模糊降噪
        blur_horizontal = self.gaussian_filter_horizontal(img_r)
        blurred_img_r = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_g)
        blurred_img_g = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_b)
        blurred_img_b = self.gaussian_filter_vertical(blur_horizontal)

        # Step2: 用 Sobel 算子求图像的强度梯度
        grad_x_r = self.sobel_filter_horizontal(blurred_img_r)
        grad_y_r = self.sobel_filter_vertical(blurred_img_r)
        grad_x_g = self.sobel_filter_horizontal(blurred_img_g)
        grad_y_g = self.sobel_filter_vertical(blurred_img_g)
        grad_x_b = self.sobel_filter_horizontal(blurred_img_b)
        grad_y_b = self.sobel_filter_vertical(blurred_img_b)

        # Step2: 确定边缘梯度和方向
        grad_mag = torch.sqrt(grad_x_r**2 + grad_y_r**2)
        grad_mag += torch.sqrt(grad_x_g**2 + grad_y_g**2)
        grad_mag += torch.sqrt(grad_x_b**2 + grad_y_b**2)
        grad_orientation = (torch.atan2(grad_y_r+grad_y_g+grad_y_b, grad_x_r+grad_x_g+grad_x_b) * (180.0/math.pi))
        grad_orientation += 180.0
        grad_orientation =  torch.round(grad_orientation / 45.0) * 45.0

        # Step3: 非最大抑制，边缘细化
        all_filtered = self.directional_filter(grad_mag)

        inidices_positive = (grad_orientation / 45) % 8
        inidices_negative = ((grad_orientation / 45) + 4) % 8

        batch, _, height, width = inidices_positive.shape
        pixel_count = height * width * batch
        pixel_range = torch.Tensor([range(pixel_count)]).to(self.device)

        indices = (inidices_positive.reshape((-1, )) * pixel_count + pixel_range).squeeze()
        channel_select_filtered_positive = all_filtered.reshape((-1, ))[indices.long()].reshape((batch, 1, height, width))

        indices = (inidices_negative.reshape((-1, )) * pixel_count + pixel_range).squeeze()
        channel_select_filtered_negative = all_filtered.reshape((-1, ))[indices.long()].reshape((batch, 1, height, width))

        channel_select_filtered = torch.stack([channel_select_filtered_positive, channel_select_filtered_negative])

        is_max = channel_select_filtered.min(dim=0)[0] > 0.0

        thin_edges = grad_mag.clone()
        thin_edges[is_max==0] = 0.0

        # Step4: 双阈值
        low_threshold = min(threshold1, threshold2)
        high_threshold = max(threshold1, threshold2)
        thresholded = thin_edges.clone()
        lower = thin_edges<low_threshold
        thresholded[lower] = 0.0
        higher = thin_edges>high_threshold
        thresholded[higher] = 1.0
        connect_map = self.connect_filter(higher.float())
        middle = torch.logical_and(thin_edges>=low_threshold, thin_edges<=high_threshold)
        thresholded[middle] = 0.0
        connect_map[torch.logical_not(middle)] = 0
        thresholded[connect_map>0] = 1.0
        thresholded[..., 0, :] = 0.0
        thresholded[..., -1, :] = 0.0
        thresholded[..., :, 0] = 0.0
        thresholded[..., :, -1] = 0.0
        thresholded = (thresholded>0.0).float()

        return thresholded


if __name__ == '__main__':
    import cv2
    img_path = '1-5.png'
    res_path = "1-5_torch_our.png"
    img = cv2.imread(img_path)/255.0 # height, width, channel
    img = np.transpose(img, [2, 1, 0]) # channel width height
    canny_operator = CannyDetector()
    result = canny_operator(torch.from_numpy(np.expand_dims(img, axis=0)).float(),threshold1=2.5, threshold2=5 ) # batch channel width height
    res = np.squeeze(result.numpy())
    res = np.transpose(res, [1, 0])
    res = (res*255).astype(np.uint8)
    cv2.imwrite(res_path, res)