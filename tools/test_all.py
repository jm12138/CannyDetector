import os
import sys  
file_path = os.path.abspath(__file__)
infer_dir = os.path.dirname(file_path)
package_dir = os.path.dirname(infer_dir)
sys.path.append(package_dir)

import os
import cv2
import time
import numpy as np

from cannydet.units import auto_canny


test_imgs = os.listdir('test_imgs')
test_imgs_path = [os.path.join('test_imgs', img) for img in test_imgs]


def split_list(input_list, n):
    for i in range(0, len(input_list), n):
        yield input_list[i:i + n]


def make_res_dir(name, ext='png'):
    if not os.path.exists(name):
        os.makedirs(name)
    res_imgs_path = []
    for img in test_imgs:
        fname, _ = os.path.splitext(img)
        res_imgs_path.append(os.path.join(name, fname+'.%s' % ext))
    return res_imgs_path


def test_paddle(test_imgs_path=test_imgs_path, thresholds=[[2.5, 5] for _ in range(5)], output_dir='results/paddle', device='cpu'):
    import paddle
    from cannydet.paddle import CannyDetector
    paddle.set_device(device)
    canny_operator = CannyDetector()
    _ = canny_operator(
        paddle.randn((1, 3, 512, 512))
    )
    res_imgs_path = make_res_dir(output_dir)
    for img_path, res_path, (threshold1, threshold2) in zip(test_imgs_path, res_imgs_path, thresholds):
        start = time.time()
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.transpose(img, [2, 1, 0]) / 255.0
        img_tensor = paddle.to_tensor(img[None, ...], dtype='float32')
        result = auto_canny(gray, input=img_tensor,
                            canny_func=canny_operator, scale=1/32)
        res = np.squeeze(result.numpy())
        res = np.transpose(res, [1, 0])
        res = (res*255).astype(np.uint8)
        cv2.imwrite(res_path, res)
        end = time.time()
        print(f'Result save at {res_path}. Run time: {end-start}s')


def test_torch(test_imgs_path=test_imgs_path, output_dir='results/torch', device='cpu'):
    import torch
    from cannydet.torch import CannyDetector
    canny_operator = CannyDetector(device=device).to(device)
    _ = canny_operator(
        torch.randn(1, 3, 512, 512).to(device)
    )
    res_imgs_path = make_res_dir(output_dir)
    for img_path, res_path in zip(test_imgs_path, res_imgs_path):
        start = time.time()
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.transpose(img, [2, 1, 0]) / 255.0
        img_tensor = torch.from_numpy(img[None, ...]).float()
        result = auto_canny(gray, input=img_tensor.to(device),
                            canny_func=canny_operator, scale=1/32)
        res = np.squeeze(result.cpu().numpy())
        res = np.transpose(res, [1, 0])
        res = (res*255).astype(np.uint8)
        cv2.imwrite(res_path, res)
        end = time.time()
        print(f'Result save at {res_path}. Run time: {end-start}s')


def test_python(test_imgs_path=test_imgs_path, thresholds=[[0.1, 0.3] for _ in range(5)], output_dir='results/python'):
    from cannydet.python import canny
    res_imgs_path = make_res_dir(output_dir)
    for img_path, res_path, (threshold1, threshold2) in zip(test_imgs_path, res_imgs_path, thresholds):
        start = time.time()
        img = cv2.imread(img_path, 0)
        res = canny(
            img,
            threshold1=threshold1,
            threshold2=threshold2
        )
        res = (res*255).astype(np.uint8)
        cv2.imwrite(res_path, res)
        end = time.time()
        print(f'Result save at {res_path}. Run time: {end-start}s')


def test_cv2(test_imgs_path=test_imgs_path, output_dir='results/cv2'):
    res_imgs_path = make_res_dir(output_dir)
    for img_path, res_path in zip(test_imgs_path, res_imgs_path):
        start = time.time()
        img = cv2.imread(img_path, 0)
        res = auto_canny(img)
        cv2.imwrite(res_path, res)
        end = time.time()
        print(f'Result save at {res_path}. Run time: {end-start}s')


def test_compare(dirs=['results/python', 'results/torch', 'results/paddle', 'results/cv2'],
            labels=['Python', 'Pytorch', 'Paddle', 'OpenCV'],
            output_dir='results/compare', test_imgs_path=test_imgs_path):
    res_imgs_path = make_res_dir(output_dir)
    font = cv2.FONT_HERSHEY_SIMPLEX
    imgs_name = os.listdir(dirs[0])
    for img, img_path, res_path in zip(imgs_name, test_imgs_path, res_imgs_path):
        imgs = [cv2.imread(os.path.join(dir, img), 0) for dir in dirs]
        resize_imgs = [cv2.resize(
            img, (int(img.shape[1]/img.shape[0]*512), 512)) for img in imgs]
        texted_imgs = [cv2.putText(img, label, (12, 500), font, 1.2, (255, 255, 255), 2)
                       for img, label in zip(resize_imgs, labels)]
        img = cv2.imread(img_path, 0)
        split_imgs = [[cv2.resize(img, (int(img.shape[1]/img.shape[0]*512), 512)),
                       np.zeros((512, int(img.shape[1]/img.shape[0]*512)))]]
        split_imgs += [imgs for imgs in split_list(texted_imgs, len(dirs)//2)]
        result = np.concatenate([np.concatenate(imgs, 1)
                                 for imgs in split_imgs], 0)
        cv2.imwrite(res_path, result)


if __name__ == '__main__':
    print('Run python test...')
    test_python()
    print('Run pytorch on cpu test...')
    test_torch()
    print('Run pytorch on gpu test...')
    test_torch(device='cuda:0')
    print('Run paddle on cpu test...')
    test_paddle()
    print('Run paddle on gpu test...')
    test_paddle(device='gpu:0')
    print('Run OpenCV test...')
    test_cv2()
    print('Run result compare...')
    test_compare()
