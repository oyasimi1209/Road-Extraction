import torch
import torch.nn as nn
import cv2
import os
import numpy as np
from networks.BSNet_structure import BSNet

def test_single_image(model_path, image_path, output_path):
    # 加载模型
    net = BSNet().cuda()
    net = torch.nn.DataParallel(net)
    net.load_state_dict(torch.load(model_path, weights_only=True))
    net.eval()

    # 读取图像
    img = cv2.imread(image_path)
    img = img.transpose(2, 0, 1)[None]
    img = torch.Tensor(np.array(img, np.float32)/255.0 * 3.2 - 1.6).cuda()

    # 预测
    with torch.no_grad():
        mask = net(img)
        mask = mask.squeeze().cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8) * 255

    # 保存结果
    cv2.imwrite(output_path, mask)
    print(f"预测结果已保存到: {output_path}")

if __name__ == "__main__":
    # 使用绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "boost_train", "weights", "17220massa_roadnet_1.th")
    test_dir = os.path.join(os.path.dirname(current_dir), "data", "Shaoxing", "test_satellite")
    
    # 创建corner_detect所需的输出目录
    output_dir = os.path.join(os.path.dirname(current_dir), "out", "corner_detect", "seg_mask")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Looking for images in: {test_dir}")
    print(f"Saving results to: {output_dir}")
    
    # 获取所有图像文件名
    image_files = [f for f in os.listdir(test_dir) if f.endswith('_sat.png')]
    image_files = image_files[:-1]
    
    print(f"将测试以下{len(image_files)}张图片:")
    for img_name in image_files:
        print(f"- {img_name}")
    
    # 测试选定的图像
    for img_name in image_files:
        image_path = os.path.join(test_dir, img_name)
        # 修改输出文件名格式为 region_seg.png
        output_name = img_name.replace('_sat.png', '_seg.png')
        output_path = os.path.join(output_dir, output_name)
        print(f"\n处理图像: {img_name}")
        test_single_image(model_path, image_path, output_path) 