from subprocess import Popen
import os
import cv2
import numpy as np
from evaluate_np import IOU, thin_image

def evaluate_initial_seg():
    """评估initial_seg的分割结果并生成第一个训练子集"""
    print("\n" + "="*50)
    print("Step 1: 评估initial_seg分割结果")
    print("="*50)
    
    # 设置路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_img_dir = os.path.join(base_dir, 'data/Shaoxing/test_satellite/')
    test_label_dir = os.path.join(base_dir, 'data/Shaoxing/test_label/')
    initial_mask_dir = os.path.join(base_dir, 'output/test_results/')
    
    print(f"\n正在使用以下目录：")
    print(f"测试图片目录: {test_img_dir}")
    print(f"测试标签目录: {test_label_dir}")
    print(f"初始分割结果: {initial_mask_dir}")
    
    # 创建boost_train目录结构
    boost_train_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'boost_train')
    txt_dir = os.path.join(boost_train_dir, 'txt_files/massa')
    os.makedirs(txt_dir, exist_ok=True)
    print(f"\n创建数据目录: {txt_dir}")
    
    # 获取所有测试图片
    test_images = [f for f in os.listdir(test_img_dir) if f.endswith('_sat.png')]
    print(f"\n找到 {len(test_images)} 张测试图片")
    
    image_prob = {}
    processed_count = 0
    low_iou_count = 0
    
    print("\n开始处理图片并计算IoU...")
    for image_name in test_images:
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"已处理: {processed_count}/{len(test_images)} 张图片")
            
        base_name = image_name[:-8]  # 移除 '_sat.png'
        mask_path = os.path.join(initial_mask_dir, base_name + '_mask.png')
        label_path = os.path.join(test_label_dir, base_name + '_osm.png')
        
        if not os.path.exists(mask_path) or not os.path.exists(label_path):
            print(f"警告: 找不到文件 - {base_name}")
            continue
            
        # 读取预测结果和真值
        prediction = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        truevalue = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        
        if prediction is None or truevalue is None:
            print(f"警告: 无法读取图片 - {base_name}")
            continue
            
        # 计算IoU
        prediction = np.array(prediction, np.float32) / 255.0
        truevalue = np.array(truevalue, np.float32) / 255.0
        dilated_kernel = np.ones((3,3))
        gt_buffer = cv2.dilate(truevalue.astype(np.uint8), dilated_kernel)
        gt_buffer = np.array(gt_buffer, np.float32)
        
        res = IOU(prediction, truevalue, gt_buffer, is_buffer=False)
        res.cal_iou(None)
        
        # 根据IoU设置初始权重
        if res.iou < 0.7:  # IoU阈值
            image_prob[image_name] = 2.0  # 给予较高权重
            low_iou_count += 1
        else:
            image_prob[image_name] = 1.0
    
    print(f"\n处理完成！")
    print(f"总共处理: {processed_count} 张图片")
    print(f"IoU < 0.7的图片数量: {low_iou_count}")
    print(f"IoU >= 0.7的图片数量: {processed_count - low_iou_count}")
    
    # 归一化权重
    print("\n正在归一化权重...")
    total_weight = sum(image_prob.values())
    for key in image_prob:
        image_prob[key] /= total_weight
    
    # 保存图片列表和权重
    print("\n保存训练数据...")
    train_list_path = os.path.join(txt_dir, 'train_image_file_1.txt')
    with open(train_list_path, 'w') as f:
        for name in image_prob.keys():
            f.write(name + '\n')
    print(f"已保存图片列表到: {train_list_path}")
            
    prob_file_path = os.path.join(txt_dir, 'image_prob_1.txt')
    with open(prob_file_path, 'w') as f:
        f.write("classifier #0's weight: 1.0\n")  # 初始分类器权重
        for name, prob in image_prob.items():
            f.write(f"{name},{prob}\n")
    print(f"已保存权重信息到: {prob_file_path}")
    
    print("\n初始评估完成！准备开始BSNet训练...")
    print("="*50 + "\n")

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

print("\n" + "="*50)
print("BSNet训练流程开始")
print("="*50)

# 首先评估initial_seg结果并生成第一个训练子集
evaluate_initial_seg()

# 使用python而不是python3，并添加完整路径
cmd1 = f"python {os.path.join(current_dir, 'train_ensemble.py')} 1"
cmd2 = f"python {os.path.join(current_dir, 'train_ensemble.py')} 2"
cmd3 = f"python {os.path.join(current_dir, 'train_ensemble.py')} 3"
cmd4 = f"python {os.path.join(current_dir, 'train_ensemble.py')} 4"
cmd5 = f"python {os.path.join(current_dir, 'train_ensemble.py')} 5"

print("\nStep 2: 开始训练BSNet分类器")
print("="*50)

print("\n训练第1个分类器...")
p1 = Popen(cmd1, shell=True)
p1.wait()
print("\n第1个分类器训练完成！")

print("\n训练第2个分类器...")
p2 = Popen(cmd2, shell=True)
p2.wait()
print("\n第2个分类器训练完成！")

print("\n训练第3个分类器...")
p3 = Popen(cmd3, shell=True)
p3.wait()
print("\n第3个分类器训练完成！")

print("\n训练第4个分类器...")
p4 = Popen(cmd4, shell=True)
p4.wait()
print("\n第4个分类器训练完成！")

print("\n训练第5个分类器...")
p5 = Popen(cmd5, shell=True)
p5.wait()
print("\n第5个分类器训练完成！")

print("\n" + "="*50)
print("所有训练已完成！")
print("="*50 + "\n")

