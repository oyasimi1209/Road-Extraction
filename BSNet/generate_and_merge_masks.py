import torch
import torch.nn as nn
import cv2
import os
import numpy as np
from networks.BSNet_structure import BSNet
from tqdm import tqdm

def read_voting_weight(prob_file):
    """从image_prob文件中读取模型权重"""
    with open(prob_file, 'r') as f:
        first_line = f.readline().strip()
        # 格式: "classifier #X's weight: Y"
        weight = float(first_line.split(':')[1].strip())
    return weight

def test_single_image(model_path, image_path, output_path):
    """使用BSNet模型生成单张图片的mask"""
    net = BSNet().cuda()
    net = torch.nn.DataParallel(net)
    net.load_state_dict(torch.load(model_path))
    net.eval()
    
    img = cv2.imread(image_path)
    img = img.transpose(2, 0, 1)[None]
    img = torch.Tensor(np.array(img, np.float32)/255.0 * 3.2 - 1.6).cuda()
    
    with torch.no_grad():
        mask = net(img)
        mask = mask.squeeze().cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8) * 255
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, mask)

def merge_boost_result(initial_mask, T, boost_mask, voting, withInitial=True):
    """融合多个BSNet的结果"""
    pro_boost_mask = [" "]
    all_weight_boost = 0
    all_voting = 0
    
    for i in range(1, T + 1):
        pro_boost_mask.append(boost_mask[i] / 255)
        all_weight_boost += voting[i] * pro_boost_mask[i]
        all_voting += voting[i]
    
    ave_weight_boost = all_weight_boost / all_voting
    result = ave_weight_boost
    
    pro_initial_mask = initial_mask / 255
    if withInitial:
        result = pro_initial_mask
    
    result[ave_weight_boost > 0.8] = ave_weight_boost[ave_weight_boost > 0.8]
    
    pro_boost = result * 255
    th, bina_boost = cv2.threshold(pro_boost.astype(np.uint8), 0, 255, cv2.THRESH_OTSU)
    if th < 80:
        bina_boost[pro_boost < 128] = 0
        bina_boost[pro_boost >= 128] = 255
    
    return pro_boost, bina_boost

if __name__ == "__main__":
    # 设置路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_dir = os.path.join(base_dir, "data/Shaoxing/test_satellite")
    initial_mask_dir = os.path.join(base_dir, "output/test_results")
    
    # BSNet模型路径
    model_paths = [
        os.path.join("boost_train/weights", "224750massa_roadnet_1.th"),
        os.path.join("boost_train/weights", "231022massa_roadnet_2.th"),
        os.path.join("boost_train/weights", "232128massa_roadnet_3.th"),
        os.path.join("boost_train/weights", "232950massa_roadnet_4.th"),
        os.path.join("boost_train/weights", "233812massa_roadnet_5.th")
    ]
    
    # BSNet输出路径
    output_dirs = [
        os.path.join(base_dir, f"output/bsnet{i}")
        for i in range(1, 5)
    ]
    
    # 最终融合结果保存路径
    final_output_dir = os.path.join(base_dir, "output/final_fusion")
    os.makedirs(final_output_dir, exist_ok=True)
    
    # 获取所有测试图像
    test_images = [f for f in os.listdir(test_dir) if f.endswith('_sat.png')]
    print(f"找到 {len(test_images)} 张测试图片")
    
    # 1. 为每个BSNet模型生成mask
    print("\n第一步：生成每个BSNet模型的mask")
    for model_path, output_dir in zip(model_paths, output_dirs):
        if not os.path.exists(model_path):
            print(f"警告：找不到模型文件 {model_path}")
            continue
            
        print(f"\n处理模型：{os.path.basename(model_path)}")
        os.makedirs(output_dir, exist_ok=True)
        
        for img_name in tqdm(test_images):
            image_path = os.path.join(test_dir, img_name)
            output_path = os.path.join(output_dir, img_name.replace('_sat.png', '_mask.png'))
            test_single_image(model_path, image_path, output_path)
    
    # 2. 读取模型权重
    print("\n第二步：读取模型权重")
    voting = [1.0]  # 初始权重
    for i in range(1, 5):
        prob_file = os.path.join("boost_train/txt_files/massa", f"image_prob_{i}.txt")
        if os.path.exists(prob_file):
            weight = read_voting_weight(prob_file)
            voting.append(weight)
            print(f"模型 {i} 权重: {weight}")
        else:
            print(f"警告：找不到权重文件 {prob_file}")
            voting.append(1.0)  # 默认权重
    
    # 3. 融合结果
    print("\n第三步：融合所有模型的结果")
    for img_name in tqdm(test_images):
        # 读取初始mask
        initial_mask_path = os.path.join(initial_mask_dir, img_name.replace('_sat.png', '_mask.png'))
        initial_mask = cv2.imread(initial_mask_path, cv2.IMREAD_GRAYSCALE)
        
        if initial_mask is None:
            print(f"警告：找不到初始mask文件 {initial_mask_path}")
            continue
        
        # 读取所有BSNet的mask
        boost_mask = [" "]  # 第0个位置留空
        for output_dir in output_dirs:
            mask_path = os.path.join(output_dir, img_name.replace('_sat.png', '_mask.png'))
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"警告：找不到BSNet mask文件 {mask_path}")
                mask = np.zeros_like(initial_mask)
            boost_mask.append(mask)
        
        # 融合
        pro_boost, bina_boost = merge_boost_result(
            initial_mask=initial_mask,
            T=4,
            boost_mask=boost_mask,
            voting=voting,
            withInitial=True
        )
        
        # 保存结果
        base_name = img_name.replace('_sat.png', '')
        cv2.imwrite(os.path.join(final_output_dir, f"{base_name}_prob.png"), pro_boost)
        cv2.imwrite(os.path.join(final_output_dir, f"{base_name}_binary.png"), bina_boost)
    
    print("\n处理完成！")
    print(f"最终结果保存在：{final_output_dir}") 