# 道路提取项目环境配置

## 系统环境
- 操作系统：Windows 10
- Python版本：3.x (Anaconda环境)
- CUDA版本：支持RTX 4090的CUDA版本
- GPU：NVIDIA RTX 4090

## Python包依赖
```bash
# 基础科学计算包
numpy
scipy
scikit-image

# 深度学习框架
torch
torchvision
tensorflow  # 用于Tracer

# 图像处理
opencv-python (cv2)

# 其他工具包
tqdm
matplotlib
```

## 项目结构
```
road_extraction/
├── BSNet/
│   ├── networks/
│   │   └── BSNet_structure.py
│   ├── boost_train/
│   │   ├── weights/
│   │   ├── logs/
│   │   └── txt_files/
│   ├── train_ensemble.py
│   ├── train_tiny.py
│   ├── framework.py
│   ├── data.py
│   └── test_current.py
├── Tracer/
│   ├── train.py
│   ├── model.py
│   ├── model_utils.py
│   ├── geom.py
│   ├── graph.py
│   ├── infer.py
│   └── tileloader.py
├── data/
│   └── Shaoxing/
│       ├── train_satellite/
│       ├── train_label/
│       ├── test_satellite/
│       └── test_label/
├── out/
│   └── corner_detect/
│       ├── seg_mask/
│       └── corners/
├── logs/
│   └── road_tracer.log
├── corner_detect.py
└── environment.md
```

## 安装步骤

1. 创建并激活Conda环境：
```bash
conda create -n road_extraction python=3.x
conda activate road_extraction
```

2. 安装PyTorch（根据CUDA版本选择）：
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

3. 安装TensorFlow（用于Tracer）：
```bash
pip install tensorflow
```

4. 安装其他依赖：
```bash
pip install numpy scipy scikit-image opencv-python tqdm matplotlib
```

## 使用说明

1. 训练模型：
```bash
python BSNet/train_ensemble.py
```

2. 测试模型：
```bash
python BSNet/test_current.py
```

3. 角点检测：
```bash
python corner_detect.py
```

4. 运行Tracer训练：
```bash
# 首先确保创建必要的目录
mkdir -p logs
mkdir -p data/imagery
mkdir -p data/graphs
mkdir -p data/json

# 然后运行训练脚本
python Tracer/train.py
```

## 注意事项

1. 确保数据目录结构正确：
   - 训练数据应放在 `data/Shaoxing/train_satellite/` 和 `data/Shaoxing/train_label/`
   - 测试数据应放在 `data/Shaoxing/test_satellite/` 和 `data/Shaoxing/test_label/`
   - Tracer所需数据应放在 `data/imagery/` 和 `data/graphs/`

2. 模型权重文件：
   - 训练好的模型权重保存在 `BSNet/boost_train/weights/` 目录
   - 最新的模型权重文件名为 `17220massa_roadnet_1.th`
   - Tracer模型保存在 `model/` 目录

3. 输出结果：
   - 分割结果保存在 `out/corner_detect/seg_mask/`
   - 角点检测结果保存在 `out/corner_detect/corners/`
   - Tracer日志保存在 `logs/road_tracer.log`

4. Tracer配置：
   - 确保 `data/json/pytiles.json` 和 `data/json/starting_locations.json` 文件存在
   - 检查 `tileloader.py` 中的路径配置是否正确
   - 确保有足够的磁盘空间存储训练数据

## 常见问题

1. 如果遇到CUDA相关错误，请检查：
   - CUDA版本是否与PyTorch版本匹配
   - GPU驱动是否正确安装
   - 是否在正确的Conda环境中运行

2. 如果遇到内存不足错误：
   - 减小batch size
   - 使用较小的图像尺寸
   - 确保GPU内存足够

3. 如果遇到路径相关错误：
   - 确保使用正确的路径分隔符
   - 检查文件权限
   - 确保目录存在
   - 检查Tracer所需的配置文件是否存在

4. 如果遇到Tracer相关错误：
   - 确保所有必要的目录都已创建
   - 检查数据文件格式是否正确
   - 验证JSON配置文件是否存在且格式正确
   - 检查日志文件权限 