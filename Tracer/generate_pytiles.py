import os
import re
import json

# 配置
tile_dir = '../data/Shaoxing/test_satellite'  # 你的 tile 图片文件夹的相对路径
region = 'Shaoxing'
tile_size = 1024

# 匹配文件名的正则
pattern = re.compile(r'[a-zA-Z]\d?_(-?\d+)_(-?\d+)_sat\.png')

tiles = []

print("tile_dir绝对路径：", os.path.abspath(tile_dir))
print("文件列表：", os.listdir(tile_dir))

for fname in os.listdir(tile_dir):
    print("检查文件名：", fname)
    match = pattern.match(fname)
    if match:
        print("匹配成功：", fname)
        x = int(match.group(1))
        y = int(match.group(2))
        bounds = [x * tile_size, y * tile_size, (x + 1) * tile_size, (y + 1) * tile_size]
        tiles.append({
            "region": region,
            "x": x,
            "y": y,
            "filename": fname,
            "bounds": bounds
        })

# 输出 pytiles.json
with open('../data/Shaoxing/pytiles.json', 'w') as f:
    json.dump(tiles, f, indent=2)

print(f"生成了 {len(tiles)} 个 tile 的 pytiles.json")