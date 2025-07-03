import os
import re
import json

# 配置
tile_dir = './test_satellite'  # 你的 tile 图片文件夹路径
region = 'Shaoxing'
tile_size = 1024

# 匹配文件名的正则
pattern = re.compile(r'@c2_(-?\d+)_(-?\d+)_sat\.png')

tiles = []

for fname in os.listdir(tile_dir):
    match = pattern.match(fname)
    if match:
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
with open('pytiles.json', 'w') as f:
    json.dump(tiles, f, indent=2)

print(f"生成了 {len(tiles)} 个 tile 的 pytiles.json")