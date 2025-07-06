import os
import json

# 你的txt文件夹路径
txt_dir = r"D:\AI_learning\road_extraction\out\corner_detect\corners"
result = {}

for fname in os.listdir(txt_dir):
    if fname.endswith('.txt'):
        key = fname[:-4]  # 去掉 .txt
        result[key] = []
        with open(os.path.join(txt_dir, fname), 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                x, y = map(int, line.split(','))
                result[key].append({"x": x, "y": y})

with open(os.path.join(txt_dir, "Shaoxing_starting_locations.json"), "w") as f:
    json.dump(result, f, indent=2)

print("已生成 Shaoxing_starting_locations.json")