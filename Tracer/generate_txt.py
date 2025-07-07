import json

json_path = r"D:\AI_learning\road_extraction\out\corner_detect\corners\Shaoxing_starting_locations.json"
txt_path = r"D:\AI_learning\road_extraction\out\corner_detect\corners\Shaoxing_seg.txt"

with open(json_path, 'r') as f:
    data = json.load(f)

with open(txt_path, 'w') as f:
    for tile_points in data.values():
        for pt in tile_points:
            f.write(f"{pt['x']},{pt['y']}\n")