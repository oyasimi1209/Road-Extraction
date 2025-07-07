# Road Graph and Satellite Image Visualization

这个项目提供了用于可视化道路图（graph）和卫星图像叠加的工具。

## 文件说明

- `simple_visualization.py`: 简化的可视化脚本，易于使用
- `visualize_graph_satellite.py`: 完整的可视化类，功能更丰富
- `requirements.txt`: 所需的Python依赖包

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 简单可视化脚本

运行 `simple_visualization.py` 来创建可视化：

```bash
python simple_visualization.py
```

这个脚本会：
- 分析graph的基本信息（顶点数、边数、边界等）
- 创建只显示graph的可视化
- 创建graph和卫星图叠加的可视化

### 2. 完整可视化类

使用 `visualize_graph_satellite.py` 中的 `GraphSatelliteVisualizer` 类：

```python
from visualize_graph_satellite import GraphSatelliteVisualizer

# 创建可视化器
visualizer = GraphSatelliteVisualizer(
    graph_file="path/to/your/graph.graph",
    satellite_dir="path/to/your/satellite/images"
)

# 加载数据
visualizer.load_graph()
visualizer.load_satellite_images()

# 分析graph
visualizer.analyze_graph()

# 创建可视化
visualizer.create_composite_image(
    output_path="output.png",
    show_vertices=True,
    show_edges=True,
    edge_color='red',
    vertex_color='blue',
    vertex_size=5,
    edge_width=3,
    alpha=0.8
)
```

## 数据格式

### Graph文件格式
Graph文件应该包含：
1. 顶点坐标（每行一个顶点，格式：x y）
2. 空行分隔
3. 边连接（每行一条边，格式：src_id dst_id）

### 卫星图像格式
卫星图像应该按照网格命名，例如：
- `c2_0_0_sat.png`
- `c2_0_1_sat.png`
- `c2_1_0_sat.png`
- 等等

其中 `c2_x_y` 表示网格坐标。

## 输出文件

脚本会生成以下文件：
- `graph_only.png`: 只显示graph的可视化
- `visualization.png`: graph和卫星图叠加的可视化
- `graph_satellite_overlay.png`: 使用完整类生成的可视化

## 自定义选项

### 可视化参数
- `edge_color`: 边的颜色
- `vertex_color`: 顶点的颜色
- `vertex_size`: 顶点大小
- `edge_width`: 边宽度
- `alpha`: 透明度

### 图像参数
- `output_path`: 输出文件路径
- `show_vertices`: 是否显示顶点
- `show_edges`: 是否显示边

## 注意事项

1. 确保graph文件和卫星图像路径正确
2. 卫星图像应该是相同尺寸的正方形图像
3. 图像文件名应该按照网格坐标命名
4. 需要足够的内存来处理大型图像

## 故障排除

### 常见问题

1. **ImportError: No module named 'discoverlib'**
   - 确保在正确的目录中运行脚本
   - 检查 `discoverlib` 模块是否存在

2. **FileNotFoundError**
   - 检查文件路径是否正确
   - 确保文件存在且有读取权限

3. **MemoryError**
   - 减少图像尺寸
   - 使用更小的tile范围

4. **图像不显示**
   - 检查matplotlib后端设置
   - 确保有图形界面支持

## 示例输出

脚本会输出类似以下信息：

```
=== Road Graph and Satellite Image Visualization ===
Loading graph from D:\AI_learning\road_extraction\out\graph_infer\line_mergec2.infer.graph
Loaded graph with 481 vertices and 480 edges

=== Graph Analysis ===
Number of vertices: 481
Number of edges: 480
Edge length statistics:
  Min: 1.00
  Max: 156.78
  Mean: 23.45
  Median: 18.92
Graph bounds:
  X: -3980 to 3976
  Y: -3977 to 3975
  Width: 7956
  Height: 7952

Creating graph-only visualization...
Saved graph visualization to graph_only.png

Creating composite visualization...
Loading satellite images from D:\AI_learning\road_extraction\data\Shaoxing\test_satellite
Found 100 satellite images
Successfully loaded 100 satellite images
Graph bounds: Point(-3980, -3977) to Point(3976, 3975)
Tile size: 1024
Tile range: X(-4 to 4), Y(-4 to 4)
Creating composite image of size 8192x8192
Saved visualization to visualization.png
``` 