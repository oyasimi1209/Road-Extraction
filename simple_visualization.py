import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from discoverlib import graph, geom

def load_graph(graph_file):
    """加载graph文件"""
    print(f"Loading graph from {graph_file}")
    g = graph.read_graph(graph_file, merge_duplicates=True)
    print(f"Loaded graph with {len(g.vertices)} vertices and {len(g.edges)} edges")
    return g

def load_satellite_images(satellite_dir):
    """加载卫星图像"""
    print(f"Loading satellite images from {satellite_dir}")
    
    satellite_images = {}
    image_files = glob.glob(os.path.join(satellite_dir, "*.png"))
    print(f"Found {len(image_files)} satellite images")
    
    for image_file in image_files:
        try:
            filename = os.path.basename(image_file)
            if filename.startswith('c2_'):
                # 解析c2格式的文件名
                parts = filename.replace('c2_', '').replace('_sat.png', '').split('_')
                if len(parts) == 2:
                    x, y = int(parts[0]), int(parts[1])
                    img = Image.open(image_file)
                    satellite_images[(x, y)] = img
        except Exception as e:
            print(f"Error loading {image_file}: {e}")
    
    print(f"Successfully loaded {len(satellite_images)} satellite images")
    return satellite_images

def create_visualization(graph_file, satellite_dir, output_path="visualization.png"):
    """创建graph和卫星图的可视化"""
    
    # 加载数据
    g = load_graph(graph_file)
    satellite_images = load_satellite_images(satellite_dir)
    
    if not satellite_images:
        print("No satellite images loaded!")
        return
    
    # 获取graph边界
    bounds = g.bounds()
    print(f"Graph bounds: {bounds.start} to {bounds.end}")
    
    # 获取图像尺寸（假设所有图像尺寸相同）
    sample_img = next(iter(satellite_images.values()))
    tile_size = sample_img.size[0]  # 假设图像是正方形
    print(f"Tile size: {tile_size}")
    
    # 计算需要的tile范围
    min_x, min_y = bounds.start.x, bounds.start.y
    max_x, max_y = bounds.end.x, bounds.end.y
    
    start_tile_x = min_x // tile_size
    end_tile_x = max_x // tile_size + 1
    start_tile_y = min_y // tile_size
    end_tile_y = max_y // tile_size + 1
    
    print(f"Tile range: X({start_tile_x} to {end_tile_x}), Y({start_tile_y} to {end_tile_y})")
    
    # 创建复合图像
    composite_width = (end_tile_x - start_tile_x) * tile_size
    composite_height = (end_tile_y - start_tile_y) * tile_size
    
    print(f"Creating composite image of size {composite_width}x{composite_height}")
    
    # 创建空白图像
    composite_img = Image.new('RGB', (composite_width, composite_height), (255, 255, 255))
    
    # 拼接卫星图像
    for tile_x in range(start_tile_x, end_tile_x):
        for tile_y in range(start_tile_y, end_tile_y):
            if (tile_x, tile_y) in satellite_images:
                img = satellite_images[(tile_x, tile_y)]
                x_offset = (tile_x - start_tile_x) * tile_size
                y_offset = (tile_y - start_tile_y) * tile_size
                composite_img.paste(img, (x_offset, y_offset))
    
    # 转换为numpy数组用于matplotlib
    composite_array = np.array(composite_img)
    
    # 创建matplotlib图像
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    ax.imshow(composite_array)
    
    # 绘制graph
    for edge in g.edges:
        # 转换坐标到复合图像坐标系
        x1 = edge.src.point.x - start_tile_x * tile_size
        y1 = edge.src.point.y - start_tile_y * tile_size
        x2 = edge.dst.point.x - start_tile_x * tile_size
        y2 = edge.dst.point.y - start_tile_y * tile_size
        
        # 绘制边
        ax.plot([x1, x2], [y1, y2], color='red', linewidth=2, alpha=0.8)
    
    # 绘制顶点
    for vertex in g.vertices:
        x = vertex.point.x - start_tile_x * tile_size
        y = vertex.point.y - start_tile_y * tile_size
        ax.scatter(x, y, color='blue', s=10, alpha=0.8)
    
    ax.set_title('Road Graph Overlay on Satellite Image', fontsize=16)
    ax.axis('off')
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    
    # 显示图像
    plt.show()
    
    plt.close()

def create_graph_only_visualization(graph_file, output_path="graph_only.png"):
    """只显示graph的可视化"""
    
    # 加载graph
    g = load_graph(graph_file)
    
    # 获取边界
    bounds = g.bounds()
    print(f"Graph bounds: {bounds.start} to {bounds.end}")
    
    # 创建图像
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    
    # 绘制graph
    for edge in g.edges:
        ax.plot([edge.src.point.x, edge.dst.point.x], 
               [edge.src.point.y, edge.dst.point.y], 
               'r-', linewidth=1, alpha=0.7)
    
    for vertex in g.vertices:
        ax.scatter(vertex.point.x, vertex.point.y, 
                  color='blue', s=3, alpha=0.8)
    
    ax.set_title('Road Graph Visualization', fontsize=16)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved graph visualization to {output_path}")
    
    # 显示图像
    plt.show()
    
    plt.close()

def analyze_graph(graph_file):
    """分析graph的基本信息"""
    g = load_graph(graph_file)
    
    print("\n=== Graph Analysis ===")
    print(f"Number of vertices: {len(g.vertices)}")
    print(f"Number of edges: {len(g.edges)}")
    
    # 计算边的长度统计
    edge_lengths = []
    for edge in g.edges:
        length = edge.segment().length()
        edge_lengths.append(length)
    
    if edge_lengths:
        print(f"Edge length statistics:")
        print(f"  Min: {min(edge_lengths):.2f}")
        print(f"  Max: {max(edge_lengths):.2f}")
        print(f"  Mean: {np.mean(edge_lengths):.2f}")
        print(f"  Median: {np.median(edge_lengths):.2f}")
    
    # 获取边界
    bounds = g.bounds()
    if bounds:
        print(f"Graph bounds:")
        print(f"  X: {bounds.start.x} to {bounds.end.x}")
        print(f"  Y: {bounds.start.y} to {bounds.end.y}")
        print(f"  Width: {bounds.end.x - bounds.start.x}")
        print(f"  Height: {bounds.end.y - bounds.start.y}")

def main():
    """主函数"""
    # 文件路径
    graph_file = r"D:\AI_learning\road_extraction\out\graph_infer\line_mergec2.infer.graph"
    satellite_dir = r"D:\AI_learning\road_extraction\data\Shaoxing\test_satellite"
    
    print("=== Road Graph and Satellite Image Visualization ===")
    
    # 分析graph
    analyze_graph(graph_file)
    
    # 创建graph-only可视化
    print("\nCreating graph-only visualization...")
    create_graph_only_visualization(graph_file)
    
    # 创建复合可视化
    print("\nCreating composite visualization...")
    create_visualization(graph_file, satellite_dir)

if __name__ == "__main__":
    main() 