import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import glob
from discoverlib import graph, geom

class GraphSatelliteVisualizer:
    def __init__(self, graph_file, satellite_dir):
        """
        初始化可视化器
        
        Args:
            graph_file: graph文件路径
            satellite_dir: 卫星图目录路径
        """
        self.graph_file = graph_file
        self.satellite_dir = satellite_dir
        self.graph = None
        self.satellite_images = {}
        self.image_size = None
        
    def load_graph(self):
        """加载graph文件"""
        print(f"Loading graph from {self.graph_file}")
        self.graph = graph.read_graph(self.graph_file)
        print(f"Loaded graph with {len(self.graph.vertices)} vertices and {len(self.graph.edges)} edges")
        
    def get_graph_bounds(self):
        """获取graph的边界"""
        if self.graph is None:
            return None
        return self.graph.bounds()
    
    def load_satellite_images(self):
        """加载卫星图像"""
        print(f"Loading satellite images from {self.satellite_dir}")
        
        # 获取所有卫星图像文件
        image_files = glob.glob(os.path.join(self.satellite_dir, "*.png"))
        print(f"Found {len(image_files)} satellite images")
        
        for image_file in image_files:
            try:
                # 从文件名解析坐标信息
                filename = os.path.basename(image_file)
                if filename.startswith('c2_'):
                    # 处理c2格式的文件名
                    parts = filename.replace('c2_', '').replace('_sat.png', '').split('_')
                    if len(parts) == 2:
                        x, y = int(parts[0]), int(parts[1])
                        img = Image.open(image_file)
                        self.satellite_images[(x, y)] = img
                        
                        # 记录图像尺寸
                        if self.image_size is None:
                            self.image_size = img.size
                            
            except Exception as e:
                print(f"Error loading {image_file}: {e}")
                
        print(f"Successfully loaded {len(self.satellite_images)} satellite images")
    
    def create_composite_image(self, output_path=None, show_vertices=True, show_edges=True, 
                             edge_color='red', vertex_color='blue', vertex_size=3, 
                             edge_width=2, alpha=0.8):
        """
        创建graph和卫星图的复合图像
        
        Args:
            output_path: 输出文件路径，如果为None则显示图像
            show_vertices: 是否显示顶点
            show_edges: 是否显示边
            edge_color: 边的颜色
            vertex_color: 顶点的颜色
            vertex_size: 顶点大小
            edge_width: 边宽度
            alpha: 透明度
        """
        if self.graph is None:
            print("Graph not loaded. Please call load_graph() first.")
            return
            
        if not self.satellite_images:
            print("Satellite images not loaded. Please call load_satellite_images() first.")
            return
        
        # 获取graph边界
        bounds = self.get_graph_bounds()
        if bounds is None:
            print("Cannot determine graph bounds.")
            return
            
        print(f"Graph bounds: {bounds.start} to {bounds.end}")
        
        # 计算需要的图像范围
        min_x, min_y = bounds.start.x, bounds.start.y
        max_x, max_y = bounds.end.x, bounds.end.y
        
        # 计算图像网格范围
        tile_size = self.image_size[0] if self.image_size else 1024  # 假设图像是正方形
        
        # 计算需要的tile范围
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
                if (tile_x, tile_y) in self.satellite_images:
                    img = self.satellite_images[(tile_x, tile_y)]
                    x_offset = (tile_x - start_tile_x) * tile_size
                    y_offset = (tile_y - start_tile_y) * tile_size
                    composite_img.paste(img, (x_offset, y_offset))
        
        # 转换为matplotlib图像
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        ax.imshow(composite_img)
        
        # 绘制graph
        if show_edges:
            for edge in self.graph.edges:
                # 转换坐标到复合图像坐标系
                x1 = edge.src.point.x - start_tile_x * tile_size
                y1 = edge.src.point.y - start_tile_y * tile_size
                x2 = edge.dst.point.x - start_tile_x * tile_size
                y2 = edge.dst.point.y - start_tile_y * tile_size
                
                # 绘制边
                ax.plot([x1, x2], [y1, y2], color=edge_color, linewidth=edge_width, alpha=alpha)
        
        if show_vertices:
            for vertex in self.graph.vertices:
                # 转换坐标到复合图像坐标系
                x = vertex.point.x - start_tile_x * tile_size
                y = vertex.point.y - start_tile_y * tile_size
                
                # 绘制顶点
                ax.scatter(x, y, color=vertex_color, s=vertex_size, alpha=alpha)
        
        ax.set_title('Road Graph Overlay on Satellite Image')
        ax.axis('off')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_interactive_visualization(self):
        """创建交互式可视化"""
        if self.graph is None:
            print("Graph not loaded. Please call load_graph() first.")
            return
            
        if not self.satellite_images:
            print("Satellite images not loaded. Please call load_satellite_images() first.")
            return
        
        # 获取graph边界
        bounds = self.get_graph_bounds()
        if bounds is None:
            print("Cannot determine graph bounds.")
            return
        
        # 创建交互式图像
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        
        # 计算图像范围
        min_x, min_y = bounds.start.x, bounds.start.y
        max_x, max_y = bounds.end.x, bounds.end.y
        
        # 设置显示范围
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        
        # 绘制graph
        for edge in self.graph.edges:
            ax.plot([edge.src.point.x, edge.dst.point.x], 
                   [edge.src.point.y, edge.dst.point.y], 
                   'r-', linewidth=1, alpha=0.7)
        
        for vertex in self.graph.vertices:
            ax.scatter(vertex.point.x, vertex.point.y, 
                      color='blue', s=2, alpha=0.8)
        
        ax.set_title('Road Graph Visualization')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        plt.show()
    
    def analyze_graph(self):
        """分析graph的基本信息"""
        if self.graph is None:
            print("Graph not loaded. Please call load_graph() first.")
            return
        
        print("\n=== Graph Analysis ===")
        print(f"Number of vertices: {len(self.graph.vertices)}")
        print(f"Number of edges: {len(self.graph.edges)}")
        
        # 计算边的长度统计
        edge_lengths = []
        for edge in self.graph.edges:
            length = edge.segment().length()
            edge_lengths.append(length)
        
        if edge_lengths:
            print(f"Edge length statistics:")
            print(f"  Min: {min(edge_lengths):.2f}")
            print(f"  Max: {max(edge_lengths):.2f}")
            print(f"  Mean: {np.mean(edge_lengths):.2f}")
            print(f"  Median: {np.median(edge_lengths):.2f}")
        
        # 获取边界
        bounds = self.get_graph_bounds()
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
    
    # 创建可视化器
    visualizer = GraphSatelliteVisualizer(graph_file, satellite_dir)
    
    # 加载数据
    visualizer.load_graph()
    visualizer.load_satellite_images()
    
    # 分析graph
    visualizer.analyze_graph()
    
    # 创建可视化
    print("\nCreating visualization...")
    visualizer.create_composite_image(
        output_path="graph_satellite_overlay.png",
        show_vertices=True,
        show_edges=True,
        edge_color='red',
        vertex_color='blue',
        vertex_size=5,
        edge_width=3,
        alpha=0.8
    )
    
    # 创建交互式可视化
    print("\nCreating interactive visualization...")
    visualizer.create_interactive_visualization()

if __name__ == "__main__":
    main() 