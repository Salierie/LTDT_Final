# Import necessary classes and methods from Main.py
from Main import Graph
import matplotlib.pyplot as plt
import networkx as nx

def visualize_graph(graph, edges, title):
    """Hàm hỗ trợ để vẽ đồ thị."""
    plt.figure(figsize=(8, 8))
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, font_size=15)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))
    plt.title(title)
    plt.show()

def visualize_colored_graph(graph, coloring_result, title):
    """Hàm hỗ trợ để vẽ đồ thị đã tô màu."""
    plt.figure(figsize=(8, 8))
    G = nx.Graph()
    for vertex, neighbors in graph.vertices.items():
        for neighbor, weight in neighbors.adjacent.items():
            G.add_edge(vertex, neighbor.data)
    pos = nx.spring_layout(G)
    color_map = [coloring_result[node] for node in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_color=color_map, cmap=plt.cm.rainbow, node_size=2000, font_size=15)
    plt.title(title)
    plt.show()

def main():
    # **Phần 1: Tìm cây khung nhỏ nhất**
    # Tạo đồ thị phức tạp hơn
    graph = Graph()
    graph.add_vertex("A")
    graph.add_vertex("B")
    graph.add_vertex("C")
    graph.add_vertex("D")
    graph.add_vertex("E")
    graph.add_vertex("F")
    graph.add_edge("A", "B", 1)
    graph.add_edge("A", "C", 2)
    graph.add_edge("B", "C", 3)
    graph.add_edge("B", "D", 4)
    graph.add_edge("C", "D", 5)
    graph.add_edge("C", "E", 6)
    graph.add_edge("D", "F", 7)
    graph.add_edge("E", "F", 8)
    edges = [("A", "B", 1), ("A", "C", 2), ("B", "C", 3), ("B", "D", 4), 
             ("C", "D", 5), ("C", "E", 6), ("D", "F", 7), ("E", "F", 8)]
    
    # Biểu diễn đồ thị ban đầu
    visualize_graph(graph, edges, "Initial Weighted Graph")

    # **1.1 Thuật toán Prim**
    prim_result = graph.prim_algorithm()
    print("\nMinimum Spanning Tree using Prim's Algorithm:")
    print(prim_result)

    # Biểu diễn cây khung bằng Prim
    visualize_graph(graph, prim_result, "Minimum Spanning Tree (Prim's Algorithm)")

    # **1.2 Thuật toán Kruskal**
    kruskal_result = graph.kruskal_algorithm()
    print("\nMinimum Spanning Tree using Kruskal's Algorithm:")
    print(kruskal_result)

    # Biểu diễn cây khung bằng Kruskal
    visualize_graph(graph, kruskal_result, "Minimum Spanning Tree (Kruskal's Algorithm)")

    # **Phần 2: Tô màu đồ thị bản đồ**
    # Tạo đồ thị mô phỏng bản đồ
    map_graph = Graph()
    map_graph.add_vertex("A")
    map_graph.add_vertex("B")
    map_graph.add_vertex("C")
    map_graph.add_vertex("D")
    map_graph.add_vertex("E")
    map_graph.add_vertex("F")

    # Thêm các cạnh (các khu vực tiếp giáp)
    map_graph.add_edge("A", "B", 1)
    map_graph.add_edge("A", "C", 1)
    map_graph.add_edge("A", "D", 1)
    map_graph.add_edge("B", "C", 1)
    map_graph.add_edge("B", "D", 1)
    map_graph.add_edge("B", "E", 1)
    map_graph.add_edge("C", "E", 1)
    map_graph.add_edge("C", "F", 1)
    map_graph.add_edge("D", "E", 1)
    map_graph.add_edge("D", "F", 1)

    # Thực hiện tô màu
    coloring_result = map_graph.graph_coloring()
    print("\nGraph Coloring Result (Map Coloring):")
    print(coloring_result)

    # Biểu diễn đồ thị đã tô màu
    visualize_colored_graph(map_graph, coloring_result, "Map Coloring Result")


if __name__ == "__main__":
    main()
