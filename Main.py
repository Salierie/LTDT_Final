import pandas as pd
from neo4j import GraphDatabase

import heapq

class Neo4jConnection:
    def __init__(self, uri, user, pwd):
        self.uri = uri
        self.user = user
        self.pwd = pwd
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.pwd))
        self.session = self.driver.session()

    def close(self):
        self.session.close()

    def query(self, query, parameters=None):
        result = self.session.run(query, parameters or {})
        return result

class Vertex:
    def __init__(self, data):
        self.data = data
        self.adjacent = {}  
        self.distance = float('inf')  
        self.visited = False
        self.previous = None

    def add_neighbor(self, neighbor, weight):
        self.adjacent[neighbor] = weight

class Graph:
    def __init__(self, neo4j_conn=None):
        self.vertices = {}
        self.neo4j_conn = neo4j_conn  # Kết nối Neo4j

    def add_vertex(self, data):
        vertex = Vertex(data)
        self.vertices[data] = vertex
        
        # Lưu đỉnh vào Neo4j
        if self.neo4j_conn:
            self.create_vertex(self.neo4j_conn, data)
        
        return vertex

    def add_edge(self, from_data, to_data, weight):
        if from_data in self.vertices and to_data in self.vertices:
            self.vertices[from_data].add_neighbor(self.vertices[to_data], weight)
            self.vertices[to_data].add_neighbor(self.vertices[from_data], weight)

            # Lưu cạnh vào Neo4j
            if self.neo4j_conn:
                self.create_edge(self.neo4j_conn, from_data, to_data, weight)

    def create_vertex(self, neo4j_conn, vertex_data):
        query = "CREATE (v:Vertex {data: $data})"
        neo4j_conn.query(query, parameters={"data": vertex_data})

    def create_edge(self, neo4j_conn, from_data, to_data, weight):
        query = """
        MATCH (a:Vertex {data: $from_data}), (b:Vertex {data: $to_data})
        CREATE (a)-[:CONNECTED {weight: $weight}]->(b)
        """
        neo4j_conn.query(query, parameters={"from_data": from_data, "to_data": to_data, "weight": weight})

    def save_graph_to_neo4j(self):
        # Xóa tất cả các node và relationship cũ trước khi lưu
        self.neo4j_conn.query("MATCH (n) DETACH DELETE n")

        # Lưu các đỉnh
        for vertex_data in self.vertices:
            self.create_vertex(self.neo4j_conn, vertex_data)

        # Lưu các cạnh
        for vertex_data in self.vertices:
            vertex = self.vertices[vertex_data]
            for neighbor, weight in vertex.adjacent.items():
                self.create_edge(self.neo4j_conn, vertex_data, neighbor.data, weight)


    def prim_algorithm(self):
        """Thuật toán Prim để tìm cây khung nhỏ nhất."""
        start_vertex = next(iter(self.vertices.values()))  # Lấy một đỉnh bất kỳ
        start_vertex.distance = 0
        priority_queue = [(0, start_vertex)]
        spanning_tree = []

        while priority_queue:
            current_distance, current_vertex = heapq.heappop(priority_queue)

            if current_vertex.visited:
                continue
            current_vertex.visited = True

            for neighbor, weight in current_vertex.adjacent.items():
                if not neighbor.visited and weight < neighbor.distance:
                    neighbor.distance = weight
                    neighbor.previous = current_vertex
                    heapq.heappush(priority_queue, (weight, neighbor))

            if current_vertex.previous:
                spanning_tree.append((current_vertex.previous.data, current_vertex.data, current_distance))

        return spanning_tree

    def kruskal_algorithm(self):
        """Thuật toán Kruskal để tìm cây khung nhỏ nhất."""
        edges = []
        for vertex_data, vertex in self.vertices.items():
            for neighbor, weight in vertex.adjacent.items():
                if (neighbor.data, vertex_data, weight) not in edges:  # Tránh thêm cạnh hai lần
                    edges.append((vertex_data, neighbor.data, weight))

        # Sắp xếp các cạnh theo trọng số
        edges.sort(key=lambda x: x[2])

        parent = {}
        rank = {}

        def find(vertex):
            if parent[vertex] != vertex:
                parent[vertex] = find(parent[vertex])
            return parent[vertex]

        def union(vertex1, vertex2):
            root1 = find(vertex1)
            root2 = find(vertex2)

            if root1 != root2:
                if rank[root1] > rank[root2]:
                    parent[root2] = root1
                elif rank[root1] < rank[root2]:
                    parent[root1] = root2
                else:
                    parent[root2] = root1
                    rank[root1] += 1

        for vertex_data in self.vertices:
            parent[vertex_data] = vertex_data
            rank[vertex_data] = 0

        spanning_tree = []
        for from_data, to_data, weight in edges:
            if find(from_data) != find(to_data):
                union(from_data, to_data)
                spanning_tree.append((from_data, to_data, weight))

        return spanning_tree

    def find_spanning_tree(self, algorithm="prim"):
        """
        Hàm cho phép người dùng chọn 1 trong 2 thuật toán tìm cây khung trên để tìm cây khung nhỏ nhất.
        :param algorithm: "prim" hoặc "kruskal"
        """
        if algorithm == "prim":
            return self.prim_algorithm()
        elif algorithm == "kruskal":
            return self.kruskal_algorithm()
        else:
            raise ValueError("Thuật toán không hợp lệ. Vui lòng chọn 'prim' hoặc 'kruskal'.")
        
    
    def graph_coloring(self):
        """Tô màu đồ thị sử dụng thuật toán Greedy (tham lam)."""
        # Khởi tạo màu cho tất cả các đỉnh là None
        color_assignment = {vertex_data: None for vertex_data in self.vertices}
        
        # Danh sách các màu có thể sử dụng
        available_colors = list(range(1, len(self.vertices) + 1))  # Duy trì số màu tối đa tương ứng với số đỉnh
        
        for vertex_data in self.vertices:
            vertex = self.vertices[vertex_data]
            # Tìm các màu của các đỉnh kề với vertex
            adjacent_colors = set()
            for neighbor, _ in vertex.adjacent.items():
                if color_assignment[neighbor.data] is not None:
                    adjacent_colors.add(color_assignment[neighbor.data])

            # Chọn màu cho vertex, màu chưa bị sử dụng bởi các đỉnh kề
            for color in available_colors:
                if color not in adjacent_colors:
                    color_assignment[vertex_data] = color
                    break

        return color_assignment    