import pandas as pd
import heapq

'''
    Đoạn mã dùng để xây dựng đồ thị và cài đặt các thuật toán 

'''
class Vertex:
    def __init__(self, data):
        self.data = data
        self.adjacent = {}  
        self.distance = float('inf')  
        self.visited = False
        self.previous = None

    def add_neighbor(self, neighbor, weight):
        self.adjacent[neighbor] = weight

class Edge:
    def __init__(self, from_vertex, to_vertex, weight):
        self.from_vertex = from_vertex
        self.to_vertex = to_vertex
        self.weight = weight

class Graph:
    def __init__(self, neo4j_conn=None):
        self.vertices = {}
        self.edges = []
        self.neo4j_conn = neo4j_conn

    def add_vertex(self, data):
        vertex = Vertex(data)
        self.vertices[data] = vertex
        if self.neo4j_conn:
            self.create_vertex(data)
        return vertex

    def add_edge(self, from_data, to_data, weight):
        if from_data in self.vertices and to_data in self.vertices:
            from_vertex = self.vertices[from_data]
            to_vertex = self.vertices[to_data]
            from_vertex.add_neighbor(to_vertex, weight)
            to_vertex.add_neighbor(from_vertex, weight)

            edge = Edge(from_vertex, to_vertex, weight)
            self.edges.append(edge)

            if self.neo4j_conn:
                self.create_edge(from_data, to_data, weight)


    def create_vertex(self, vertex_data):
        query = "CREATE (v:Vertex {data: $data})"
        self.neo4j_conn.query(query, parameters={"data": vertex_data})

    def create_edge(self, from_data, to_data, weight):
        query = """
        MATCH (a:Vertex {data: $from_data}), (b:Vertex {data: $to_data})
        CREATE (a)-[:CONNECTED {weight: $weight}]->(b)
        """
        self.neo4j_conn.query(query, parameters={"from_data": from_data, "to_data": to_data, "weight": weight})

    def save_graph_to_neo4j(self):
        self.neo4j_conn.query("MATCH (n) DETACH DELETE n")
        for vertex_data in self.vertices:
            self.create_vertex(vertex_data)
        for edge in self.edges:
            self.create_edge(edge.from_vertex.data, edge.to_vertex.data, edge.weight)

    ''' 
    -------------------------------------------
        1 . Thuật toán tìm đường đi ngắn nhất
    -------------------------------------------
    '''
    # Thuật toán Dijkstra
    def dijkstra(self, start_data):
        if start_data not in self.vertices:
            return "Start vertex not found"

        start_vertex = self.vertices[start_data]
        start_vertex.distance = 0

        unvisited = list(self.vertices.values())
        log_steps = []

        while unvisited:
            current = min(unvisited, key=lambda vertex: vertex.distance)
            if current.distance == float('inf'):
                break

            current.visited = True
            unvisited.remove(current)

            for neighbor, weight in current.adjacent.items():
                if not neighbor.visited:
                    alt_distance = current.distance + weight
                    if alt_distance < neighbor.distance:
                        neighbor.distance = alt_distance
                        neighbor.previous = current

            log_steps.append({
                'Current Node': current.data,
                'Visited Nodes': [v.data for v in self.vertices.values() if v.visited],
                'Distances': {v.data: v.distance for v in self.vertices.values()}
            })

        return pd.DataFrame(log_steps)  

    def save_shortest_path_to_neo4j(self, path):
        for i in range(len(path) - 1):
            from_vertex = path[i]
            to_vertex = path[i + 1]
            self.neo4j_conn.query("""
            MATCH (a:Vertex {data: $from_data}), (b:Vertex {data: $to_data})
            MERGE (a)-[:SHORTEST_PATH]->(b)
            """, parameters={"from_data": from_vertex, "to_data": to_vertex})

    '''
    ---------------------------------------------    
        2 . Thuật toán tìm chu trình euler
    ---------------------------------------------
    '''
    def _count_connected_components(self):
        """Đếm số thành phần liên thông của đồ thị."""
        visited = set()
        count = 0
        
        def dfs(vertex):
            visited.add(vertex)
            for neighbor in self.vertices[vertex].adjacent:
                if neighbor.data not in visited:
                    dfs(neighbor.data)
        
        for vertex in self.vertices:
            if vertex not in visited:
                dfs(vertex)
                count += 1
        return count
    
    def fleury_algorithm(self):
        """Thuật toán Fleury để tìm chu trình Euler."""
        log_steps = []
    
        def is_bridge(u_data, v_data):
            """Kiểm tra xem cạnh (u, v) có phải cầu không."""
            temp_adjacent = {}
            for vertex in self.vertices:
                temp_adjacent[vertex] = {
                    n.data: w for n, w in self.vertices[vertex].adjacent.items()
                }
            
            temp_adjacent[u_data].pop(v_data, None)
            temp_adjacent[v_data].pop(u_data, None)
            
            visited = set()
            def dfs(vertex):
                visited.add(vertex)
                for neighbor in temp_adjacent[vertex]:
                    if neighbor not in visited:
                        dfs(neighbor)
            
            dfs(u_data)
            return len(visited) != len(self.vertices)
    
        def has_eulerian_circuit():
            """Kiểm tra xem đồ thị có chu trình Euler không."""
            for vertex in self.vertices.values():
                if len(vertex.adjacent) % 2 != 0:
                    return False
            return True
    
        # Kiểm tra điều kiện tồn tại chu trình Euler
        if not has_eulerian_circuit():
            log_steps.append({
                'Step': 'Check Euler Circuit',
                'Current State': 'Failed',
                'Message': 'Đồ thị không có chu trình Euler',
                'Circuit': []
            })
            return pd.DataFrame(log_steps)
    
        # Tạo bản sao của đồ thị
        remaining_edges = {}
        for vertex in self.vertices:
            remaining_edges[vertex] = {
                n.data: w for n, w in self.vertices[vertex].adjacent.items()
            }
    
        circuit = []
        current = next(iter(self.vertices))
    
        # Ghi lại trạng thái ban đầu
        log_steps.append({
            'Step': 'Initial State',
            'Current Vertex': current,
            'Remaining Edges': str({v: list(edges.keys()) for v, edges in remaining_edges.items()}),
            'Circuit': []
        })
    
        while any(remaining_edges[v] for v in remaining_edges):
            # Tìm đỉnh kế tiếp
            next_vertex = None
            for neighbor in list(remaining_edges[current].keys()):
                if not is_bridge(current, neighbor) or len(remaining_edges[current]) == 1:
                    next_vertex = neighbor
                    break
                
            if next_vertex is None and remaining_edges[current]:
                next_vertex = list(remaining_edges[current].keys())[0]
            
            if next_vertex:
                # Thêm cạnh vào chu trình
                circuit.append((current, next_vertex))
                
                # Xóa cạnh đã đi qua
                remaining_edges[current].pop(next_vertex)
                remaining_edges[next_vertex].pop(current)
                
                # Ghi lại bước thực hiện
                log_steps.append({
                    'Step': f'Move {len(circuit)}',
                    'Current Vertex': current,
                    'Next Vertex': next_vertex,
                    'Remaining Edges': str({v: list(edges.keys()) for v, edges in remaining_edges.items()}),
                    'Circuit': str(circuit)
                })
                
                current = next_vertex
            else:
                break
            
        # Ghi lại trạng thái cuối cùng
        log_steps.append({
            'Step': 'Final State',
            'Current Vertex': current,
            'Circuit': str(circuit),
            'Message': 'Completed'
        })
    
        return pd.DataFrame(log_steps)
    
    def remove_edge(self, from_data, to_data):
        """Xóa cạnh từ đồ thị."""
        if from_data in self.vertices and to_data in self.vertices:
            if self.vertices[to_data] in self.vertices[from_data].adjacent:
                del self.vertices[from_data].adjacent[self.vertices[to_data]]
                del self.vertices[to_data].adjacent[self.vertices[from_data]]
                # Xóa cạnh từ danh sách edges nếu cần
                self.edges = [edge for edge in self.edges 
                             if not ((edge.from_vertex.data == from_data and edge.to_vertex.data == to_data) or 
                                    (edge.from_vertex.data == to_data and edge.to_vertex.data == from_data))]
    
    def save_euler_cycle(self, euler_circuit):
        """Lưu chu trình Euler vào Neo4j."""
        if not isinstance(euler_circuit, str) and euler_circuit:  # Kiểm tra không phải là thông báo lỗi
            # Xóa các quan hệ EULER_CIRCUIT cũ
            if self.neo4j_conn:
                self.neo4j_conn.query("MATCH ()-[r:EULER_CIRCUIT]->() DELETE r")
                
                # Tạo các quan hệ mới cho chu trình Euler
                for idx, (from_data, to_data) in enumerate(euler_circuit):
                    self.neo4j_conn.query("""
                    MATCH (a:Vertex {data: $from_data}), (b:Vertex {data: $to_data})
                    CREATE (a)-[:EULER_CIRCUIT {order: $order}]->(b)
                    """, parameters={
                        "from_data": from_data,
                        "to_data": to_data,
                        "order": idx
                    })

    '''
    ----------------------------------------------    
        3 . Thuật toán tìm đường đi hamilton
    ----------------------------------------------       
    '''
    
    def hamiltonian_path(self):
        """Tìm đường đi Hamilton bằng backtracking."""
        log_steps = []
        
        def backtrack(current_vertex, visited, path):
            log_steps.append({
                'Current Vertex': current_vertex,
                'Visited Nodes': list(visited),
                'Current Path': path.copy()
            })
            
            if len(path) == len(self.vertices):
                return path
            
            for neighbor in self.vertices[current_vertex].adjacent:
                if neighbor.data not in visited:
                    visited.add(neighbor.data)
                    path.append(neighbor.data)
                    result = backtrack(neighbor.data, visited, path)
                    if result:
                        return result
                    path.pop()
                    visited.remove(neighbor.data)
            return None

        for start_vertex in self.vertices:
            result = backtrack(start_vertex, {start_vertex}, [start_vertex])
            if result:
                return pd.DataFrame(log_steps)
            
        return "Không tìm thấy đường đi Hamilton."


    '''
    -----------------------------------------------------
        4 . Thuật toán tìm cây khung nhỏ nhất của đồ thị
    -----------------------------------------------------
    '''
    # 4.1. Thuật toán Prim

    def prim_algorithm(self):
        """Thuật toán Prim để tìm cây khung nhỏ nhất."""
        log_steps = []
        
        start_vertex = next(iter(self.vertices.values()))
        start_vertex.distance = 0
        priority_queue = [(0, start_vertex)]
        spanning_tree = []
    
        while priority_queue:
            current_distance, current_vertex = heapq.heappop(priority_queue)
    
            if current_vertex.visited:
                continue
            current_vertex.visited = True
    
            if current_vertex.previous:
                spanning_tree.append((current_vertex.previous.data, current_vertex.data, current_distance))
    
            log_steps.append({
                'Current Vertex': current_vertex.data,
                'Current Distance': current_distance,
                'Visited Nodes': [v.data for v in self.vertices.values() if v.visited],
                'Current Tree': spanning_tree.copy()
            })
    
            for neighbor, weight in current_vertex.adjacent.items():
                if not neighbor.visited and weight < neighbor.distance:
                    neighbor.distance = weight
                    neighbor.previous = current_vertex
                    heapq.heappush(priority_queue, (weight, neighbor))
    
        return pd.DataFrame(log_steps)

    # 4.2. Thuật toán Kruskal

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

    # Hàm chọn thuật toán tìm cây khung nhỏ nhất
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
        
    def save_spanning_tree_to_neo4j(self, spanning_tree):
        """Lưu cây khung nhỏ nhất vào Neo4j."""
        # Xóa các quan hệ SPANNING_TREE cũ
        self.neo4j_conn.query("MATCH ()-[r:SPANNING_TREE]->() DELETE r")
        
        # Tạo các quan hệ mới cho cây khung
        for from_vertex, to_vertex, weight in spanning_tree:
            self.neo4j_conn.query("""
            MATCH (a:Vertex {data: $from_data}), (b:Vertex {data: $to_data})
            CREATE (a)-[:SPANNING_TREE {weight: $weight}]->(b)
            """, parameters={
                "from_data": from_vertex,
                "to_data": to_vertex,
                "weight": weight
            })
    
    '''
    -------------------------------------------
        5. Thuật toán tô màu đồ thị 
    -------------------------------------------
    ''' 
    def graph_coloring(self):
        """Tô màu đồ thị sử dụng thuật toán Greedy."""
        log_steps = []
        color_assignment = {vertex_data: None for vertex_data in self.vertices}
        available_colors = list(range(1, len(self.vertices) + 1))
        
        for vertex_data in self.vertices:
            vertex = self.vertices[vertex_data]
            adjacent_colors = set()
            
            for neighbor, _ in vertex.adjacent.items():
                if color_assignment[neighbor.data] is not None:
                    adjacent_colors.add(color_assignment[neighbor.data])

            for color in available_colors:
                if color not in adjacent_colors:
                    color_assignment[vertex_data] = color
                    break
                
            log_steps.append({
                'Current Vertex': vertex_data,
                'Adjacent Colors': list(adjacent_colors),
                'Assigned Color': color_assignment[vertex_data],
                'Current Coloring': color_assignment.copy()
            })

        return pd.DataFrame(log_steps)
    
    def save_colored_graph_to_neo4j(self, colored_graph):
        """Lưu thông tin tô màu đồ thị vào Neo4j."""
        for vertex_data, color in colored_graph.items():
            self.neo4j_conn.query("""
            MATCH (v:Vertex {data: $vertex_data})
            SET v.color = $color
            """, parameters={
                "vertex_data": vertex_data,
                "color": color
            })