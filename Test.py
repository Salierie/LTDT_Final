# Import necessary classes and methods from Main.py
import pandas as pd
from Main import Graph
from Connect import Neo4jConnection
import networkx as nx
import matplotlib.pyplot as plt

neo4j_conn = Neo4jConnection(
    uri="bolt+ssc://ef6a4b43.databases.neo4j.io",
    user="neo4j",
    pwd="NGc-axA6NfD4lLYYgOS6EuxlNK-Fi8MrV41-bQep68I"
)


if __name__ == "__main__":
    # Tạo đồ thị mẫ
    euler_graph = Graph(neo4j_conn=neo4j_conn)
    for v in ['A', 'B', 'C']:
        euler_graph.add_vertex(v)

    euler_edges = [
        ('A', 'B', 1), ('B', 'C', 1), ('C', 'A', 1),
        ('A', 'B', 2), ('B', 'C', 2), ('C', 'A', 2)
    ]
    for edge in euler_edges:
        euler_graph.add_edge(*edge)

    # Tìm chu trình Euler và lưu các bước
    result_df = euler_graph.fleury_algorithm()
    print(result_df)

    # Lấy chu trình cuối cùng từ DataFrame
    final_circuit = eval(result_df.iloc[-1]['Circuit'])
    # Lưu vào Neo4j
    euler_graph.save_euler_cycle(final_circuit)