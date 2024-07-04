import networkx as nx
import pandas as pd
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary

# 创建图
G = nx.DiGraph()

# 读取 excel 文件
df_nodes = pd.read_excel("nodes_stations_line_results.xlsx", sheet_name="Sheet1", header=0)  # 节点信息
df_edges = pd.read_excel("merged_result_with_cost.xlsx", sheet_name="Sheet1", header=0)  # 路径信息

# 添加节点和路径到图
for index, row in df_nodes.iterrows():
    G.add_node(row["station"], name=row["station"])
for index, row in df_edges.iterrows():
    G.add_edge(row["start"], row["end"], length=row["length"], time=row["average_time_seconds"])

# 定义票价计算函数
def calculate_fare(total_distance):
    fare_brackets = [(6, 2), (11, 3), (17, 4), (24, 5), (32, 6), (41, 7), (51, 8), (63, 9)]
    for (distance, fare) in fare_brackets:
        if total_distance <= distance:
            return fare
    return 10

# 选择起始站和终点站
start_station = "[环线]上新街"
end_station = "[1号线]大学城"

# 创建优化问题
prob = LpProblem("SubwayOptimization", LpMinimize)

# 创建决策变量
x = LpVariable.dicts("x", [(i, j) for i in G.nodes() for j in G.nodes() if i != j], 0, 1, LpBinary)

# 目标函数
prob += lpSum([x[(i, j)] * (G.edges[i, j]['time'] + calculate_fare(G.edges[i, j]['length'])) for i, j in G.edges()])

# 约束条件
for node in G.nodes():
    # 离开每个站点的约束
    prob += lpSum([x[(node, j)] for j in G.neighbors(node)]) == (1 if node == start_station else 0)
    # 到达每个站点的约束
    prob += lpSum([x[(i, node)] for i in G.predecessors(node)]) == (1 if node == end_station else 0)

# 解决问题
prob.solve()

# 输出结果
for v in prob.variables():
    if v.varValue > 0:
        print(v.name, "=", v.varValue)
