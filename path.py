# 导入模块
import pandas as pd
import numpy as np
import networkx as nx
import math
import scipy as sp

# 定义一个无穷大的常量
INF = float("inf")

# 定义票价计算函数
def calculate_fare(total_distance):
    # 根据距离计算票价
    fare_brackets = [(6, 2), (11, 3), (17, 4), (24, 5), (32, 6), (41, 7), (51, 8), (63, 9)]
    for (distance, fare) in fare_brackets:
        if total_distance <= distance:
            return fare
    return 10

# 修改后的 Dijkstra 算法
def dijkstra(G, start_node, end_node, time_weight, fare_weight):
    nodes = list(G.nodes)
    dist = {node: (0 if node == start_node else INF) for node in nodes}
    prev = {node: None for node in nodes}
    total_distance = {node: 0 for node in nodes}  # 新增总距离字典
    visited = set()

    while len(visited) < len(nodes) and end_node not in visited:
        min_node = None
        min_cost = INF
        for node in nodes:
            if node not in visited and dist[node] < min_cost:
                min_node = node
                min_cost = dist[node]
        if min_node is None:
            break
        visited.add(min_node)
        for neighbor in G.neighbors(min_node):
            if neighbor not in visited:
                edge_distance = G.edges[min_node, neighbor]['length']
                edge_time = G.edges[min_node, neighbor]['time']
                new_total_distance = total_distance[min_node] + edge_distance
                fare = calculate_fare(new_total_distance)
                new_cost = dist[min_node] + time_weight * edge_time + fare_weight * fare
                if new_cost < dist[neighbor]:
                    dist[neighbor] = new_cost
                    prev[neighbor] = min_node
                    total_distance[neighbor] = new_total_distance  # 更新总距离
    if dist[end_node] == INF:
        return [], INF, 0

    path = [end_node]
    node = end_node
    while prev[node] is not None:
        node = prev[node]
        path.append(node)
    path.reverse()
    final_cost = dist[end_node]
    final_fare = calculate_fare(total_distance[end_node])  # 根据总距离计算最终票价
    return path, final_cost, final_fare, total_distance[end_node]


# 定义Vincenty公式函数
def vincenty(lat1, lon1, lat2, lon2):
    if lat1 == lat2 and lon1 == lon2:
        return 0

    # WGS-84椭球体参数
    a = 6378137.0  # 赤道半径
    f = 1 / 298.257223563  # 扁率
    b = (1 - f) * a

    # 将经纬度转换为弧度
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    lambda1 = math.radians(lon1)
    lambda2 = math.radians(lon2)

    # 计算两点之间的经度差
    L = lambda2 - lambda1

    # 迭代计算
    U1 = math.atan((1 - f) * math.tan(phi1))
    U2 = math.atan((1 - f) * math.tan(phi2))
    sinU1 = math.sin(U1)
    cosU1 = math.cos(U1)
    sinU2 = math.sin(U2)
    cosU2 = math.cos(U2)

    lambda_ = L
    for _ in range(100):
        sinLambda = math.sin(lambda_)
        cosLambda = math.cos(lambda_)
        sinSigma = math.sqrt((cosU2 * sinLambda) ** 2 + (cosU1 * sinU2 - sinU1 * cosU2 * cosLambda) ** 2)
        cosSigma = sinU1 * sinU2 + cosU1 * cosU2 * cosLambda
        sigma = math.atan2(sinSigma, cosSigma)
        sinAlpha = cosU1 * cosU2 * sinLambda / sinSigma
        cos2Alpha = 1 - sinAlpha ** 2
        cos2SigmaM = cosSigma - 2 * sinU1 * sinU2 / cos2Alpha
        C = f / 16 * cos2Alpha * (4 + f * (4 - 3 * cos2Alpha))
        lambdaPrev = lambda_
        lambda_ = L + (1 - C) * f * sinAlpha * (
                    sigma + C * sinSigma * (cos2SigmaM + C * cosSigma * (-1 + 2 * cos2SigmaM ** 2)))

        # 如果变化量小于阈值，则停止迭代
        if abs(lambda_ - lambdaPrev) < 1e-12:
            break

    u2 = cos2Alpha * (a ** 2 - b ** 2) / (b ** 2)
    A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
    deltaSigma = B * sinSigma * (cos2SigmaM + B / 4 * (cosSigma * (-1 + 2 * cos2SigmaM ** 2)) - B / 6 * cos2SigmaM * (
                -3 + 4 * sinSigma ** 2) * (-3 + 4 * cos2SigmaM ** 2))

    # 计算并返回两点之间的距离
    s = b * A * (sigma - deltaSigma)
    return s / 1000  # 单位为千米


# 读取 excel 文件
df_nodes = pd.read_excel("nodes_stations_line_results.xlsx", sheet_name="Sheet1", header=0)  # 节点信息
df_edges = pd.read_excel("merged_result_with_cost.xlsx", sheet_name="Sheet1", header=0)  # 路径信息

# 创建图s
G = nx.Graph()
# 添加节点
for index, row in df_nodes.iterrows():
    # print(row)
    G.add_node(row["station"], name=row["station"], lat=row["lat"], lon=row["lon"])
# 添加路径
for index, row in df_edges.iterrows():
    G.add_edge(row["start"], row["end"], length=row["length"], time=row["average_time_seconds"], cost=row["cost"])


# for node, data in G.nodes(data=True):
#   print(f"Node: {node}, Data: {data}")


# 定义寻找最短路径的函数
def shortest_path(start_lat, start_lon, end_lat, end_lon, time_weight, fare_weight):
    start_dists = [vincenty(start_lat, start_lon, node[1]["lat"], node[1]["lon"]) for node in G.nodes(data=True)]
    start_node = list(G.nodes)[np.argmin(start_dists)]
    print(f'start_lat: {start_lat}, start_lon: {start_lon}')
    print("start: ", start_node)
    # 找到距离目标点最近的节点
    end_dists = [vincenty(end_lat, end_lon, node[1]["lat"], node[1]["lon"]) for node in G.nodes(data=True)]
    end_node = list(G.nodes)[np.argmin(end_dists)]
    print(f'end_lat: {end_lat}, start_lon: {end_lon}')
    print("end: ", end_node)
    # 计算最短路径
    # path = nx.dijkstra_path (G, start_node, end_node, weight = "length")
    # 计算最短路径长度
    # length = nx.dijkstra_path_length (G, start_node, end_node, weight = "length")

    # 使用新的dijkstra函数
    path, cost, fare, distance= dijkstra(G, start_node, end_node, time_weight, fare_weight)
    return path, cost, fare, distance
    # 返回结果
    # return path, length


# 测试
# start_lat = 106.61231 # 出发点纬度
# start_lon = 29.541387 # 出发点经度
# end_lat = 106.618203 # 目标点纬度
# end_lon = 29.539298 # 目标点经度

# start_lat = 106.61231 # 出发点纬度
# start_lon = 29.541387 # 出发点经度
# end_lat = 106.613321 # 目标点纬度
# end_lon = 29.536115 # 目标点经度



# 重庆大学
end_lat, end_lon = 29.569027,106.462187

# 西南大学
end_lat, end_lon = 29.820659,106.423923

# 西南政法大学(渝北校区)
end_lat, end_lon = 29.663923, 106.593213

# 重庆交通大学(科学城校区)
end_lat, end_lon = 29.420404,106.316285

# 重庆邮电大学
start_lat,start_lon = 29.532326,106.60796

# 重庆医科大学袁家岗校区
start_lat,start_lon = 29.533087,106.508289

# 重庆师范大学(大学城校区)
start_lat,start_lon = 29.612266,106.3015

# 重庆工商大学(南岸校区)
start_lat,start_lon = 29.505781,106.580243

# 四川外国语大学
start_lat,start_lon = 29.569908,106.436765

# 四川美术学院(大学城校区)
start_lat,start_lon = 29.602056,106.298647

# 重庆理工大学(花溪校区)
start_lat,start_lon = 29.452786,106.53017

# 重庆科技大学
start_lat,start_lon = 29.59313,106.324247

# 上新街站
start_lat = 29.556089
start_lon = 106.597201


# 重庆理工大学(花溪校区)
# end_lat, end_lon = 29.454875, 106.525322



#path, cost = shortest_path(start_lat, start_lon, end_lat, end_lon)
#print("The best path is:", path)
#print("The cost of the path is: ", cost)


# 读取Excel文件
df = pd.read_excel('university.xlsx')

# 初始化结果DataFrame
results = pd.DataFrame(columns=['start', 'end', 'path', 'cost'])

# 计算每对起点和终点之间的最短路径和成本
for i, start_row in df.iterrows():
    for j, end_row in df.iterrows():
        if i != j:  # 确保起点和终点不是同一个地点
            start_name = start_row['location']
            end_name = end_row['location']
            path, cost, fare, distance = shortest_path(start_row['lat'], start_row['lon'], end_row['lat'], end_row['lon'], time_weight=1, fare_weight=1)
            # 创建一个新的DataFrame来存储当前行的结果
            new_row = pd.DataFrame({'start': [start_name], 'end': [end_name], 'path': [path], 'distance': [distance], 'fare': [fare], 'cost': [cost]})
            # 使用concat方法将新行添加到结果DataFrame中
            results = pd.concat([results, new_row], ignore_index=True)

# 将结果保存到新的Excel文件中
results.to_excel('question1_results.xlsx', index=False)

# -----------------------------

# 读取Excel文件
df = pd.read_excel('university.xlsx')

# 创建一个空的DataFrame，用于存储路径
# 行和列都是学校名称
path_df = pd.DataFrame(index=df['location'], columns=df['location'])

# 计算每对起点和终点之间的最短路径
for start_name in df['location']:
    for end_name in df['location']:
        if start_name != end_name:
            # 获取起点和终点的经纬度
            start_lat = df.loc[df['location'] == start_name, 'lat'].values[0]
            start_lon = df.loc[df['location'] == start_name, 'lon'].values[0]
            end_lat = df.loc[df['location'] == end_name, 'lat'].values[0]
            end_lon = df.loc[df['location'] == end_name, 'lon'].values[0]

            # 计算路径
            path, _ , _, _ = shortest_path(start_lat, start_lon, end_lat, end_lon, 1, 1)

            # 将路径保存到对应的单元格中
            path_df.at[start_name, end_name] = path

# 将路径矩阵保存到新的Excel文件中
path_df.to_excel('question1_matrix.xlsx')

print('路径矩阵已保存到 question1_matrix.xlsx 文件中。')

# 生成邻接矩阵
adj_matrix = nx.adjacency_matrix(G).toarray()

# 获取图中的节点列表，用作行名和列名
nodes_list = list(G.nodes())

# 创建一个 DataFrame 来表示邻接矩阵
df_adj_matrix = pd.DataFrame(adj_matrix, index=nodes_list, columns=nodes_list)

from prettyprinter import pprint

# 假设您已经有了一个名为 df_adj_matrix 的 DataFrame，它包含了邻接矩阵的数据
# 截取左上角的6x6部分
sub_df = df_adj_matrix.iloc[:6, :6]

# 使用 prettyprinter 的 pprint 函数打印
pprint(sub_df)
