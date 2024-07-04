# 导入模块
import pandas as pd
import numpy as np
import networkx as nx
import math

# 定义一个无穷大的常量
INF = float("inf")

# 定义一个 Dijkstra 算法函数
def dijkstra(G, start_node, end_node, weight):
    # 获取图中的所有节点
    nodes = list(G.nodes)
    # 初始化一个字典，用于存储每个节点的距离和前驱节点
    dist = {node: (0 if node == start_node else INF) for node in nodes}
    prev = {node: None for node in nodes}
    # 初始化一个集合，用于存储已访问的节点
    visited = set()
    # 循环直到所有节点都被访问或者目标节点被访问
    while len(visited) < len(nodes) and end_node not in visited:
        # 从未访问的节点中选择一个距离最小的节点
        min_node = None
        min_dist = INF
        for node in nodes:
            if node not in visited and dist[node] < min_dist:
                min_node = node
                min_dist = dist[node]
        # 如果没有找到合适的节点，说明无法到达目标节点，退出循环
        if min_node is None:
            break
        # 将该节点加入已访问集合
        visited.add(min_node)
        # 更新该节点的邻接节点的距离和前驱节点
        for neighbor in G.neighbors(min_node):
            if neighbor not in visited:
                new_dist = dist[min_node] + G.edges[min_node, neighbor][weight]
                if new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    prev[neighbor] = min_node
    # 如果目标节点的距离仍然是无穷大，说明无法到达目标节点，返回空列表和无穷大
    if dist[end_node] == INF:
        return [], INF
    # 否则，从目标节点回溯前驱节点，构造最短路径列表和长度
    path = [end_node]
    node = end_node
    while prev[node] is not None:
        node = prev[node]
        path.append(node)
    path.reverse()
    cost = dist[end_node]
    # 返回结果
    return path, cost



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
df_nodes = pd.read_excel("chengdu_subway_stations_with_coordinates.xlsx", sheet_name="Sheet1", header=0)  # 节点信息
df_nodes2 = pd.read_excel("nodes_stations_line_results.xlsx", sheet_name="Sheet1", header=0)
df_edges = pd.read_excel("chengdu_merged_stations_line_results.xlsx", sheet_name="Sheet1", header=0)  # 路径信息

# 创建图s
G = nx.Graph()
# 添加节点
for index, row in df_nodes.iterrows():
    # print(row)
    G.add_node(row["station"], name=row["station"], lat=row["lat"], lon=row["lon"], area="成都")
for index, row in df_nodes2.iterrows():
    # print(row)
    G.add_node(row["station"], name=row["station"], lat=row["lat"], lon=row["lon"], area="重庆")

# 添加路径
for index, row in df_edges.iterrows():
    G.add_edge(row["start"], row["end"], length=row["length"], time=row["average_time_seconds"])


# for node, data in G.nodes(data=True):
#   print(f"Node: {node}, Data: {data}")

# 定义寻找最短路径的函数
def shortest_path(start_lat, start_lon, end_lat, end_lon):
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

    path, length = dijkstra(G, start_node, end_node, weight="length")
    # 返回结果
    return path, length


# 读取Excel文件
# end
df = pd.read_excel('chengduuniversity.xlsx')
# start
dfState = pd.read_excel('university.xlsx')

# 初始化结果DataFrame
results = pd.DataFrame(columns=['start', 'end', 'path', 'cost'])

# 计算每对起点和终点之间的最短路径和成本
for i, start_row in df.iterrows():
    for j, end_row in df.iterrows():
        if i != j:  # 确保起点和终点不是同一个地点
            start_name = start_row['location']
            end_name = end_row['location']
            path, cost = shortest_path(start_row['lat'], start_row['lon'], end_row['lat'], end_row['lon'])
            # 创建一个新的DataFrame来存储当前行的结果
            new_row = pd.DataFrame({'start': [start_name], 'end': [end_name], 'path': [path], 'cost': [cost]})
            # 使用concat方法将新行添加到结果DataFrame中
            results = pd.concat([results, new_row], ignore_index=True)

# 将结果保存到新的Excel文件中
results.to_excel('question3_minlength_resultst.xlsx', index=False)

# -----------------------------

# 读取Excel文件
df = pd.read_excel('chengduuniversity.xlsx')
dfState = pd.read_excel('university.xlsx')

# 创建一个空的DataFrame，用于存储路径
# 行和列都是学校名称
path_df = pd.DataFrame(index=dfState['location'], columns=df['location'])

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
            path, _ = shortest_path(start_lat, start_lon, end_lat, end_lon)

            # 将路径保存到对应的单元格中
            path_df.at[start_name, end_name] = path

# 将路径矩阵保存到新的Excel文件中
path_df.to_excel('question3_minlength_matrix.xlsx')

print('路径矩阵已保存到 question3_minlength_matrix.xlsx 文件中。')