# 导入模块
import pandas as pd
import numpy as np
import networkx as nx
import math

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
    # 为路径中的每个节点添加 area 属性
    # path_with_area = [G.nodes[node]['area'] + node for node in path]

    final_cost = dist[end_node]
    final_fare = calculate_fare(total_distance[end_node])
    return path, final_cost, final_fare



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


# 读取边表
df_edges = pd.read_excel("chengdu_merged_stations_line_results2.xlsx", sheet_name="Sheet1", header=0)

# 创建图
G = nx.Graph()

# 添加路径和节点
for index, row in df_edges.iterrows():
    # 如果节点不存在，将会自动创建
    G.add_edge(row["start"], row["end"], length=row["length"], time=row["average_time_seconds"])
    # 为新添加的节点设置属性
    if not G.has_node(row["start"]):
        G.nodes[row["start"]]['name'] = row["start"]
        # 如果有其他属性也需要添加，可以在这里设置
    if not G.has_node(row["end"]):
        G.nodes[row["end"]]['name'] = row["end"]
        # 如果有其他属性也需要添加，可以在这里设置

# for node, data in G.nodes(data=True):
#   print(f"Node: {node}, Data: {data}")

# 修改后的 shortest_path 函数
def shortest_path(G, start_node, end_node, time_weight=1, fare_weight=0):
    print(f"start: {start_node}, end: {end_node}")
    # 调用自定义的 dijkstra 函数
    path, cost, fare = dijkstra(G, start_node, end_node, time_weight, fare_weight)

    # 如果路径存在，则返回路径和成本
    if path:
        return path, cost
    else:
        # 如果路径不存在，返回 None 和无穷大的成本
        return None, INF


# 起点站和终点站的列表
start_stations = [
    "[重庆6号线]北碚", "[重庆10号线]江北机场T2航站楼", "[重庆4号线]头塘",
    "[重庆环线]天星桥", "[重庆2号线]动物园", "[重庆2号线]大渡口",
    "[重庆3号线]学堂湾", "[重庆6号线]邱家湾", "[重庆1号线]璧山"
]

end_stations = [
    "[成都13号线]娇子立交", "[成都1号线]文殊院", "[成都6号线]沙湾",
    "[成都3号线]高升桥", "[成都6号线]新鸿路", "[成都2号线]龙泉驿",
    "[成都3号线]钟楼", "[成都6号线]郫筒", "[成都16号线]温江站",
    "[成都3号线]双流广场", "[成都15号线]华岩", "[成都10号线]儒林路",
    "[成都16号线]蒲江", "[成都16号线]邛崃"
]

station_to_district = {
    "[重庆6号线]北碚": "北碚区",
    "[重庆10号线]江北机场T2航站楼": "渝北区",
    "[重庆4号线]头塘": "江北区",
    "[重庆环线]天星桥": "沙坪坝区",
    "[重庆2号线]动物园": "九龙坡区",
    "[重庆2号线]大渡口": "大渡口区",
    "[重庆3号线]学堂湾": "巴南区",
    "[重庆6号线]邱家湾": "南岸区",
    "[重庆1号线]璧山": "璧山区",
    "[成都13号线]娇子立交": "锦江区",
    "[成都1号线]文殊院": "青羊区",
    "[成都6号线]沙湾": "金牛区",
    "[成都3号线]高升桥": "武侯区",
    "[成都6号线]新鸿路": "成华区",
    "[成都2号线]龙泉驿": "龙泉驿区",
    "[成都3号线]钟楼": "新都区",
    "[成都6号线]郫筒": "郫都区",
    "[成都16号线]温江站": "温江区",
    "[成都3号线]双流广场": "双流区",
    "[成都15号线]华岩": "青白江区",
    "[成都10号线]儒林路": "新津区",
    "[成都16号线]蒲江": "蒲江县",
    "[成都16号线]邛崃": "邛崃市"
}

# 创建一个空的DataFrame，用于存储路径

# 初始化结果DataFrame
results = pd.DataFrame(columns=['start', 'end', 'path', 'cost'])

# 计算每对起点和终点之间的最短路径
for start_name in start_stations:
    for end_name in end_stations:
        if start_name != end_name:
            # 计算路径
            path, cost = shortest_path(G, start_name, end_name, 1,0)

            # 创建一个新的DataFrame来存储当前行的结果
            new_row = pd.DataFrame({'start': [station_to_district[start_name]], 'end': [station_to_district[end_name]], 'path': [path], 'cost': [cost]})
            # 使用concat方法将新行添加到结果DataFrame中
            results = pd.concat([results, new_row], ignore_index=True)

# 将结果保存到新的Excel文件中
results.to_excel('question4_resultst.xlsx', index=False)

# -----------------------------

# 创建一个空的DataFrame，用于存储路径
# 行是起点站名称，列是终点站名称
path_df = pd.DataFrame(index=start_stations, columns=end_stations)


# 计算每对起点和终点之间的最短路径
for start_name in start_stations:
    for end_name in end_stations:
        if start_name != end_name:
            # 这里应该是获取起点和终点的经纬度的逻辑
            # 由于我们直接使用站点名称，所以不需要经纬度
            # 直接调用 shortest_path 函数计算路径
            path, _ = shortest_path(G, start_name, end_name, 1,0)

            # 将路径保存到对应的单元格中
            path_df.at[start_name, end_name] = path

path_df.rename(index=station_to_district, columns=station_to_district, inplace=True)

# 将路径矩阵保存到新的Excel文件中
path_df.to_excel('question4_matrix.xlsx')

print('路径矩阵已保存到 question4_matrix.xlsx 文件中。')