import pandas as pd
import math

# 定义Vincenty公式函数
def vincenty(lat1, lon1, lat2, lon2):
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
        lambda_ = L + (1 - C) * f * sinAlpha * (sigma + C * sinSigma * (cos2SigmaM + C * cosSigma * (-1 + 2 * cos2SigmaM ** 2)))

        # 如果变化量小于阈值，则停止迭代
        if abs(lambda_ - lambdaPrev) < 1e-12:
            break

    u2 = cos2Alpha * (a ** 2 - b ** 2) / (b ** 2)
    A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
    deltaSigma = B * sinSigma * (cos2SigmaM + B / 4 * (cosSigma * (-1 + 2 * cos2SigmaM ** 2)) - B / 6 * cos2SigmaM * (-3 + 4 * sinSigma ** 2) * (-3 + 4 * cos2SigmaM ** 2))

    # 计算并返回两点之间的距离
    s = b * A * (sigma - deltaSigma)
    return s / 1000  # 单位为千米

# 初始化一个空的DataFrame用于存放边的信息
columns = ["line", "start", "end", "length"]
df_edges = pd.DataFrame(columns=columns)

# 读取Excel文件中所有的工作表
# xls = pd.ExcelFile("nodes.xlsx")
xls = pd.ExcelFile("中间时间.xlsx")

# 遍历每个工作表（每条线路）
for sheet_name in xls.sheet_names:
    # 读取当前工作表的数据
    df_line = pd.read_excel(xls, sheet_name=sheet_name)
    print(f"正在处理线路：{sheet_name}")
    # 遍历当前线路的每个站点，计算相邻站点之间的距离
    for i in range(len(df_line) - 1):
        start_station, end_station = df_line.iloc[i]["station"], df_line.iloc[i + 1]["station"]
        start_lat, start_lon = df_line.iloc[i]["lat"], df_line.iloc[i]["lon"]
        end_lat, end_lon = df_line.iloc[i + 1]["lat"], df_line.iloc[i + 1]["lon"]
        edge_length = vincenty(start_lat, start_lon, end_lat, end_lon)
        print(f"正在计算：{start_station} 到 {end_station} 的距离")
        print(f"起始站点坐标：({start_lat}, {start_lon})")
        print(f"结束站点坐标：({end_lat}, {end_lon})")
        print(f"距离：({edge_length})")
        # 将结果添加到df_edges DataFrame中
        new_row = pd.DataFrame({
            "line": [sheet_name],
            "start": [start_station],
            "end": [end_station],
            "length": [edge_length]
        })
        df_edges = pd.concat([df_edges, new_row], ignore_index=True)

# 将边的数据保存到一个新的Excel文件中
df_edges.to_excel("chengdu_subway_stations_edges.xlsx", sheet_name="edges", index=False)
