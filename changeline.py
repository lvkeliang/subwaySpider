import pandas as pd

# 读取Excel文件
df = pd.read_excel('chengdu_subway_stations_with_coordinates.xlsx')

# 创建一个字典来存储站点和线路的对应关系
station_lines = {}

# 遍历DataFrame，填充字典
for index, row in df.iterrows():
    # 从station列读取站点信息
    full_station_info = row['station']

    # 从[]内读取线路
    line = full_station_info[full_station_info.find("[") + 1:full_station_info.find("]")]

    # 从[]后面读取站点名称
    station = full_station_info[full_station_info.find("]") + 1:].strip()

    print(line, station)

    if station not in station_lines:
        station_lines[station] = set()
    station_lines[station].add(line)

# 找出换乘站点及其对应的线路
interchange_stations = {station: lines for station, lines in station_lines.items() if len(lines) > 1}

# 创建一个新的DataFrame来存储换乘站点及其对应的线路
interchange_df = pd.DataFrame([(station, ', '.join(sorted(lines))) for station, lines in interchange_stations.items()], columns=['Station', 'Interchange Lines'])

# 保存结果到新的Excel文件
interchange_df.to_excel('chengdu_interchange_stations.xlsx', index=False)
