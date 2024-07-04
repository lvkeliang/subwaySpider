from itertools import combinations

import pandas as pd

# 读取 merged_results_co.xlsx 文件
station_df = pd.read_excel("nodes.xlsx")

# 将 'line' 与 'start' 和 'end' 连接
station_df['station'] = "[" + station_df['line'] + "]" + station_df['station']
station_df.drop('line', axis=1, inplace=True)

# 保存修改后的 merged_results_co.xlsx 文件
station_df.to_excel("nodes_stations_line_results.xlsx", index=False)

# --------------------

# 读取 merged_results_co.xlsx 文件
merged_df = pd.read_excel("chengdu_subway.xlsx")

# 将 'line' 与 'start' 和 'end' 连接
#merged_df['start'] = "[" + merged_df['line'] + "]" + merged_df['start']
#merged_df['end'] = "[" + merged_df['line'] + "]" + merged_df['end']
#merged_df.drop('line', axis=1, inplace=True)

# 保存修改后的 merged_results_co.xlsx 文件
# merged_df.to_excel("updated_merged_results_co.xlsx", index=False)

# 读取 interchange_stations.xlsx 文件
interchange_df = pd.read_excel("chengdu_interchange_stations.xlsx")

# 创建一个空的 DataFrame 来存储新的换乘站点数据
interchange_rows = []

# 遍历 interchange_stations.xlsx 中的每一行
for index, row in interchange_df.iterrows():
    station = row['Station']
    lines = row['Interchange Lines'].split(', ')
    # 对于每个换乘站点，创建所有可能的换乘线路组合
    for combo in combinations(lines, 2):
        interchange_rows.append({
            'start': f"[{combo[0]}]{station}",
            'end': f"[{combo[1]}]{station}",
            'length': 0,  # 固定值 0
            'min': "5",  # 固定值 5 分钟
            'second': 300
        })

# 将新的换乘站点数据转换为 DataFrame
interchange_data_df = pd.DataFrame(interchange_rows)

# 将新的换乘站点数据添加到原始数据中
final_df = pd.concat([merged_df, interchange_data_df], ignore_index=True)

output_file_path = "chengdu_merged_stations_line_results.xlsx"
# 保存最终结果到新的 Excel 文件
final_df.to_excel(output_file_path, index=False)
print(f"结果已保存到 {output_file_path}")

