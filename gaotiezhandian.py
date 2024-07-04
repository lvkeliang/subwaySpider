import pandas as pd

# 加载Excel文件
df = pd.read_excel('chengdu_merged_stations_line_results.xlsx')

# 定义要搜索的关键词
keywords = ['重庆北站', '重庆西站', '沙坪坝', '成都东客站', '火车南站']

# 初始化一个空列表来存储结果
stations_dict = {}

# 遍历每个关键词
for keyword in keywords:
    # 在start列中搜索包含关键词的站点，并将结果存储在字典中
    stations_dict[keyword] = df[df['start'].str.contains(keyword)]['start'].tolist()
    # 在end列中搜索包含关键词的站点，并将结果添加到字典中对应的列表
    stations_dict[keyword].extend(df[df['end'].str.contains(keyword)]['end'].tolist())
    # 去重
    stations_dict[keyword] = list(set(stations_dict[keyword]))

# 打印去重后的站点列表
print(stations_dict)

# 您提供的站点列表
#stations_list = ['[7号线]火车南站', '[18号线]火车南站', '成都东客站', '[4号线]重庆北站北广场', '[环线]重庆北站南广场', '[1号线]火车南站', '[7号线]成都东客站', '[5号线]重庆西站', '[环线]重庆西站', '[2号线]成都东客站', '[10号线]重庆北站北广场', '[3号线]重庆北站南广场', '[10号线]重庆北站南广场']

# 定义映射关系
mappings = {
    '重庆北站': '成都东客站',
    '重庆西站': '成都东客站',
    '沙坪坝': '成都东客站',
    '重庆西站': '火车南站',
    '重庆北站': '火车南站'
}

# 对应的映射之间的时间数据
time_data = {
    ('重庆北站', '成都东客站'): {'average_time': 118.7143, 'average_time_seconds': 7122.858},
    ('重庆西站', '成都东客站'): {'average_time': 93.75, 'average_time_seconds': 5625},
    ('沙坪坝', '成都东客站'): {'average_time': 74.5625, 'average_time_seconds': 4473.75},
    ('重庆西站', '火车南站'): {'average_time': 136.5, 'average_time_seconds': 8190},
    ('重庆北站', '火车南站'): {'average_time': 81, 'average_time_seconds': 4860}
}

# 创建新的列表来存储组合后的映射和时间数据
combined_mappings = []

# 遍历新的映射关系
for start_station, end_station in mappings.items():
    # 获取起始站点列表
    start_stations = stations_dict.get(start_station, [])
    # 获取终点站点列表
    end_stations = stations_dict.get(end_station, [])
    # 获取对应的时间数据
    time_info = time_data.get((start_station, end_station), {})
    # 组合起始站点和终点站点，并添加时间数据
    for start in start_stations:
        for end in end_stations:
            combined_mappings.append({
                'start': start,
                'end': end,
                'average_time': time_info.get('average_time', ''),
                'average_time_seconds': time_info.get('average_time_seconds', '')
            })

# 转换为DataFrame
combined_mappings_df = pd.DataFrame(combined_mappings)

# 去重
combined_mappings_df.drop_duplicates(inplace=True)

# 保存到Excel文件
combined_mappings_df.to_excel('combined_mappings.xlsx', index=False)

print('组合后的映射和时间数据已保存到Excel文件中。')