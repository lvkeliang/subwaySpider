import pandas as pd

# 读取Excel文件
df = pd.read_excel("merged_stations_line_results.xlsx")

# 检查 'average_time' 列的数据类型
if df['average_time'].dtype != 'timedelta64[ns]':
    # 如果不是 timedelta 类型，则转换为字符串
    df['average_time'] = df['average_time'].astype(str)

# 确保 'average_time' 列是正确的时间格式字符串
df['average_time'] = pd.to_timedelta(df['average_time'])

# 计算总秒数
df['average_time_seconds'] = df['average_time'].dt.total_seconds()

# 标准化 'length' 和 'average_time_seconds'
length_mean, length_std = df['length'].mean(), df['length'].std()
average_time_mean, average_time_std = df['average_time_seconds'].mean(), df['average_time_seconds'].std()
df['length_standardized'] = (df['length'] - length_mean) / length_std
df['average_time_standardized'] = (df['average_time_seconds'] - average_time_mean) / average_time_std

# 归一化 'length' 和 'average_time_seconds'
df['length_normalized'] = (df['length_standardized'] - df['length_standardized'].min()) / (df['length_standardized'].max() - df['length_standardized'].min())
df['average_time_normalized'] = (df['average_time_standardized'] - df['average_time_standardized'].min()) / (df['average_time_standardized'].max() - df['average_time_standardized'].min())

# 计算 'cost'
df['cost'] = df['length_normalized'] + df['average_time_normalized']

# 保存结果到新的Excel文件
output_file_path = "merged_result_with_cost.xlsx"
df.to_excel(output_file_path, index=False)

print(f"结果已保存到 {output_file_path}")
