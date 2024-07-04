import pandas as pd

# 读取两个 Excel 文件的数据
edges_df = pd.read_excel("edges_new.xlsx")
avg_time_df = pd.read_excel("average_travel_time_results.xlsx")

# 重命名 avg_time_df 中的列，以便与 edges_df 中的列名匹配
avg_time_df.rename(columns={"站点1": "start", "站点2": "end", "平均用时": "average_time"}, inplace=True)

# 使用全外连接合并两个 DataFrame
merged_df = pd.merge(edges_df, avg_time_df, on=["line", "start", "end"], how='outer')

# 将合并后的 DataFrame 保存到新的 Excel 文件中
output_file_path = "merged_results.xlsx"
merged_df.to_excel(output_file_path, index=False)

print(f"合并结果已保存到 {output_file_path}")
