import pandas as pd
from datetime import datetime, timedelta


def calculate_average_travel_time_between_stations(df):
    # Initialize an empty DataFrame to store results
    result_df = pd.DataFrame(columns=["站点1", "站点2", "平均用时"])

    # Iterate through adjacent station pairs
    for i in range(len(df) - 1):
        station1 = df.loc[i, "站点"]
        station2 = df.loc[i + 1, "站点"]

        # Initialize total time and valid pairs count
        total_time = timedelta()
        valid_pairs_count = 0

        print(f'{station1} - {station2} : ')
        # 假设 df 是您的 DataFrame，i 是当前行的索引
        for col in df.columns[1:]:
            time1, time2 = df.loc[i, col], df.loc[i + 1, col]
            if time1 != "--" and time2 != "--":
                time1_obj = datetime.strptime(time1, "%H:%M")
                time2_obj = datetime.strptime(time2, "%H:%M")
                # 检查是否跨天
                if time1_obj.hour == 23 and time2_obj.hour == 0:
                    # midnight = datetime.strptime("00:00", "%H:%M")

                    time1_obj -= timedelta(days=1)
                    # time_diff = (midnight - time1_obj).total_seconds() + (time2_obj - midnight).total_seconds()
                elif time1_obj.hour == 0 and time2_obj.hour == 23:
                    # midnight = datetime.strptime("00:00", "%H:%M")

                    time2_obj -= timedelta(days=1)
                    # time_diff = (time1_obj - midnight).total_seconds() + (midnight - time2_obj).total_seconds()

                time_diff = abs((time2_obj - time1_obj).total_seconds())

                # 将秒转换回 timedelta 对象
                print(time_diff)
                total_time += pd.to_timedelta(abs(time_diff), unit='s')
                valid_pairs_count += 1

        # Calculate average time
        if valid_pairs_count > 0:
            average_time = total_time / valid_pairs_count
            print(f'ave : {average_time}')
            # Create a new DataFrame for the current row and concatenate it with the result DataFrame
            new_row = pd.DataFrame([[station1, station2, average_time]], columns=["站点1", "站点2", "平均用时"])
            result_df = pd.concat([result_df, new_row], ignore_index=True)

    return result_df


# Read your original Excel file
input_file_path = "latesttime.xlsx"
# 初始化一个空的 DataFrame 用于存储所有结果
all_lines_result_df = pd.DataFrame()

# 加载 Excel 文件，遍历所有的 sheet
xls = pd.ExcelFile(input_file_path)
for sheet_name in xls.sheet_names:
    # 读取当前 sheet 的 DataFrame
    df = pd.read_excel(xls, sheet_name=sheet_name)
    # 计算当前线路的平均旅行时间
    result_df = calculate_average_travel_time_between_stations(df)
    # 添加线路名称作为新的列
    result_df['line'] = sheet_name
    # 将结果添加到总的 DataFrame 中
    all_lines_result_df = pd.concat([all_lines_result_df, result_df], ignore_index=True)

# 将 'line' 列移动到第一列
cols = ['line'] + [col for col in all_lines_result_df if col != 'line']
all_lines_result_df = all_lines_result_df[cols]

# 重命名列以匹配所需的格式
all_lines_result_df.rename(columns={'站点1': '站点1', '站点2': '站点2', '平均用时': '平均用时'}, inplace=True)

# 保存结果到新的 Excel 文件
output_file_path = "average_travel_time_results.xlsx"
all_lines_result_df.to_excel(output_file_path, index=False)

print(f"Results saved to {output_file_path}")

