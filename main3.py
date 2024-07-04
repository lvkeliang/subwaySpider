import pandas as pd
from datetime import datetime, timedelta


def calculate_and_save_average_time(excel_file_path, outputpath):
    # 读取Excel文件
    df = pd.read_excel(excel_file_path)

    # 初始化一个空列表来存储每一行的平均用时
    average_times = []

    # 遍历每一行
    for index, row in df.iterrows():
        # 获取该行的时间数据（忽略空值"--"）
        times = [time for time in row[1:] if time != "--"]

        # 如果有有效的时间数据，计算平均用时
        if times:
            # 将时间转换为timedelta对象
            time_deltas = [timedelta(hours=t.hour, minutes=t.minute) for t in
                           [datetime.strptime(time, "%H:%M") for time in times]]
            # 计算所有timedelta的总和
            total_delta = sum(time_deltas, timedelta())
            # 计算平均时间
            average_delta = total_delta / len(time_deltas)
            # 将平均时间转换回小时和分钟
            average_time = (datetime.min + average_delta).time().strftime("%H:%M")
            average_times.append(average_time)
        else:
            average_times.append("--")

    # 将平均用时添加到新的一列
    df["平均用时"] = average_times

    # 保存更新后的DataFrame回Excel文件
    df.to_excel(outputpath, index=False)


# 示例用法
excel_file_path = "latesttime.xlsx"
outputpath = excel_file_path[:-5] + "ave.xlsx"
calculate_and_save_average_time(excel_file_path, outputpath)

print(f"计算结果已保存到 {outputpath}")
