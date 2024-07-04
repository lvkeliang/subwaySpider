import os

import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import quote, urlencode
import json

# 目标网址
url = 'https://www.cqmetro.cn/index.shtml'

jsonp = 'jsonp_1fb782989268e40'

# 发送请求获取网页内容
response = requests.get(url)
response.encoding = 'utf-8'

# 解析网页
soup = BeautifulSoup(response.text, 'html.parser')

# 创建一个字典来存储每条线路及其站点
subway_lines = {}

# 找到包含所有线路名称的 HTML 部分
lines_names = soup.find_all('div', class_='linebox')[0].find_all('a', class_='li')

# 找到包含所有站点信息的 HTML 部分
stations_boxes = soup.find_all('div', class_='stationbox')[0:len(lines_names)]  # 取每两个中的第一个

# 遍历每条线路
for line, stations_box in zip(lines_names, stations_boxes):
    # 获取线路名称
    line_name = line.text.strip()
    # 初始化线路站点列表
    subway_lines[line_name] = []

    # 遍历该线路下的所有站点
    stations = stations_box.find_all('a')
    for station in stations:
        # 获取站点名称
        station_name = station.text.strip()
        # 将站点添加到线路列表中
        subway_lines[line_name].append(station_name)

# 准备存储数据的 DataFrame
df_subway = pd.DataFrame(columns=['line', 'station', 'lat', 'lon'])

# 用于请求经纬度数据的基础 URL
# base_url = 'https://h5gw.map.qq.com/ws/place/v1/search'
base_url = 'https://h5gw.map.qq.com/ws/place/v1/suggestion'

for line, stations in subway_lines.items():
    print(line, ': ', stations)

# 遍历每个站点，获取经纬度信息
for line, stations in subway_lines.items():
    for station in stations:
        try:
            # 构造请求参数
            params = {
                'region': '重庆市',
                'region_fix': 1,
                'keyword': f'{station}[地铁站]',
                'apptag': 'lbsplace_sug',
                'key': '[你的key]',  # 替换为您的key
                'output': 'jsonp',
                'callback': 'jsonp'
            }

            # 对参数进行URL编码
            encoded_params = urlencode(params)

            # 构造完整的URL
            full_url = f'{base_url}?{encoded_params}'

            # 打印请求路径
            # print(full_url)
            print(line, ': ', station)

            # 发送请求
            response = requests.get(full_url)

            # 解析响应内容
            if response.status_code == 200:
                # 去除JSONP格式的前缀和后缀
                json_str = response.text.split('&&')[1].strip('jsonp();')
                data = json.loads(json_str)
                # 如果成功获取数据，则添加到DataFrame
                if data['status'] == 0:
                    # 提取经纬度
                    lat = data['data'][0]['location']['lat']
                    lon = data['data'][0]['location']['lng']
                else:
                    lat = None
                    lon = None
            else:
                lat = None
                lon = None

            # 创建一个新的DataFrame来存储当前行的数据
            new_row = pd.DataFrame({'line': [line], 'station': [station], 'lat': [lat], 'lon': [lon]})
            # 使用concat方法添加到df_subway
            df_subway = pd.concat([df_subway, new_row], ignore_index=True)

        except Exception as e:
            print(f"Error fetching data for station: {station}, error: {e}")
            # 创建一个新的DataFrame来存储当前行的数据，经纬度为空
            new_row = pd.DataFrame({'line': [line], 'station': [station], 'lat': [None], 'lon': [None]})
            # 使用concat方法添加到df_subway
            df_subway = pd.concat([df_subway, new_row], ignore_index=True)
            continue

# 将数据保存到 Excel 文件
df_subway.to_excel('subway_stations.xlsx', index=False)
print('saved to: ', os.getcwd(), 'subway_stations.xlsx')
