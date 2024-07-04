import pandas as pd
from urllib.parse import urlencode
import requests
import json

# 读取Excel文件
df_stations = pd.read_excel('chengdu_subway_stations2.xlsx')

# 准备存储数据的 DataFrame
df_subway = pd.DataFrame(columns=['station', 'lat', 'lon'])

# 用于请求经纬度数据的基础 URL
base_url = 'https://h5gw.map.qq.com/ws/place/v1/suggestion'

# 遍历每个站点，提取线路和站点名称
for index, row in df_stations.iterrows():
    # 从station列读取站点信息
    full_station_info = row['station']

    # 从[]内读取线路
    line = full_station_info[full_station_info.find("[") + 1:full_station_info.find("]")]

    # 从[]后面读取站点名称
    station_name = full_station_info[full_station_info.find("]") + 1:].strip()

    print(f"正在处理: {line} : {station_name}")

    # 构造请求参数
    params = {
        # 'region': '成都市',
        'region_fix': 1,
        'keyword': f'{station_name}地铁站',  # 移除了方括号[]
        'apptag': 'lbsplace_sug',
        'key': '[你的key]',  # 替换为您的key
        'output': 'jsonp',
        'callback': 'jsonp_b33735cc1c5d98'  # 修改了callback的值
    }

#"https://apis.map.qq.com/ws/place/v1/suggestion?region=%E6%88%90%E9%83%BD%E5%B8%82&region_fix=1&keyword=%E9%9F%A6%E5%AE%B6%E7%A2%BE%E5%9C%B0%E9%93%81%E7%AB%99&apptag=lbsplace_sug&key=%5B%E4%BD%A0%E7%9A%84key%5D&output=jsonp&callback=jsonp_b33735cc1c5d98"
#"https://h5gw.map.qq.com/ws/place/v1/suggestion?keyword=&region_fix=&key=&apptag=lbsplace_sug&output=jsonp&callback=jsonp_b33735cc1c5d98"
    # 对参数进行URL编码
    encoded_params = urlencode(params)

    # 构造完整的URL
    full_url = f'{base_url}?{encoded_params}'

    print(full_url)

    # 发送请求
    response = requests.get(full_url)

    # 解析响应内容
    if response.status_code == 200:
        # print(response)
        # 去除JSONP格式的前缀和后缀
        # 找到第一个左括号和最后一个右括号的位置
        start = response.text.find('(') + 1
        end = response.text.rfind(')')

        # 提取JSON字符串
        json_str = response.text[start:end]
        data = json.loads(json_str)
        # 如果成功获取数据，则添加到DataFrame
        if data['status'] == 0 and data['data']:
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
    new_row = pd.DataFrame({'station': [f"[{line}]{station_name}"], 'lat': [lat], 'lon': [lon]})
    print(f"lat: {lat}, lon: {lon}")
    # 使用concat方法添加到df_subway
    df_subway = pd.concat([df_subway, new_row], ignore_index=True)

# 将数据保存到 Excel 文件
df_subway.to_excel('chengdu_subway_stations_with_coordinates2.xlsx', index=False)
print('Data saved to chengdu_subway_stations_with_coordinates2.xlsx')
