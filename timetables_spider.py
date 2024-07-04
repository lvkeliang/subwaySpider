import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
import re

# 目标网址
url = 'https://www.cqmetro.cn/smbsj.html'

# 发送请求获取网页内容
response = requests.get(url)
response.encoding = 'utf-8'

# 解析HTML内容
soup = BeautifulSoup(response.text, 'html.parser')

# 查找页面上的所有表格
tables = soup.find_all('table')

# 创建Pandas ExcelWriter对象，使用openpyxl引擎
with pd.ExcelWriter('timetables.xlsx', engine='openpyxl') as writer:
    # 遍历每个表格
    for i, table in enumerate(tables):
        # 将表格转换为字符串，并将所有HTML标签转换为大写
        table_html_upper = re.sub(r'<.*?>', lambda x: x.group(0).upper(), str(table))
        # 使用StringIO包装修改后的HTML内容
        table_html_io = StringIO(table_html_upper)

        # 将表格读入DataFrame
        df = pd.read_html(table_html_io)[0]

        # 如果DataFrame有MultiIndex列，重置索引
        if isinstance(df.columns, pd.MultiIndex):
            df.reset_index(inplace=True)

        # 将DataFrame写入ExcelWriter对象
        df.to_excel(writer, sheet_name=f'Table_{i}')

print('所有表格已保存到Excel文件。')
