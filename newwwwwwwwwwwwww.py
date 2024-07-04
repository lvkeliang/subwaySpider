import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import FactorAnalysis
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import numpy as np

# 读取Excel文件
file_path = '/path_to_your_file/原始数据.xlsx'  # 替换为你的文件路径
current_all_data = pd.read_excel(file_path, sheet_name='current all')

# 生成描述性统计
descriptive_stats = current_all_data.describe(include='all')
print("描述性统计：")
print(descriptive_stats)

# 设置视觉风格
sns.set(style="whitegrid")

# 定义数值和分类列
numerical_columns = ['Publication', 'EntryAge', 'YearofStudent']
categorical_columns = ['Gender', 'School', 'Scholarship']

# 绘制数值列的直方图
plt.figure(figsize=(15, 5))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(1, 3, i)
    sns.histplot(current_all_data[column], kde=True)
    plt.title(f'{column}的直方图')
plt.tight_layout()
plt.show()

# 绘制分类列的条形图
plt.figure(figsize=(15, 5))
for i, column in enumerate(categorical_columns, 1):
    plt.subplot(1, 3, i)
    sns.countplot(x=column, data=current_all_data)
    plt.title(f'{column}的条形图')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# T检验：比较不同性别的发表文章数量
male_publications = current_all_data[current_all_data['Gender'] == 'M']['Publication']
female_publications = current_all_data[current_all_data['Gender'] == 'F']['Publication']
t_stat, p_val = stats.ttest_ind(male_publications, female_publications)
print(f"T检验结果: t_stat = {t_stat}, p_val = {p_val}")

# 方差分析 (ANOVA)：比较不同导师经验组别的发表文章数量
anova_result = stats.f_oneway(
    current_all_data[current_all_data['Supervisor Experience'] == '>5']['Publication'],
    current_all_data[current_all_data['Supervisor Experience'] == '3-5']['Publication']
)
print(f"方差分析结果: F_stat = {anova_result.statistic}, p_val = {anova_result.pvalue}")

# 相关分析：计算发表文章数量与入学年龄之间的相关性
correlation = current_all_data[['Publication', 'EntryAge']].corr()
print("相关分析结果：")
print(correlation)

# 回归分析：探讨发表文章数量的影响因素
X = current_all_data[['EntryAge', 'YearofStudent']]  # 自变量
y = current_all_data['Publication']  # 因变量
reg = LinearRegression().fit(X, y)
print(f"回归分析结果: 回归系数 = {reg.coef_}, 截距 = {reg.intercept_}, R^2 = {reg.score(X, y)}")

# 因子分析：识别潜在的因素结构
fa = FactorAnalysis(n_components=2)
factors = fa.fit_transform(current_all_data[numerical_columns].dropna())
print("因子分析结果：")
print(factors)

# 聚类分析：识别数据中的不同特征组合
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(current_all_data[numerical_columns].dropna())
current_all_data['Cluster'] = np.nan
current_all_data.loc[current_all_data[numerical_columns].dropna().index, 'Cluster'] = clusters
print("聚类分析结果：")
print(current_all_data[['StudentID', 'Cluster']])
