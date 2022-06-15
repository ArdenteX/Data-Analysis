# 近邻传播算法，聚类
import random
from sklearn.cluster import affinity_propagation
from sklearn.covariance import GraphicalLassoCV
from sklearn import metrics
import pandas as pd
import numpy as np
import json

filename = '/Users/xuhongtao/PycharmProjects/Resource/12-16.csv'

# 读取数据
data = pd.read_csv(filename)
# 删除缺失值
data.dropna(inplace=True)
# 去重
data.drop_duplicates(inplace=True)

# 转化时间为datetime类型
data['time'] = pd.to_datetime(data.time, format="%Y-%m-%d")

# 构建新特征
data = data.eval("diff=(closing - opening)/opening")

# 获取部分数据
new_data = data[(data['time'] > '2010-1-1')]
new_data = new_data[['time', 'BK300_name', 'diff']]

# # 构建组
# a = new_data.groupby(['time', 'BK300_name'], as_index=False).sum()
#
# b = pd.pivot_table(a, index='time', columns='BK300_name')

# 构建以时间序列为index，股票名为列名的表
new_data_2 = new_data.pivot_table("diff", "time", "BK300_name")

# 在这一天的股票数据为nan则要清除掉这个列，因为这时候只股票还未上市
a = new_data_2.loc['2010-01-28'].isna().to_dict()

# 清除2010-01-28年未上市的股票
for key in a.keys():
    if a[key] is True:
        print(key)
        new_data_2 = new_data_2.drop(columns=key, axis=1)

# 缺失值处理(向上填充)
new_data_2.bfill(inplace=True)

# 转化为numpy array
new_data_np = np.array(new_data_2)
# 正则化(标准差)
new_data_np /= np.std(new_data_np, axis=0)

# 相关性训练
edge_model = GraphicalLassoCV()
edge_model.fit(new_data_np)

# 聚类(通过相关性训练出来的结果进行聚类)
_, labels = affinity_propagation(edge_model.covariance_)

# 蔟数量
labels_max = max(labels)

print("Total Stock: {}".format(labels_max + 1))

stock_name = list(new_data_2.columns)
for i in range(0, labels_max+1):
    print("Cluster: {} -------> stock: {}".format(i, ','.join(np.array(stock_name)[labels == i])))

# 评分
score = metrics.silhouette_score(edge_model.covariance_, labels, metric='euclidean')
print("silhouette_score: ", score)


to_json = {
    "nodes": [],
    "links": [],
    "categories": []
}
#
# # 二象限
# x = random.uniform(-300, -200)
# y = random.uniform(300, 200)
cat_ = ['股份', '生物制品', '铁公基', '工业类', '建筑类', '银行', '电力相关', '医药', '食品', '证券', '科技']


# 将label转化为name_value形式，用于制作饼图
def process_label_into_json(result_labels):
    label_count = {}
    for i in range(0, len(result_labels)):
        if cat_[result_labels[i]] not in label_count:
            label_count[cat_[result_labels[i]]] = 1
            continue

        label_count[cat_[result_labels[i]]] += 1

    label_to_json = []
    for key in label_count.keys():
        a = {}
        a['name'] = key
        a['value'] = label_count[key]
        label_to_json.append(a)

    with open('/Users/xuhongtao/PycharmProjects/Resource/cluster_pie.json', 'w+') as f:
        json.dump(label_to_json, f)


process_label_into_json(labels)

for i in range(0, len(stock_name)):
    category = labels[i]
    if category == 0:
        a = {
                "id": str(i),
                "name": stock_name[i],
                'x': random.uniform(0, 100),
                'y': random.uniform(50, 150),
                'value': str(10),
                'category': cat_[category]
        }
        to_json["nodes"].append(a)

    if category == 1:
        a = {
            "id": str(i),
            "name": stock_name[i],
            'x': random.uniform(100, 150),
            'y': random.uniform(-50, 0),
            'value': str(15),
            'category': cat_[category]
        }
        to_json["nodes"].append(a)

    if category == 2:
        a = {
            "id": str(i),
            "name": stock_name[i],
            'x': random.uniform(150, 250),
            'y': random.uniform(150, 250),
            'value': str(12),
            'category': cat_[category]
        }
        to_json["nodes"].append(a)

    if category == 3:
        a = {
            "id": str(i),
            "name": stock_name[i],
            'x': random.uniform(0, -100),
            'y': random.uniform(0, 50),
            'value': str(13),
            'category': cat_[category]
        }
        to_json["nodes"].append(a)

    if category == 4:
        a = {
            "id": str(i),
            "name": stock_name[i],
            'x': random.uniform(250, 350),
            'y': random.uniform(-50, -150),
            'value': str(11),
            'category': cat_[category]
        }
        to_json["nodes"].append(a)

    if category == 5:
        a = {
            "id": str(i),
            "name": stock_name[i],
            'x': random.uniform(-100, -250),
            'y': random.uniform(-100, -200),
            'value': str(13),
            'category': cat_[category]
        }
        to_json["nodes"].append(a)

    if category == 6:
        a = {
            "id": str(i),
            "name": stock_name[i],
            'x': random.uniform(-50, -150),
            'y': random.uniform(-50, 50),
            'value': str(14),
            'category': cat_[category]
        }
        to_json["nodes"].append(a)

    if category == 7:
        a = {
            "id": str(i),
            "name": stock_name[i],
            'x': random.uniform(-60, -160),
            'y': random.uniform(150, 200),
            'value': str(10),
            'category': cat_[category]
        }
        to_json["nodes"].append(a)

    if category == 8:
        a = {
            "id": str(i),
            "name": stock_name[i],
            'x': random.uniform(60, 160),
            'y': random.uniform(-50, -150),
            'value': str(12),
            'category': cat_[category]
        }
        to_json["nodes"].append(a)

    if category == 9:
        a = {
            "id": str(i),
            "name": stock_name[i],
            'x': random.uniform(160, 260),
            'y': random.uniform(-150, -280),
            'value': str(11),
            'category': cat_[category]
        }
        to_json["nodes"].append(a)

    if category == 10:
        a = {
            "id": str(i),
            "name": stock_name[i],
            'x': random.uniform(300, 400),
            'y': random.uniform(100, 50),
            'value': str(8),
            'category': cat_[category]
        }
        to_json["nodes"].append(a)

for i in range(0, len(stock_name)):

    for j in range(0, len(stock_name)):
        if i == j:
            continue

        b = {
            "source": str(i),
            "target": str(j)
             }
        to_json['links'].append(b)

for i in range(0, labels_max+1):
    a = {"name": str(cat_[i])}
    to_json["categories"].append(a)


with open("/Users/xuhongtao/PycharmProjects/Resource/relation.json", 'w+') as f:
    json.dump(to_json, f)
