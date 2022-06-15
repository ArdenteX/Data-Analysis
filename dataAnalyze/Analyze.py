import datetime
import json
import pandas as pd
import numpy as np

# 读取数据
dataset = pd.read_csv('/Users/xuhongtao/PycharmProjects/Resource/12-16.csv')


# 将股票代码补全
def change_code(row):
    if len(row) < 6:
        row = '0' * (6 - len(row)) + row

    return row


# 股票代码转化为str类型
dataset['BK300_code'] = dataset['BK300_code'].astype("str")
# 批处理
dataset['BK300_code'] = dataset['BK300_code'].apply(change_code)

# 获取今天的日期(年月日)
today = datetime.datetime.today()
today = today.strftime("%Y-%m-%d")

dataset['time'] = pd.to_datetime(dataset.time, format="%Y-%m-%d")

# 获取今日数据
a = dataset[(dataset['time'] == today)]
# a = dataset[(dataset['time'] == '2021-12-16')]

# 在新版本的pandas中要变更列数据要先copy一下不然会报错
a = a.copy()
# 转化为float64
a['change'] = a['change'].astype("float64")
# 排序
a.sort_values(by=['change'], inplace=True, ascending=False)
# 取前10只股票，用于做柱状图
top_10 = a[:10][['BK300_name', 'vol', 'closing']]

# 用于存储成为json文件
to_json = {
    "BK300_name": [],
    "vol": [],
    "closing": []
}

# 转化为数组
top_10_dict = top_10.to_dict()
for key in top_10_dict:
    for k in top_10_dict[key].keys():
        to_json[key].append(top_10_dict[key][k])

# 写进json文件里
with open("/Users/xuhongtao/PycharmProjects/Resource/{}_Top10.json".format("2021-12-16"), 'w+') as f:
    json.dump(top_10_dict, f)


dataset['time'] = pd.to_datetime(dataset.time, format="%Y-%m-%d")
new_data = dataset[(dataset['time'] == '2021-12-16')]
new_data = new_data[['closing', 'BK300_name', 'vol']].astype(str)

new_data = new_data.iloc[: 10]

to_bar = {
    'closing': [],
    'BK300_name': [],
    'vol': [],
}

with open('/Users/xuhongtao/PycharmProjects/Resource/to_bar.json', 'w+') as f:
    to_bar['closing'] = new_data['closing'].tolist()
    to_bar['BK300_name'] = new_data['BK300_name'].tolist()
    to_bar['vol'] = new_data['vol'].tolist()
    json.dump(to_bar, f)

max(to_bar['closing'])
max(to_bar['vol'])
min(to_bar['vol'])
