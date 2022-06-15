import pandas as pd
import numpy as np
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import json
import matplotlib.pyplot as plt

filename = '/Users/xuhongtao/PycharmProjects/Resource/12-16.csv'

# 读取数据
data = pd.read_csv(filename)
safe_bank = data[(data['BK300_name'] == '平安银行')]


def change_code(row):
    if len(row) < 6:
        row = '0' * (6 - len(row)) + row

    return row


data['BK300_code'] = data['BK300_code'].astype("str")
data["BK300_code"] = data["BK300_code"].apply(change_code)

# 删除缺失值、去重
safe_bank = safe_bank.dropna(how='any')
safe_bank = safe_bank.drop_duplicates()
# 排序index
safe_bank.sort_index(axis=0, ascending=True, inplace=True)

# 声明新DataFrame
new_data = pd.DataFrame(index=range(0, len(safe_bank)), columns=['date', 'close'])
# 传入数据
for i in range(0, len(safe_bank)):
    new_data['date'][i] = safe_bank['time'].iloc[i]
    new_data['close'][i] = safe_bank['closing'].iloc[i]

# 加入最新一日数据，用于预测未来一天的数据
new_data.loc[len(new_data)] = ['2021-12-17', '0.0']

# 转化date为datatime类型，可用于查询时间段
new_data['date'] = pd.to_datetime(new_data.date, format='%Y-%m-%d')
new_data.index = new_data['date']

# 排序索引
new_data.sort_index(ascending=True, inplace=True)

# 取给定时间范围内的数据
# new_data = new_data[(new_data['date'] > '2020-1-1')]
new_data = new_data[(new_data['date'] > '2021-1-1')]

# 确定训练数量
node = int(len(new_data) * 0.8)

forecast = []

# node之后为测试数据
prediction = new_data[node:]
for i in range(0, len(new_data) - node):

    # node之前为训练数据
    train = new_data[: node + i]
    # node之后为测试数据
    valid = new_data[node + i:]

    training = train['close']
    validation = valid['close']

    # 声明模型
    model = auto_arima(training, start_p=1, start_q=1, max_p=2, max_q=2, m=12, start_P=0, seasonal=True, d=1, D=1, trace=True,
                        error_action='ignore', suppress_warnings=True)

    # 训练模型
    model.fit(training)

    # 取训练数据往后一天的预测数据，挺高精准率
    forecast.append(model.predict(n_periods=1)[0])

# 新版本的pandas在加列数据时需要copy不然会报错
prediction = prediction.copy()
prediction['prediction'] = forecast
prediction['close'][-1] = 0.0
prediction['close'] = prediction['close'].astype("float64")

# 计算误差率
# prediction = prediction[:-1].eval("error_rate = ( prediction - close ) / close")
a = prediction[:-1].eval("error_rate = ( prediction - close ) / close")

def convert_to_positive(row):
    if '-' in row:
        return row[1:]

    return row


# 将负值转化为正数
a['error_rate'] = a['error_rate'].astype(str).apply(convert_to_positive).astype("float64")
# 计算总误差率(误差率为2.1%时的模型评分)
# 数据量对结果印象影响不大
print("误差率为2.1%时的模型评分： ", len(a[(a['error_rate'] <= 0.021)]) / len(a))      # 2019年开始——2年数据 61%  2020年开始 60%
print("误差率为2%时的模型评分： ", len(a[(a['error_rate'] <= 0.02)]) / len(a))
print("误差率为1.5%时的模型评分： ", len(a[(a['error_rate'] <= 0.015)]) / len(a))
print('模型均方误差为： ', mean_squared_error(prediction['close'][:-1], prediction['prediction'][:-1]))
print('模型绝对误差为： ', mean_absolute_error(prediction['close'][:-1], prediction['prediction'][:-1]))
print('模型标准差差为： ', np.sqrt(mean_squared_error(prediction['close'][:-1], prediction['prediction'][:-1])))

# plt.plot(train['Close'])
plt.plot(prediction['close'][:-1])
plt.plot(prediction['prediction'])
# plt.savefig('auto_arima.png')
plt.show()


def to_make_chart(prediction_result):

    to_json = {
        "date": [],
        "prediction": [],
    }

    # 将datetime转化为str
    date_dict = prediction_result['date'].astype("str").to_dict()
    prediction_dict = prediction_result['prediction'].astype("str").to_dict()

    for key in date_dict.keys():
        to_json["date"].append(date_dict[key])
        to_json["prediction"].append(prediction_dict[key])

    to_json['length'] = len(prediction_dict)

    # 将预测数据写入json中，用于在vue工程中做图表
    with open("/Users/xuhongtao/PycharmProjects/Resource/prediction.json", 'w+') as f:
        json.dump(to_json, f)

    stock_data = data[(data['BK300_name'] == '平安银行')][['BK300_code', 'closing', 'opening', 'lowest', 'highest', 'time']]
    a = stock_data['closing']

    ma_5 = []
    ma_10 = []
    ma_20 = []
    for i in range(0, len(a)+1):
        if i < 4:
            ma_5.append(None)
            continue

        ma_5.append(np.average(a[i-4: i+1]))

    for i in range(0, len(a) + 1):
        if i < 9:
            ma_10.append(None)
            continue

        ma_10.append(np.average(a[i - 9: i + 1]))


    for i in range(0, len(a) + 1):
        if i < 19:
            ma_20.append(None)
            continue

        ma_20.append(np.average(a[i - 19: i + 1]))

    stock_data['ma_5'] = ma_5[:-1]
    stock_data['ma_10'] = ma_10[:-1]
    stock_data['ma_20'] = ma_20[:-1]

    stock_data = stock_data[20:]

    stock_data['time'] = pd.to_datetime(stock_data.time, format='%Y-%m-%d')
    stock_data.index = stock_data['time']

    stock_data = stock_data[(stock_data['time'] > '2020-1-1')]
    stock_data['time'] = stock_data['time'].astype("str")

    stock_data_json = {
        'closing': [],
        'opening': [],
        'lowest': [],
        'highest': [],
        'time': [],
        'ma_5': [],
        'ma_10': [],
        'ma_20': [],
    }

    stock_data_dict = stock_data.to_dict()
    for key in stock_data_dict:
        if key == 'BK300_code':
            continue
        for k in stock_data_dict[key].keys():
            stock_data_json[key].append(stock_data_dict[key][k])

    with open('/Users/xuhongtao/PycharmProjects/Resource/{}_stock_data.json'.format(stock_data['BK300_code'][0]), 'w+') as f:
        json.dump(stock_data_json, f)


# to_make_chart(prediction)


