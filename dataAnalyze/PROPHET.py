from fbprophet import Prophet
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

filename = '/Users/xuhongtao/PycharmProjects/Resource/12-16.csv'

data = pd.read_csv(filename)
safe_bank = data[(data['BK300_name'] == '平安银行')]

# 获取平安银行的数据
safe_bank = safe_bank.dropna(how='any')
safe_bank = safe_bank.drop_duplicates()
safe_bank.sort_index(axis=0, ascending=True, inplace=True)

# 创建新DataFrame
new_data = pd.DataFrame(index=range(0, len(safe_bank)), columns=['date', 'close'])
# 传入数据
for i in range(0, len(safe_bank)):
    new_data['date'][i] = safe_bank['time'].iloc[i]
    new_data['close'][i] = safe_bank['closing'].iloc[i]

# 转化date为datetime类型，可用于查询时间段
new_data['date'] = pd.to_datetime(new_data.date, format='%Y-%m-%d')
# 将新的DataFrame的索引转化为date类型
new_data.index = new_data['date']

# 排序
new_data.sort_index(ascending=True, inplace=True)

# 取给定时间范围内的数据
# new_data = new_data[(new_data['date'] > '2020-1-1')]
new_data = new_data[(new_data['date'] > '2021-1-1')]
new_data.rename(columns={'date': 'ds', 'close': 'y'}, inplace=True)

node = int(len(new_data) * 0.8)

forecast_list = []
prediction = new_data[node:]
for i in range(0, len(new_data) - node):

    train = new_data[: node + i]
    valid = new_data[node + i:]

    model = Prophet(daily_seasonality=True, yearly_seasonality=True)
    # 训练
    model.fit(train)

    # 获取日期
    closing_prediction = model.make_future_dataframe(periods=1)
    # 预测
    forecast = model.predict(closing_prediction)
    # 获取预测数据
    forecast_list.append(forecast['yhat'][len(train)])
    print(forecast['yhat'][len(train): len(train)+1])

prediction = prediction.copy()
prediction['prediction'] = forecast_list
prediction['y'] = prediction['y'].astype("float64")

# 计算误差率
prediction = prediction.eval("error_rate = ( prediction - y ) / y")
a = prediction.eval("error_rate = ( prediction - y ) / y")

def convert_to_positive(row):
    if '-' in row:
        return row[1:]

    return row


# 将负值转化为正数
a['error_rate'] = a['error_rate'].astype(str).apply(convert_to_positive).astype("float64")
# 计算总误差率(误差率为2.1%时的模型评分)
# 数据量对结果印象影响不大
print("误差率为2.1%时的模型评分： ", len(a[(a['error_rate'] <= 0.021)]) / len(a))      # 2019年开始——2年数据 61%
print("误差率为2%时的模型评分： ", len(a[(a['error_rate'] <= 0.02)]) / len(a))
print("误差率为1.5%时的模型评分： ", len(a[(a['error_rate'] <= 0.015)]) / len(a))
print('模型均方误差为： ', mean_squared_error(prediction['y'][:-1], prediction['prediction'][:-1]))
print('模型绝对误差为： ', mean_absolute_error(prediction['y'][:-1], prediction['prediction'][:-1]))
print('模型标准差差为： ', np.sqrt(mean_squared_error(prediction['y'][:-1], prediction['prediction'][:-1])))

mean = np.mean(a['error_rate'])
average = np.average(a['error_rate'])


# plt.plot(train['Close'])
plt.plot(prediction[['y', 'prediction']])
plt.show()


train_1 = new_data[: node]
valid_1 = new_data[node:]

model = Prophet()
model.fit(train_1)

close_prices = model.make_future_dataframe(periods=len(valid_1))
forecast = model.predict(close_prices)
forecast_valid = forecast['yhat'][node:]
valid_1['Predictions'] = forecast_valid.values
plt.plot(valid_1['y'], label = '训练集')
plt.plot(valid_1[['y', 'Predictions']], label = ['真实值', '预测值'])
plt.show()