import pandas as pd
import numpy as np
from sklearn import svm, preprocessing

filename = '/Users/xuhongtao/PycharmProjects/Resource/12-16.csv'

# 读取数据，并转化数据 'time' 特征为datetime类型，并将其作为索引
data = pd.read_csv(filename, parse_dates=[10], index_col='time')
safe_bank = data[(data['BK300_name'] == '平安银行')]

# 删除缺失值、去重
safe_bank = safe_bank.dropna(how='any')
safe_bank = safe_bank.drop_duplicates()
# 排序索引
safe_bank.sort_index(axis=0, ascending=True, inplace=True)

# 转化为list，为后面计算均线作准备
a = safe_bank['closing'].to_list()
d = safe_bank['vol'].to_list()

# 计算5日、10日、20日均线
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

# 计算5、10、20日成交量均线
v_ma_5 = []
v_ma_10 = []
v_ma_20 = []

for i in range(0, len(d)+1):
    if i < 4:
        v_ma_5.append(None)
        continue

    v_ma_5.append(np.average(d[i-4: i+1]))


for i in range(0, len(d) + 1):
    if i < 9:
        v_ma_10.append(None)
        continue

    v_ma_10.append(np.average(d[i - 9: i + 1]))

for i in range(0, len(d) + 1):
    if i < 19:
        v_ma_20.append(None)
        continue

    v_ma_20.append(np.average(d[i - 19: i + 1]))

safe_bank['ma_5'] = ma_5[:-1]
safe_bank['ma_10'] = ma_10[:-1]
safe_bank['ma_20'] = ma_20[:-1]
safe_bank['v_ma_5'] = v_ma_5[:-1]
safe_bank['v_ma_10'] = v_ma_10[:-1]
safe_bank['v_ma_20'] = v_ma_20[:-1]

safe_bank = safe_bank[20: ]
# 声明value为两天数据的收市价格之差
value = pd.Series(safe_bank['closing'] - safe_bank['closing'].shift(1), index=safe_bank.index)
# 向上填充
value = value.bfill()
# 将value分类为0和1
value[value >= 0] = 1
value[value < 0] = 0

# 加入新特征
safe_bank['value'] = value

# 将数据集中的数字特征转化为float64
pre_train = safe_bank[safe_bank.columns[3:]].astype("float64")
# 长度
L = len(pre_train)

# 训练级占80%
train = int(L * 0.8)
# 测试集数量
predict_test = L - train

# 删除value特征
pre_train = pre_train.drop(['value'], axis=1)
# 这里因为数据量太大无法直接用preprocessing正则化，所以先转化成ndarry再进行正则化就可以了
pre_train = np.matrix(pre_train).astype(float)
pre_train = preprocessing.scale(pre_train)

correct = 0
train_original = train

while train < L:
    # train会增大，一开始为0：train
    data_train = pre_train[train - train_original: train]
    data_value = value[train - train_original: train]
    # 每次只有一个测试样本
    data_test = pre_train[train: train + 1]
    data_real_value = value[train: train + 1]

    # 因为版本更新的原因，所以这里gamma需要显式注明为scale不然会报错
    classifier = svm.SVC(C=1.0, kernel='rbf', gamma='scale')
    classifier.fit(data_train, data_value)
    value_predict = classifier.predict(data_test)

    print("value_real=%d value_predict=%d" % (data_real_value[0], value_predict))
    # 预测正确则 +1
    if data_real_value[0] == int(value_predict):
        correct = correct + 1
    train = train + 1

# 打印精确度
print(correct)
print(predict_test)
correct = correct * 100 / predict_test
print("Correct=%.2f%%" % correct)