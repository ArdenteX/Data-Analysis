import pandas as pd
import numpy as np
from sklearn import svm, preprocessing

filename = 'D:\\广商网课\\作业\\scienceOfData\\DFCF.csv'

data = pd.read_csv(filename, parse_dates=[10], index_col='time')
safe_bank = data[(data['BK300_name'] == '平安银行')]

safe_bank = safe_bank.dropna(how='any')
safe_bank = safe_bank.drop_duplicates()
safe_bank.sort_index(axis=0, ascending=True, inplace=True)

a = safe_bank['closing'].to_list()
d = safe_bank['vol'].to_list()

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

pre_train = safe_bank[safe_bank.columns[3:]].astype("float64")
L = len(pre_train)

# 测试 未完成,label数据可能太大导致训练太慢而且准确率也不高
train = int(L * 0.8)
correct = 0

train_original = train
predict_test = L - train

predict_closing = safe_bank[safe_bank.columns[3:]].astype("float64")
predict_closing = predict_closing.drop(["closing"], axis=1)
predict_closing = np.array(predict_closing).astype(float)
predict_closing = preprocessing.scale(predict_closing)

closing = safe_bank['closing'].astype(float) * 100
closing = closing.astype('int')

classify_tester = svm.SVC(C=1.0, kernel='rbf', gamma='scale')

while train < L:
    train_data = predict_closing[train - train_original: train]
    train_value = closing[train - train_original: train]
    test_data1 = predict_closing[train: train + 1]
    test_value = closing[train: train + 1]

    classify_tester.fit(train_data, train_value)
    predict_value = classify_tester.predict(test_data1)
    print("value_real=%.2f value_predict=%.2f" % (test_value[0] / 100.0, predict_value / 100.0))
    if test_value[0] == int(predict_value):
        correct = correct + 1
    train = train + 1

print(correct)
print(predict_test)
correct = correct * 100 / predict_test
print("Correct=%.2f%%" % correct)
