# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import pandas as pd
import seaborn as sns
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from Environment.Request import LSTMRequestGenerator

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# fix random seed for reproducibility
np.random.seed(7)

gen = LSTMRequestGenerator(task_t=10, user_n=20, time_slot=100)
dataset = gen.task_cnt[:, 0]  # 0号任务请求数量与时隙的关系
# plt.plot(dataset)
# plt.show()

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = np.array(dataset).reshape(-1,1)
dataset = scaler.fit_transform(dataset)


# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# use this function to prepare the train and test datasets for modeling
look_back = 5
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(8, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=1000, batch_size=30, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
# 训练数据拟合效果
ch = font_manager.FontProperties(fname="/Library/Fonts/Arial Unicode.ttf")
sns.set(style="whitegrid",font=ch.get_name())
plt.subplot(1,2,1)
x = np.linspace(0, len(trainPredict), len(trainPredict))
trainPredict = trainPredict[:, 0]
trainY = trainY[0, :]
# 对预测数据进行取整处理
# for i in range(len(trainPredict)):
#     if trainPredict[i] < 1:
#         trainPredict[i] = 1
#     else:
#         trainPredict[i] = round(trainPredict[i])
data = {
        "训练预测值": trainPredict,
        "训练真实值": trainY,
    }
df = pd.DataFrame(data, index=x, columns=["训练预测值", "训练真实值"])
sns.lineplot(data=df)
plt.xlabel("时隙")
plt.ylabel("任务请求数量")
# plt.show()

# 测试数据
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# 测试数据拟合效果
plt.subplot(1,2,2)
x = np.linspace(0, len(testPredict), len(testPredict))
testPredict = testPredict[:, 0]
testY = testY[0, :]
# # 对预测数据进行取整处理
# for i in range(len(testPredict)):
#     if testPredict[i] < 1:
#         testPredict[i] = 1
#     else:
#         testPredict[i] = round(testPredict[i])
data = {
        "测试预测值": testPredict,
        "测试真实值": testY,
    }
df = pd.DataFrame(data, index=x, columns=["测试预测值", "测试真实值"])
sns.lineplot(data=df)
plt.xlabel("时隙")
plt.ylabel("任务请求数量")
plt.show()

# trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
# print('Train Score: %.2f RMSE' % (trainScore))
# testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
# print('Test Score: %.2f RMSE' % (testScore))
#
# # shift train predictions for plotting
# trainPredictPlot = np.empty_like(dataset)
# trainPredictPlot[:, :] = np.nan
# trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
#
# # shift test predictions for plotting
# testPredictPlot = np.empty_like(dataset)
# testPredictPlot[:, :] = np.nan
# testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
#
# # plot baseline and predictions
# plt.plot(scaler.inverse_transform(dataset))
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)