import numpy as np
import math

# delay_his = np.array([10,5,6,4,8,7])
# length = len(delay_his)
# n = 2
# for i in range(0, int(length/n)):
#     smooth = sum(delay_his[n*i:n*(i+1)])
#     delay_his[n*i:n*(i+1)] = np.array([smooth]*n)
# print(delay_his)

# record = [[1,2,3],[4,5,6],[7,8,9]]
# print(sum(record, []))
# print(np.arange(10))
# print(np.random.choice(np.arange(10), 1))

# task_t = 10
# user_n = 5
#
# tasks = np.zeros([2, task_t], dtype=float)
# # 任务上传数据量
# upload_data = np.random.normal(50, 10, task_t)
# while sum(upload_data < 0):  # 确保生成的数据中不包含负值
#     upload_data = np.random.normal(50, 10, task_t)
# print(upload_data)
# # 任务计算量
# cpu_cycle = np.random.normal(100, 20, task_t)
# while sum(cpu_cycle < 0):
#     cpu_cycle = np.random.normal(100, 20, task_t)
# print(cpu_cycle)
# tasks[0, :] = upload_data
# tasks[1, :] = cpu_cycle
# task = np.transpose(tasks)
#
# choice_arr = np.random.choice(np.random.randint(0, task_t, task_t, dtype=int), size=user_n,
#                                 replace=True, p=None)
# print(choice_arr)
# print(sum(choice_arr))
# request = np.zeros([user_n, 2], dtype=float)
# user_index = 0
# for choice in choice_arr:
#     request[user_index, :] = task[choice, :]
#     user_index += 1
# print(request)
# print(request.flatten())
#
# observation_n = user_n * 3
# observation = np.zeros(observation_n, dtype=float)  # 观测
# observation[0:user_n] = np.zeros(user_n, dtype=float)
# observation[user_n:observation_n] = request.flatten()
# print(observation)

# def smooth(record, n):
#     # 对结果进行平滑处理
#     length = len(record)
#     for i in range(int(length/n)):
#         average = sum(record[n*i:n*(i+1)]) / n
#         record[n*i:n*(i+1)] = np.array([average]*n)
#     return record

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math


def zero(y):
    for i in range(len(y)):
        if y[i] < 0:
            y[i] = 0
    return y


def reverse(y):
    for i in range(len(y)):
        if y[i] < 0:
            y[i] = -y[i]
    return y


def zipf(task_t, c, a):
    p = np.zeros(task_t)
    for rank in range(task_t):
        p[rank] = c / ((rank+1)**a)
    return p

def request(task_t, cycle, time_slot):
    x = np.linspace(0,time_slot,time_slot)
    p = zipf(task_t, 0.5, 0.8)
    f = np.zeros([task_t, time_slot], dtype=int)
    for i in range(task_t):
        y = zero(np.sin(2*cycle*x+(task_t-i)*0.5*cycle) * zero(np.sin(cycle*x+(task_t-i)*0.5*cycle))) * p[i]
        f[i, :] = y
        plt.plot(x, y)


sns.set()
#request(10, 0.2*math.pi, 50)
#设置seaborn默认格式
x = np.linspace(0,50,50)
plt.subplot(311)
c = 0.4*math.pi
y1 = reverse(np.sin(c*x))
plt.plot(x, y1)
plt.subplot(312)
x_sub = np.linspace(0,50,50)
y2 = 1/(2**(x_sub%10))
# y2 = np.sin(2*c*x)
plt.plot(x_sub, y2)
plt.subplot(313)
y3 = zero(y1*y2)
plt.plot(x, y3)
# p = zipf(10, 0.5, 0.8)
# plt.plot(p)
plt.show()