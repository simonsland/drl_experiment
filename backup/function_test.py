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

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.font_manager as font_manager
# import seaborn as sns
# import math
#
# # 设置中文字体
# ch = font_manager.FontProperties(fname="/Library/Fonts/Arial Unicode.ttf")
# sns.set(font=ch.get_name())
#
# data = {
#     # "迭代次数": np.linspace(0,5,5),
#     "损失变化": [1.5,1.7,3.6,2.4,2.9],
#     "奖励变化": [1,1,3,2,2]
# }
# df = pd.DataFrame(data, index=np.linspace(0,5,5), columns=["损失变化", "奖励变化"])
# sns.lineplot(data=df)
# plt.xlabel("iter")
# plt.ylabel("change")
# plt.show()


# def zero(y):
#     for i in range(len(y)):
#         if y[i] < 0:
#             y[i] = 0
#     return y
#
#
# def reverse(y):
#     for i in range(len(y)):
#         if y[i] < 0:
#             y[i] = -y[i]
#     return y
#
#
# def zipf(task_t, c, a):
#     p = np.zeros(task_t)
#     for rank in range(task_t):
#         p[rank] = c / ((rank + 1) ** a)
#     return p
#
#
# def request(user_n, task_t, time_slot):
#     x = np.linspace(0, time_slot, time_slot)
#     cycle = time_slot / 5
#     c = 2 * math.pi / cycle
#     f = np.zeros([task_t, time_slot], dtype=float)
#     r = np.zeros([task_t, time_slot], dtype=int)
#     request = np.zeros([user_n, time_slot], dtype=int)
#     for i in range(task_t):
#         y = 0.3 * np.sin(c*(x+i*cycle/task_t)) + 0.5
#         f[i, :] = y
#         # plt.plot(x, y)
#     for j in range(time_slot):
#         for i in range(task_t):
#             r[i,j] = round(f[i,j] / sum(f[:,j]) * user_n)
#     for j in range(time_slot):
#         idx = 0
#         for i in range(task_t):
#             if r[i, j] > 0:
#                 s = idx
#                 e = idx + r[i, j] if (idx + r[i, j]) <= user_n else user_n
#                 request[s:e, j] = i
#                 idx = e

# import matplotlib.pyplot as plt
# import seaborn as sns
#
# sns.set()
# request(20, 10, 100)

# c = []
# a = [1,2,3,4,5]
# b = [1,2,3]
# c.append(a)
# c.append(b)
# print(c[1])

a = np.array([1,2,2,3], dtype=int)
print(np.sum(a==2))

# #设置seaborn默认格式
# # x = np.linspace(0,100,100)
# # c = 0.1*math.pi
# # plt.subplot(311)
# # y1 = 0.3 * np.sin(c*x) + 0.5
# # plt.plot(x, y1)
# # plt.subplot(311)
# # y1 = 0.3 * np.sin(c*(x+2)) + 0.5
# # plt.plot(x, y1)
#
# # plt.subplot(312)
# # x_sub = np.linspace(0, 100, 100)
# # y2 = 1/(2**(x_sub%10))
# # # y2 = np.sin(2*c*x)
# # plt.plot(x_sub, y2)
# # plt.subplot(313)
# # y3 = zero(y1*y2)
# # plt.plot(x, y3)
# # p = zipf(10, 0.5, 0.8)
# # plt.plot(p)
# plt.show()

# dict ={}
# dict[1] = 1
# dict[2] = 2
# dict[3] = 3
# dict[4] = 4
# dict[5] = 5
# sort = sorted(dict, key=lambda k: dict[k], reverse=True)
# # 通过双指针算法得到满足缓存容量情况下的最大任务序列
# l = 0
# r = 0
# max_benefit = 0
# benefit_tmp = 0
# cache_v = []
# cache_tmp = []
# capacity_r = 10  # 剩余的缓存容量
# while l < 5:
#     while r < 5 and capacity_r >= sort[r]:
#         cache_tmp.append(sort[r])
#         benefit_tmp += dict[sort[r]]
#         capacity_r -= sort[r]
#         r += 1
#     if benefit_tmp > max_benefit or (math.isclose(benefit_tmp, max_benefit) and len(cache_tmp) > len(cache_v)):
#         max_benefit = benefit_tmp
#         cache_v = cache_tmp.copy()
#     capacity_r += sort[l]
#     cache_tmp.remove(sort[l])
#     benefit_tmp -= dict[sort[l]]
#     l += 1
# cache = np.zeros(5, dtype=int)
# for elem in cache_v:
#     cache[elem-1] = 1
# print(cache)
a = [3, 2, 1]
b = [1, 1, 1]
print(abs(np.array(b) - np.array(a)))