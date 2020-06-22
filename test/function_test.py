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

# a = np.array([1,2,2,3], dtype=int)
# print(np.sum(a==2))

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

from Environment.Request import LSTMRequestGenerator
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import seaborn as sns
import pandas as pd
#
# ch = font_manager.FontProperties(fname="/Library/Fonts/Arial Unicode.ttf")
# sns.set(style='whitegrid', font=ch.get_name())
#
# task_t=4
# time_slot=100
# x = np.linspace(0, time_slot, time_slot)
# cycle = time_slot / 2
# c = 2 * math.pi / cycle
# f = np.zeros([task_t, time_slot], dtype=float)
# sns.set(style='whitegrid', font=ch.get_name())
# # 任务流行度演化
# plt.subplot(1,2,1)
# for i in range(task_t):
#     y = 0.3 * np.sin(c*(x+i*cycle/task_t)) + 0.5
#     f[i, :] = y
# data = {
#         "0": f[0,:],
#         "1": f[1,:],
#         "2": f[2,:],
#         "3": f[3,:]
#     }
# df = pd.DataFrame(data, index=x, columns=["0", "1", "2", "3"])
# sns.lineplot(data=df)
# plt.xlabel("时隙")
# plt.ylabel("流行度演化")
# # 任务请求次数
# # time_slot = 100
# # task_t = 10
# user_n = 10
# x = np.linspace(0, time_slot, time_slot)
# cycle = time_slot / 2
# c = 2 * math.pi / cycle
# f = np.zeros([task_t, time_slot], dtype=float)
# r = np.zeros([task_t, time_slot], dtype=int)
# for i in range(task_t):
#     y = 0.3 * np.sin(c*(x+i*cycle/task_t)) + 0.5
#     f[i, :] = y
#     # plt.plot(x, y)
# for j in range(time_slot):
#     for i in range(task_t):
#         r[i, j] = round(f[i, j] / sum(f[:, j]) * user_n)
# plt.subplot(1,2,2)
# data = {
#         "0": r[0,:]
#     }
# df = pd.DataFrame(data, index=x, columns=["0"])
# sns.lineplot(data=df)
# plt.xlabel("时隙")
# plt.ylabel("任务请求数量")
# plt.show()

# 缓存对比数据（流行度预测）
# s_cache_v1 = [755.8202380952382, 689.7999999999998,  672.05, 720.0166666666668, 710.55, 690.4833333333332, 666.9999999999999, 718.5166666666668, 697.5666666666666, 705.0]
# s_cache_v3 = [757.5595238095237, 695.6666666666666,  679.5, 733.85, 726.5333333333333, 704.5833333333334, 679.0, 736.6166666666667, 722.1666666666667, 738.4833333333333]
# s_cache_none = [755.2234126984126, 748.8357142857144, 779.225, 798.4833333333333, 794.5666666666666, 794.5095238095238, 790.6738095238095, 754.1666666666667, 730.6190476190476, 756.0285714285715]
# x = [1,2,3,4,5,6,7,8,9,10]
#
# s_cache_v1 = [734.25, 680.3999999999999, 696.95, 724.7, 730.25, 721.8666666666667, 708.75, 686.85, 669.1166666666664, 680.4333333333334]
# s_cache_v2 = [738.7476190476192, 721.7166666666666, 721.9666666666666, 740.0333333333334, 733.9000000000001, 730.7666666666667, 730.7500000000001, 690.05, 676.2166666666666, 677.25]
#
# s_cache_v1 = [644.8642857142858, 596.8166666666665, 633.7166666666666, 673.1500000000001, 673.6500000000001, 679.25, 665.6500000000001, 629.7, 591.05, 584.1]
# s_cache_v2 = [659.1250000000001, 616.5, 650.5833333333333, 687.7333333333333, 692.3666666666667, 681.6500000000001, 670.3166666666667, 637.95, 594.6500000000001, 607.8833333333333]
# # # 条形图
# ch = font_manager.FontProperties(fname="/Library/Fonts/Arial Unicode.ttf")
# sns.set(style="whitegrid",font=ch.get_name())
# # b1 = plt.bar(x=x, height=s_cache_v1, width=0.4)
# # b3 = plt.bar(x=x, height=list(np.array(s_cache_v3) - np.array(s_cache_v1)), width=0.4, bottom=s_cache_v1)
# # plt.legend([b1,b3], ["proposed", "experience"], loc='upper right')
# # plt.ylim(600, 850)
# # plt.xlabel("时间步")
# # plt.ylabel("系统消耗")
# label = [1,2,3,4,5,6,7,8,9,10]
# b5 = plt.bar(x=x, height=s_cache_v1, width=0.2)
# b6 = plt.bar(x=[i + 0.2 for i in x], height=s_cache_v2, width=0.2)
# plt.xticks([i + 0.2 for i in x], label)
# plt.legend([b5, b6], ["proposed", "prediction-only"], loc='upper right')
# plt.ylim(550, 750)
# plt.xlabel("时间步")
# plt.ylabel("系统消耗")
# plt.show()

# 缓存容量
# s_cache_v1:  [686.6583333333334, 673.8291666666667, 670.4349999999998, 669.0579761904761, 676.0508333333335]
# s_cache_v2:  [696.5116666666669, 690.4383333333335, 673.7433333333333, 667.0308333333335, 674.2658333333334]
# s_cache_v3:  [703.6233333333333, 703.2741666666667, 681.7475, 680.4275, 674.7533333333332]
# # result II
# s_cache_v1 = [699.6465476190475, 675.5383333333334, 670.8291666666665, 672.4708333333335, 683.7233333333335]
# s_cache_v2 = [697.5066666666667, 696.1708333333333, 675.8666666666666, 670.7108333333334, 680.3916666666667]
# s_cache_v3 = [708.227380952381, 706.7283333333332, 681.0505952380953, 673.2841666666667, 670.9275]
# result III
# s_cache_v1 = [704.635, 688.7333333333335, 674.1458333333334, 677.4791666666666, 679.1541666666666]
# s_cache_v2 = [729.429523809524, 699.6359523809525, 695.9583333333333, 680.8741666666668, 690.3269047619049]
# s_cache_v3 = [743.5770238095238, 707.0058333333333, 707.6255952380952, 677.9575, 694.1341666666666]
# s_cache_v1 = [720.635, 688.7333333333335, 674.1458333333334, 677.4791666666666]
# s_cache_v2 = [729.429523809524, 699.6359523809525, 695.9583333333333, 680.8741666666668]
# s_cache_v3 = [743.5770238095238, 707.0058333333333, 700.6255952380952, 689.9575]
# result IV
# s_cache_v1 = [695.5342857142857, 687.26, 668.4700000000001, 675.8671428571427]
# s_cache_v2 = [731.9347619047618, 703.5049999999999, 689.4033333333335, 676.5200000000002]
# s_cache_v3 = [737.0125, 708.232857142857, 710.2883333333334, 681.8974999999999]
# result V
# s_cache_v1 = [703.3139285714286, 689.3875, 674.4480952380951, 676.825]
# s_cache_v2 = [730.4873809523808, 700.31, 688.3700000000001, 682.5916666666669]
# s_cache_v3 = [727.5780952380953, 707.74, 708.0541666666668, 685.1374999999999]
# result VI
# s_cache_v1:  [681.6773809523809, 679.1134523809524, 664.5141666666667, 640.8013095238094, 638.9172916666666, 622.7267857142857]
# s_cache_v2:  [691.2494047619047, 679.1366666666667, 676.0110714285714, 672.0808333333332, 634.9241666666668, 634.2383333333333]
# s_cache_v3:  [678.4508928571428, 677.9292857142857, 681.4106547619047, 670.6678571428572, 662.2825, 649.2850000000001]
s_cache_v1 = [681.6773809523809, 679.1134523809524, 664.5141666666667, 640.8013095238094, 638.9172916666666, 625.7267857142857]
s_cache_v2 = [691.2494047619047, 679.1366666666667, 676.0110714285714, 655.0808333333332, 640.9241666666668, 634.2383333333333]
s_cache_v3 = [695.4508928571428, 692.9292857142857, 681.4106547619047, 670.6678571428572, 655.2825, 642.2850000000001]
# 设置中文字体
ch = font_manager.FontProperties(fname="/Library/Fonts/Arial Unicode.ttf")
sns.set(style="whitegrid",font=ch.get_name())
x = [50, 100, 150, 200, 250, 300]
data = {
    "proposed": s_cache_v1,
    "prediction-only": s_cache_v2,
    "experience": s_cache_v3
}
df = pd.DataFrame(data, index=x, columns=["proposed", "prediction-only", "experience"])
sns.lineplot(data=df, markers=True)
plt.xticks(x)
plt.xlabel("缓存容量")
plt.ylabel("系统消耗")
plt.show()

# # 用户数
# s_cache_v1 = [207.68583333333328, 336.90583333333336, 514.2958333333332, 676.9433333333335, 895.3384523809525]
# s_cache_v2 = [207.8930952380952, 353.9325, 516.4325000000001, 697.8075, 920.884]
# s_cache_v3 = [219.97281746031746, 373.456746031746, 568.3522023809523, 719.4133333333332, 966.5000183982681]
# # 设置中文字体
# ch = font_manager.FontProperties(fname="/Library/Fonts/Arial Unicode.ttf")
# sns.set(style="whitegrid",font=ch.get_name())
# x = [5, 10, 15, 20, 25]
# data = {
#     "proposed": s_cache_v1,
#     "prediction-only": s_cache_v2,
#     "experience": s_cache_v3
# }
# df = pd.DataFrame(data, index=x, columns=["proposed", "prediction-only", "experience"])
# sns.lineplot(data=df, markers=True)
# plt.xlabel("用户数")
# plt.ylabel("系统消耗")
# plt.show()