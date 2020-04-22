import numpy as np

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math


class RequestGenerator:
    def __init__(self, task_t, user_n):
        self.task_t = task_t
        self.user_n = user_n
        self.network_tasks = self.task_generate()
        self.request = self.request_generate()

    # 产生环境中的任务数据
    def task_generate(self):
        task = np.zeros([2, self.task_t], dtype=int)
        # 任务上传数据量
        upload_data = np.random.normal(30, 5, self.task_t)
        while sum(upload_data < 0):  # 确保生成的数据中不包含负值
            upload_data = np.random.normal(30, 5, self.task_t)
        # 任务计算量
        cpu_cycle = np.random.normal(100, 20, self.task_t)
        while sum(cpu_cycle < 0):
            cpu_cycle = np.random.normal(100, 20, self.task_t)
        task[0, :] = upload_data
        task[1, :] = cpu_cycle
        return np.transpose(task)

    # 产生用户的任务请求数据
    def request_generate(self):
        choice_arr = np.random.choice(np.random.randint(0, self.task_t, self.task_t, dtype=int), size=self.user_n,
                                      replace=True, p=None)
        request = np.zeros([self.user_n, 2], dtype=float)
        user_i = 0
        for choice in choice_arr:
            request[user_i, :] = self.network_tasks[choice, :]
            user_i += 1
        return request.flatten()


class LSTMRequestGenerator:
    def __init__(self, task_t, user_n, time_slot):
        self.task_t = task_t
        self.user_n = user_n
        self.time_slot = time_slot
        self.task = self.task_generate()
        self.task_cnt, self.request_v = self.request_v_generator()
        self.request = self.request_generator()
        self.popularity = self.popularity_generator()

    # 产生环境中的任务数据
    def task_generate(self, data_size=30, sigma=15):
        task = np.zeros([2, self.task_t], dtype=int)
        # 任务上传数据量
        upload_data = np.random.normal(data_size, sigma, self.task_t)
        while sum(upload_data < 0):  # 确保生成的数据中不包含负值
            upload_data = np.random.normal(data_size, sigma, self.task_t)
        print("传输量: ", upload_data)
        # 任务计算量
        cpu_cycle = np.random.normal(100, 20, self.task_t)
        while sum(cpu_cycle < 0):
            cpu_cycle = np.random.normal(100, 20, self.task_t)
        print("计算量: ", cpu_cycle)
        task[0, :] = upload_data
        task[1, :] = cpu_cycle
        return np.transpose(task)

    def request_v_generator(self):
        x = np.linspace(0, self.time_slot, self.time_slot)
        cycle = self.time_slot / 10
        c = 2 * math.pi / cycle
        f = np.zeros([self.task_t, self.time_slot], dtype=float)
        r = np.zeros([self.task_t, self.time_slot], dtype=int)
        request = np.zeros([self.user_n, self.time_slot], dtype=int)
        for i in range(self.task_t):
            y = 0.3 * np.sin(c*(x+i*cycle/self.task_t)) + 0.5
            f[i, :] = y
            # plt.plot(x, y)
        for j in range(self.time_slot):
            for i in range(self.task_t):
                r[i, j] = round(f[i, j] / sum(f[:, j]) * self.user_n)
        for j in range(self.time_slot):
            idx = 0
            for i in range(self.task_t):
                if r[i, j] > 0:
                    s = idx
                    e = idx + r[i, j] if (idx + r[i, j]) <= self.user_n else self.user_n
                    request[s:e, j] = i
                    idx = e
        return np.transpose(r), np.transpose(request)

    def request_generator(self):
        request = np.zeros([self.time_slot, 2*self.user_n], dtype=int)
        for j in range(self.time_slot):
            request_v = self.request_v[j, :]
            for i in range(self.user_n):
                request[j, i*2:(i+1)*2] = self.task[request_v[i], :]
        return request

    def popularity_generator(self):
        cnt_sum = np.zeros([self.time_slot, self.task_t], dtype=int)
        for i in range(self.time_slot):
            for j in range(self.task_t):
                cnt_sum[i, j] = sum(self.task_cnt[0:i, j])
        popularity = np.zeros([self.time_slot, self.task_t], dtype=float)
        for i in range(self.time_slot):
            for j in range(self.task_t):
                popularity[i, j] = cnt_sum[i, j] / sum(cnt_sum[i, :])
        return popularity

    def request_v_t(self, time):
        return self.request_v[time, :]

    def request_t(self, time):
        return self.request[time, :]

    def popularity_t(self, time):
        return self.popularity[time, :]
