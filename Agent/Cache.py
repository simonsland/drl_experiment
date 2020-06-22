import numpy as np
import math


class Cache:
    def __init__(self, cache_capacity, task, task_t, user_n):
        self.cache_capacity = cache_capacity
        self.task_t = task_t
        self.user_n = user_n
        self.task = task
        self.offload = np.zeros(user_n, dtype=int)
        self.cache = np.zeros(task_t, dtype=int)

    def cache_update_size(self, offload_v, request_v):
        # 候选任务集
        # candidate = np.zeros(self.task_t, dtype=int)
        # for j in range(self.task_t):
        #     if self.cache[j] == 1:
        #         candidate[j] = 1
        # for i in range(len(offload_v)):
        #     candidate[offload_v[i]] = 1
        # 创建任务-期望收益字典
        dict = {}
        # max_upload = max(self.task[:, 0])
        # max_cpu = max(self.task[:, 1])
        for j in range(self.task_t):
            # if candidate[j] == 1:
            dict[j] = np.sum(request_v == j) * math.log(1 + self.task[j][0]/30 + self.task[j][1]/100)
        # 按照期望收益对任务进行排序
        sort = sorted(dict, key=lambda k: dict[k], reverse=True)
        print("sort_v1: ", sort)
        # 缓存期望收益排序靠前的几个任务
        idx = 0
        length = len(sort)
        cache_v = []
        capacity_r = self.cache_capacity  # 剩余的缓存容量
        while idx < length:
            if capacity_r >= self.task[sort[idx]][0]:
                cache_v.append(sort[idx])
                capacity_r -= self.task[sort[idx]][0]
            idx += 1
        self.cache = np.zeros(self.task_t, dtype=int)
        for elem in cache_v:
            self.cache[elem] = 1

    # 流行度优先的缓存算法
    def cache_update(self, offload_v, request_v):
        # 候选任务集
        # candidate = np.zeros(self.task_t, dtype=int)
        # for j in range(self.task_t):
        #     if self.cache[j] == 1:
        #         candidate[j] = 1
        # for i in range(len(offload_v)):
        #     candidate[offload_v[i]] = 1
        # 创建任务-流行度(不考虑任务量)字典
        dict = {}
        for j in range(self.task_t):
            # if candidate[j] == 1:
            dict[j] = np.sum(request_v == j)
        # 按照流行度对任务进行排序
        sort = sorted(dict, key=lambda k: dict[k], reverse=True)
        print("sort_v2: ", sort)
        # 缓存流行度排序靠前的几个任务
        idx = 0
        length = len(sort)
        cache_v = []
        capacity_r = self.cache_capacity  # 剩余的缓存容量
        while idx < length:
            if capacity_r >= self.task[sort[idx]][0]:
                cache_v.append(sort[idx])
                capacity_r -= self.task[sort[idx]][0]
            idx += 1
        self.cache = np.zeros(self.task_t, dtype=int)
        for elem in cache_v:
            self.cache[elem] = 1

    # 基于经验流行度的缓存算法
    def cache_update_experience(self, offload_v, popularity_v):
        # 候选任务集
        # candidate = np.zeros(self.task_t, dtype=int)
        # for j in range(self.task_t):
        #     if self.cache[j] == 1:
        #         candidate[j] = 1
        # for i in range(len(offload_v)):
        #     candidate[offload_v[i]] = 1
        # 创建任务-流行度字典
        dict = {}
        for j in range(self.task_t):
            # if candidate[j] == 1:
            dict[j] = popularity_v[j]
        # 按照流行度对任务进行排序
        sort = sorted(dict, key=lambda k: dict[k], reverse=True)
        # 缓存流行度排序靠前的几个任务
        idx = 0
        length = len(sort)
        cache_v = []
        capacity_r = self.cache_capacity  # 剩余的缓存容量
        while idx < length:
            if capacity_r >= self.task[sort[idx]][0]:
                cache_v.append(sort[idx])
                capacity_r -= self.task[sort[idx]][0]
            idx += 1
        self.cache = np.zeros(self.task_t, dtype=int)
        for elem in cache_v:
            self.cache[elem] = 1





