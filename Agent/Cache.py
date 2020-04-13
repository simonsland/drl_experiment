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
        candidate = np.zeros(self.task_t, dtype=int)
        for j in range(self.task_t):
            if self.cache[j] == 1:
                candidate[j] = 1
        for i in range(len(offload_v)):
            candidate[offload_v[i]] = 1
        # 创建任务-期望收益字典
        dict = {}
        for j in range(self.task_t):
            if candidate[j] == 1:
                dict[j] = self.task[j][0] * np.sum(request_v == j)
        # 按照期望收益对任务进行排序
        sort = sorted(dict, key=lambda k: dict[k], reverse=True)
        # 通过双指针算法得到满足缓存容量情况下的最大任务序列
        l = 0
        r = 0
        length = len(sort)
        max_benefit = 0
        benefit_tmp = 0
        cache_v = []
        cache_tmp = []
        capacity_r = self.cache_capacity  # 剩余的缓存容量
        while l < length:
            while r < length and capacity_r >= self.task[sort[r]][0]:
                cache_tmp.append(sort[r])
                benefit_tmp += dict[sort[r]]
                capacity_r -= self.task[sort[r]][0]
                r += 1
            if benefit_tmp > max_benefit or (math.isclose(benefit_tmp, max_benefit) and len(cache_tmp) > len(cache_v)):
                max_benefit = benefit_tmp
                cache_v = cache_tmp.copy()
            capacity_r += self.task[sort[l]][0]
            cache_tmp.remove(sort[l])
            benefit_tmp -= dict[sort[l]]
            l += 1
        self.cache = np.zeros(self.task_t, dtype=int)
        for elem in cache_v:
            self.cache[elem-1] = 1

    # 流行度优先的缓存算法
    def cache_update(self, offload_v, request_v):
        # 候选任务集
        candidate = np.zeros(self.task_t, dtype=int)
        for j in range(self.task_t):
            if self.cache[j] == 1:
                candidate[j] = 1
        for i in range(len(offload_v)):
            candidate[offload_v[i]] = 1
        # 创建任务-流行度(不考虑任务量)字典
        dict = {}
        for j in range(self.task_t):
            if candidate[j] == 1:
                dict[j] = np.sum(request_v == j)
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
            else:
                break
        self.cache = np.zeros(self.task_t, dtype=int)
        for elem in cache_v:
            self.cache[elem-1] = 1

    # 基于经验流行度的缓存算法
    def cache_update_experience(self, offload_v, popularity_v):
        # 候选任务集
        candidate = np.zeros(self.task_t, dtype=int)
        for j in range(self.task_t):
            if self.cache[j] == 1:
                candidate[j] = 1
        for i in range(len(offload_v)):
            candidate[offload_v[i]] = 1
        # 创建任务-流行度字典
        dict = {}
        for j in range(self.task_t):
            if candidate[j] == 1:
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
            else:
                break
        self.cache = np.zeros(self.task_t, dtype=int)
        for elem in cache_v:
            self.cache[elem - 1] = 1





