#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 17:29:20 2020

@author: youxinghui
"""

import numpy as np


class Environment:
    def __init__(self):
        self.user_n = 4
        self.task_t = 6
        self.mec_capacity = 5
        self.cache_capacity = 3
        self.ue_capacity = 1
        self.bandwidth = 5
        self.task = self.task_generate()  # 产生网络中的任务列表
        # 观测空间: [用户请求任务序列, 缓存状态（用十进制表示)]
        self.observation_n = self.user_n + 1
        self.observation = np.zeros(self.observation_n, dtype=int)  # 观测
        self.action_space = self.action_generate()  # 动作空间
        self.action_n = np.shape(self.action_space)[0]
        self.action = -1  # 当前动作
        self.step_cnt = 0  # 步数统计

    def reset(self):
        # 初初始化环境观测
        self.update_observation(cache=0)
        return self.observation
        
    def step(self, action):
        _action = self.int_to_binary_array(int(action))
        x = _action[0:self.user_n]
        y = _action[self.user_n:]
        reward = self.reward_caculate(x, y)
        done = False
        if self.step_cnt == 50:
            done = True
            self.step_cnt = 0
        # 改变环境
        self.update_observation(cache=self.binary_array_to_int(y))
        next_observation = self.observation
        self.step_cnt += 1
        return next_observation, reward, done
    
    def reward_caculate(self, x, y):
        request = self.observation[0:self.user_n]
        # 本地计算耗时
        local_t_sum = 0
        for i in range(self.user_n):
            t_local = self.task[request[i], 1] / self.ue_capacity
            local_t_sum += t_local
        # 卸载计算耗时
        offload_t_sum = 0
        if sum(x == 1) > 0:
            transmit_rate = self.bandwidth / sum(x == 1)  # 均分通信资源
            cpu = self.mec_capacity / sum(x == 1)  # 均分CPU资源
            for i in range(self.user_n):
                t_offload = self.task[request[i], 0] / transmit_rate + self.task[request[i], 1] / cpu  # 卸载计算耗时
                offload_t_sum += (1 - y[i]) * ((1 - x[i]) * t_local + x[i] * t_offload)
        print('local: ', local_t_sum, ' offload: ', offload_t_sum)
        reward = (local_t_sum - offload_t_sum) / local_t_sum
        return reward

    def task_generate(self):
        tasks = np.zeros([2, self.task_t], dtype=float)
        # 任务上传数据量
        upload_data = np.random.normal(2, 1, self.task_t)
        while sum(upload_data < 0): #确保生成的数据中不包含负值
            upload_data = np.random.normal(2, 1, self.task_t)
        # 任务计算量
        cpu_cycle = np.random.normal(2, 1, self.task_t)
        while sum(cpu_cycle < 0):
            cpu_cycle = np.random.normal(2, 1, self.task_t)
        tasks[0, :] = upload_data
        tasks[1, :] = cpu_cycle
        return np.transpose(tasks)

    def update_observation(self, cache):
        self.observation[0:self.user_n] = self.request_generate()
        self.observation[self.user_n] = cache

    def request_generate(self):

        return np.random.choice(np.random.randint(0, self.task_t, self.task_t, dtype=int), size=self.user_n, replace=True, p=None)

    def action_generate(self):
        actions = []
        for i in range(2**(self.user_n + self.task_t)):
            action = self.int_to_binary_array(i)
            if self.valid_action(action):
                actions.append(action)
        return actions

    def valid_action(self, action):
        is_valid = True
        # 判断缓存是否符合容量要求
        a_cache = action[self.user_n:]
        if sum(a_cache == 1) > self.cache_capacity:
            is_valid = False
        return is_valid

    def int_to_binary_array(self, n):
        arr_len = self.user_n + self.task_t
        binary_arr = np.zeros(arr_len, dtype=int)
        for i in range(arr_len):
            binary_arr[arr_len - 1 - i] = n & 1
            n >>= 1
        return binary_arr

    def binary_array_to_int(self, arr):
        n = 0
        for i in range(0, len(arr)):
            n = n * 2 + arr[i]
        return n
