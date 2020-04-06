#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: youxinghui

环境模型-1：单时隙卸载模型
"""

import numpy as np
import math


# one-step action model
class OffloadingV1:
    def __init__(self, user_n, task_t, f_mec, f_unit, f_ue, p_ue, bandwidth, w1, w2, reward_function='v_1'):
        self.user_n = user_n
        self.task_t = task_t
        self.f_mec = f_mec  # GHz/sec
        self.f_unit = f_unit  # 最小可分配单元(GHz/sec),但是如果是第一次分配，则至少保证能获得ue_capacity+mec_uni的资源
        self.f_ue = f_ue  # 用户设备计算容量(GHz/sec)，满足正态分布, 长度为user_n的向量
        self.p_ue = p_ue  # 用户设备的传输功率(W)，满足正态分布，长度为user_n的向量
        self.bandwidth = bandwidth  # 数据传输延时应该小一些，否则发挥不出卸载的优势(MHz)
        self.w1 = w1  # 时延权重，长度为user_n的向量
        self.w2 = w2  # 能耗权重，长度为user_n的向量
        self.task = self.task_generate()  # 产生网络中的任务列表
        self.request = self.request_generate()  # 产生用户请求任务
        self.reward_function = reward_function  # 奖励函数

        # 观测空间: [计算资源分配情况]
        self.observation_n = self.user_n
        self.observation = np.zeros(self.observation_n, dtype=float)  # 观测
        self.action_n = self.user_n  # 动作空间

        self.current_consumption = 0  # 当前系统状态的消耗
        self.offload_user = 0  # 当前卸载人数
        self.consumption_record = []  # 系统消耗变化
        self.offload_record = []  # 系统卸载决策覆盖人数的变化

    def reset(self):
        # 重置环境状态
        self.observation = np.zeros(self.user_n, dtype=float)  # 用户资源分配初始化
        self.current_consumption = self.weight_sum()
        self.offload_user = 1
        return self.observation

    def step(self, action):
        done = False
        # 改变环境状态
        # 增加资源分配
        reward = 0
        done = False
        offload_user = self.offload_user
        if math.isclose(self.observation[action], 0):
            offload_user += 1
        self.observation[action] += self.f_unit
        while sum(self.observation) <= self.f_mec:
            consumption = self.weight_sum()  # 计算系统消耗
            if consumption < self.current_consumption:
                reward = self.current_consumption - consumption
                if self.reward_function == 'v_5' and self.offload_user < offload_user:  # 辅助reward
                    reward += 5
                self.current_consumption = consumption
                self.offload_user = offload_user
                break
            self.observation[action] += self.f_unit
        if sum(self.observation) > self.f_mec:
            done = True
            self.observation[action] -= self.f_unit  # 还原一次操作
            self.offload_record.append(self.offload_user)
            self.consumption_record.append(self.current_consumption)
        return self.observation, reward, done

    # 计算某状态的时延和功耗加权和
    def weight_sum(self):
        consumption = 0
        r_up = self.bandwidth / sum(self.observation > 0) if sum(self.observation > 0) else 0
        for i in range(0, self.user_n):
            s_i = self.request[i*2]
            w_i = self.request[i*2+1]
            if self.observation[i] > 0:  # 卸载计算
                delay = s_i / r_up + w_i / self.observation[i]
                energy = self.p_ue[i] * (s_i / r_up)
                consumption += self.w1[i] * delay + self.w2[i] * energy
            else:  # 本地计算
                delay = w_i / self.f_ue[i]
                energy = w_i * ((self.f_ue[i])**2)
                consumption += self.w1[i] * delay + self.w2[i] * energy
        return consumption

    # 奖励函数
    def reward_calc(self, consumption):
        reward = 0
        if self.reward_function == 'v_1':
            reward = self.current_consumption - consumption
        elif self.reward_function == 'v_2':
            reward = self.current_consumption - consumption
            if reward > 0:
                reward = 1
            elif reward < 0:
                reward = -1
        elif self.reward_function == 'v_3':
            reward = (self.current_consumption - consumption) / self.current_consumption
        elif self.reward_function == 'v_4':
            reward = 2 ** ((self.current_consumption - consumption) / 5)
        return reward

    # 产生用户的任务请求数据
    def request_generate(self):
        choice_arr = np.random.choice(np.random.randint(0, self.task_t, self.task_t, dtype=int), size=self.user_n,
                                replace=True, p=None)
        request = np.zeros([self.user_n, 2], dtype=float)
        user_i = 0
        for choice in choice_arr:
            request[user_i, :] = self.task[choice, :]
            user_i += 1
        return request.flatten()

    # 产生环境中的任务数据
    def task_generate(self):
        task = np.zeros([2, self.task_t], dtype=float)
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

    def reset_consumption_record(self):
        self.consumption_record = []

    def reset_reward_function(self, reward_function):
        self.reward_function = reward_function

    def reset_f_unit(self, f_unit):
        self.f_unit = f_unit

    def reset_offload_record(self):
        self.offload_record = []

    def plot_consumption(self):
        import matplotlib.pyplot as plt
        # 对结果进行平滑处理，每N个episode平均一次
        length = len(self.consumption_record)
        n = 20
        for i in range(0, int(length/n)):
            smooth = sum(self.consumption_record[n*i:n*(i+1)]) / n
            self.consumption_record[n*i:n*(i+1)] = np.array([smooth]*n)
        plt.plot(np.arange(length), self.consumption_record)
        plt.ylabel('consumption')
        plt.xlabel('training steps')
        plt.show()
