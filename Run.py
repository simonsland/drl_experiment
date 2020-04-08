from Environment.Offloading_v_2 import OffloadingV2
from Environment.Offloading_v_1 import OffloadingV1
from Environment.Offloading_v_3 import OffloadingV3
from Environment.Offloading_v_5 import OffloadingV5
from Environment.Request import RequestGenerator
from Environment.Request import LSTMRequestGenerator
from Agent.Cache import Cache
from Agent.DQN import DeepQNetwork
from Agent.Random import Random
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.font_manager as font_manager
import seaborn as sns
import numpy as np


# 训练agent
def train(env, agent, time_slots, episode, pre_train):
    step = 0
    for slot in range(time_slots):
        # initial observation
        if slot > 0:
            env.next_time_slot_init()
        for epi in range(episode):
            observation = env.reset()
            print("episode begin")
            while True:
                # RL choose action based on observation
                action = agent.choose_action(observation)
                # RL take action and get next observation and reward
                observation_, reward, done = env.step(action)
                print(observation, action, reward)
                agent.store_transition(observation, action, reward, observation_)
                if step > pre_train:  # 生成预训练数据
                    agent.learn()
                # swap observation
                observation = observation_
                # break while loop when end of this episode
                if done:
                    break
                step += 1
            print("episode end")
    # end of game


def run(env, agent, episode):
    for epi in range(episode):
        observation = env.reset()
        print("test episode begin")
        while True:
            # RL choose action based on observation
            action = agent.choose_action(observation)
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            print(observation, action, reward)
            # swap observation
            observation = observation_
            # break while loop when end of this episode
            if done:
                break
        print("test episode end")


def smooth_average(record, n):
    # 对结果进行平均处理
    res = []
    length = len(record)
    for i in range(int(length/n)):
        average = sum(record[n*i:n*(i+1)]) / n
        res.append(average)
    return res


def smooth_min(record, n):
    # 对结果进行平均处理
    res = []
    length = len(record)
    for i in range(int(length / n)):
        res.append(min(record[n * i:n * (i + 1)]))
    return res


def extract_last_n(record, n):
    return record[-1*n:]


def extract_last_n_min(record, n):
    # 取最后n次结果的最小值
    result = record[-1*n:]
    return min(result)


def extract_last_n_max(record, n):
    # 取最后n次结果的最大值
    result = record[-1*n:]
    return max(result)


def extract_last_n_average(record, n):
    # 取最后n次结果的平均值
    result = record[-1 * n:]
    average = sum(result) / n
    return average


if __name__ == "__main__":
    # 用户数为20的缓存辅助效果实验
    w1 = np.array([1]*20)
    w2 = np.array([0]*20)
    f_ue = np.array([2]*20)
    p_ue = np.array([0.3]*20)
    gen = LSTMRequestGenerator(task_t=10, user_n=20, time_slot=100)
    cache_v1 = Cache(cache_capacity=150, task=gen.task, task_t=10, user_n=20)  # 考虑流行度与任务量的缓存策略
    cache_v2 = Cache(cache_capacity=150, task=gen.task, task_t=10, user_n=20)  # 只考虑任务量的缓存策略
    # 10个时隙
    s_cache_v1 = []
    s_cache_v2 = []
    s_cache_none = []
    hit_rate_v1 = []
    hit_rate_v2 = []
    time_slot = 5
    for t in range(time_slot):
        print("时隙开始")
        request_v = gen.request_v_t(t)
        request = gen.request_t(t)
        # 考虑流行度与任务量的主动缓存策略
        cache_aside_v1 = OffloadingV5(request_v=request_v, request=request, cache=cache_v1.cache, user_n=20, f_mec=40, f_unit=1,
                              f_ue=f_ue, p_ue=p_ue, bandwidth=20, w1=w1, w2=w2)
        dqn_v1 = DeepQNetwork(cache_aside_v1.action_n, cache_aside_v1.observation_n, learning_rate=0.001, reward_decay=0.9,
                       e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
        train(cache_aside_v1, dqn_v1, 1, 10, 500)
        # 更新缓存
        # 最小消耗位置索引
        consumption = extract_last_n(cache_aside_v1.consumption_record, 10)
        idx = consumption.index(min(consumption))
        # 最小消耗卸载决策向量
        offload_v = cache_aside_v1.offload_v[idx]
        offload_t = []
        for i in range(20):
            if offload_v[i] > 0:
                offload_t.append(request_v[i])
        # 更新缓存
        cache_v1.cache_update_size(offload_t, gen.request_v_t(t+1))
        print("cache_v1: ", cache_v1.cache)
        s_cache_v1.append(consumption[idx])
        hit_rate_v1.append(extract_last_n_max(cache_aside_v1.hit_rate, 10))
        # 考虑流行度的主动缓存策略
        cache_aside_v2 = OffloadingV5(request_v=request_v, request=request, cache=cache_v2.cache, user_n=20, f_mec=40, f_unit=1,
                              f_ue=f_ue, p_ue=p_ue, bandwidth=20, w1=w1, w2=w2)
        dqn_v2 = DeepQNetwork(cache_aside_v2.action_n, cache_aside_v2.observation_n, learning_rate=0.001, reward_decay=0.9,
                       e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
        train(cache_aside_v2, dqn_v2, 1, 10, 500)
        # 更新缓存
        # 最小消耗位置索引
        consumption = extract_last_n(cache_aside_v2.consumption_record, 10)
        idx = consumption.index(min(consumption))
        # 最小消耗卸载决策向量
        offload_v = cache_aside_v2.offload_v[idx]
        offload_t = []
        for i in range(20):
            if offload_v[i] > 0:
                offload_t.append(request_v[i])
        # 更新缓存
        cache_v2.cache_update(offload_t, gen.request_v_t(t + 1))
        print("cache_v2: ", cache_v2.cache)
        s_cache_v2.append(consumption[idx])
        hit_rate_v2.append(extract_last_n_max(cache_aside_v2.hit_rate, 10))
        # # 无缓存
        # without_cache = OffloadingV1(request=request, user_n=20, f_mec=40, f_unit=1, f_ue=f_ue, p_ue=p_ue, bandwidth=20, w1=w1, w2=w2)
        # dqn_v3 = DeepQNetwork(without_cache.action_n, without_cache.observation_n, learning_rate=0.001, reward_decay=0.9,
        #                    e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
        # train(without_cache, dqn_v3, 1, 10, 500)
        # # 取最后10次中最小系统消耗
        # s_cache_none.append(extract_last_n_min(without_cache.consumption_record, 10))
    # 设置中文字体
    ch = font_manager.FontProperties(fname="/Library/Fonts/Arial Unicode.ttf")
    sns.set(font=ch.get_name())
    plt.subplot(1, 2, 1)
    data = {
        "缓存辅助": s_cache_v1,
        "流行度缓存": s_cache_v2,
        "无缓存": s_cache_none
    }
    df = pd.DataFrame(data, index=np.linspace(0, time_slot, time_slot), columns=["缓存辅助", "流行度缓存", "无缓存"])
    sns.lineplot(data=df)
    plt.xlabel("时间步")
    plt.ylabel("系统消耗")
    plt.subplot(1, 2, 2)
    data = {
        "缓存辅助": hit_rate_v1,
        "流行度缓存": hit_rate_v2,
    }
    df = pd.DataFrame(data, index=np.linspace(0, time_slot, time_slot), columns=["缓存辅助", "流行度缓存"])
    sns.lineplot(data=df)
    plt.xlabel("时间步")
    plt.ylabel("缓存命中率")
    plt.show()
