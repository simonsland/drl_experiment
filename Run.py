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
            # print("episode begin")
            while True:
                # RL choose action based on observation
                action = agent.choose_action(observation)
                # RL take action and get next observation and reward
                observation_, reward, done = env.step(action)
                # print(observation, action, reward)
                agent.store_transition(observation, action, reward, observation_)
                if step > pre_train:  # 生成预训练数据
                    agent.learn()
                # swap observation
                observation = observation_
                # break while loop when end of this episode
                if done:
                    break
                step += 1
            # print("episode end")
    # end of game


def run(env, agent, episode):
    for epi in range(episode):
        observation = env.reset()
        # print("test episode begin")
        while True:
            # RL choose action based on observation
            action = agent.choose_action(observation)
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            # print(observation, action, reward)
            # swap observation
            observation = observation_
            # break while loop when end of this episode
            if done:
                break
        # print("test episode end")


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
# 用户数为[8, 12, 16，20]
    # 系统消耗
    s_local = []
    s_proposed = []
    # s_jtoba = []
    s_random = []
    f_mec = 20
    bandwidth = 20
    for user_n in [8, 12, 16, 20]:
        w1 = np.array([1] * user_n)
        w2 = np.array([0] * user_n)
        f_ue = np.array([2] * user_n)
        p_ue = np.array([0.3] * user_n)
        gen = RequestGenerator(task_t=10, user_n=user_n)
        # 算法I-Proposed Algorithm
        proposed = OffloadingV1(request=gen.request, user_n=user_n, f_mec=f_mec, f_unit=1,
                             f_ue=f_ue, p_ue=p_ue, bandwidth=bandwidth, w1=w1, w2=w2, reward_function="Proposed")
        dqn_v1 = DeepQNetwork(proposed.action_n, proposed.observation_n, learning_rate=0.001, reward_decay=0.9,
                              e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
        train(proposed, dqn_v1, 1, 500, 500)
        # 提取最后10次结果中的统计量
        s_local.append(proposed.consumption_record[0])
        consumption = smooth_average(proposed.consumption_record, 10)
        s_proposed.append(extract_last_n_min(consumption, 10))
        # # 算法II-SAQ-learning
        # saq_learning = OffloadingV1(request=gen.request, user_n=user_n, f_mec=f_mec, f_unit=1,
        #                               f_ue=f_ue, p_ue=p_ue, bandwidth=bandwidth, w1=w1, w2=w2, reward_function="SAQ-learning")
        # dqn_v2 = DeepQNetwork(saq_learning.action_n, saq_learning.observation_n, learning_rate=0.001, reward_decay=0.9,
        #                       e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
        # train(saq_learning, dqn_v2, 1, 1000, 500)
        # consumption = smooth_average(saq_learning.consumption_record, 10)
        # s_saq.append(extract_last_n_min(consumption, 10))
        # 算法III-JTOBA
        # JTOBA = OffloadingV1(request=gen.request, user_n=user_n, f_mec=f_mec, f_unit=1,
        #                             f_ue=f_ue, p_ue=p_ue, bandwidth=bandwidth, w1=w1, w2=w2, reward_function="JTOBA")
        # dqn_v3 = DeepQNetwork(JTOBA.action_n, JTOBA.observation_n, learning_rate=0.001, reward_decay=0.9,
        #                       e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
        # train(JTOBA, dqn_v3, 1, 500, 500)
        # consumption = smooth_average(JTOBA.consumption_record, 10)
        # s_jtoba.append(extract_last_n_average(consumption, 100))
        # 算法IV-Random
        random = OffloadingV1(request=gen.request, user_n=user_n, f_mec=f_mec, f_unit=1,
                                f_ue=f_ue, p_ue=p_ue, bandwidth=bandwidth, w1=w1, w2=w2)
        random_agent = Random(random.action_n)
        run(random, random_agent, 500)  # random算法无须训练
        consumption = smooth_average(random.consumption_record, 10)
        s_random.append(extract_last_n_average(consumption, 10))
        print("Proposed:", s_proposed)
        print("Local:", s_local)
        # print("Proposed:", s_jtoba)
        print("Random:", s_random)
    # 设置中文字体
    ch = font_manager.FontProperties(fname="/Library/Fonts/Arial Unicode.ttf")
    sns.set(style='whitegrid', font=ch.get_name())
    data = {
        "Proposed": s_proposed,
        "Local": s_local,
        # "JTOBA": s_jtoba,
        "Random": s_random,
    }
    print("Proposed:", s_proposed)
    print("Local:", s_local)
    # print("Proposed:", s_jtoba)
    print("Random:", s_random)
    df = pd.DataFrame(data, index=[8, 12, 16, 20], columns=["Proposed", "Local", "Random"])
    sns.lineplot(data=df, markers=True)
    plt.xlabel("小区用户数")
    plt.ylabel("系统消耗")
    plt.show()
