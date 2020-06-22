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
# 用户数为20的卸载收敛性分析
    w1 = np.array([1]*20)
    w2 = np.array([0]*20)
    f_ue = np.array([2]*20)
    p_ue = np.array([0.3]*20)
    gen = RequestGenerator(task_t=40, user_n=20)
    # 查看分配的均衡情况
    convergence = OffloadingV1(user_n=20, request=gen.request, f_mec=40, f_unit=1, f_ue=f_ue, p_ue=p_ue, bandwidth=20, w1=w1, w2=w2)
    dqn = DeepQNetwork(convergence.action_n, convergence.observation_n, learning_rate=0.001, reward_decay=0.9,
                   e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
    train(convergence, dqn, 1, 2000, 500)
    loss = dqn.loss_record
    # # 损失分析
    # # 设置中文字体
    # ch = font_manager.FontProperties(fname="/Library/Fonts/Arial Unicode.ttf")
    # sns.set(style="whitegrid", font=ch.get_name())
    # data = {
    #     "迭代次数": np.linspace(0, len(loss), len(loss)),
    #     "损失变化": loss
    # }
    # df = pd.DataFrame(data)
    # sns.lineplot(x="迭代次数", y="损失变化", data=df)
    # plt.show()
    # 累积奖励分析
    reward = smooth_average(convergence.consumption_record, 10)
    # 设置中文字体
    ch = font_manager.FontProperties(fname="/Library/Fonts/Arial Unicode.ttf")
    sns.set(style="whitegrid", font=ch.get_name())
    data = {
        "迭代决策回合": np.linspace(0, len(reward), len(reward)),
        "累积奖励变化": reward
    }
    df = pd.DataFrame(data)
    sns.lineplot(x="迭代决策回合", y="累积奖励变化", data=df)
    plt.show()