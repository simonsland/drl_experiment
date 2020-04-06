from Environment.Offloading_v_2 import OffloadingV2
from Environment.Offloading_v_1 import OffloadingV1
from Agent.DQN import DeepQNetwork
from Agent.Random import Random
import matplotlib.pyplot as plt
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


# 对比实验
def compare(env, rl_agent, random_agent, time_slot, episode):
    loc, rand, rl = [], [], []
    for slot in range(time_slot):
        env.next_time_slot_init()
        # RL
        run(env, rl_agent, episode)
        loc.append(env.consumption_record_local)
        rl.append(env.consumption_record)
        env.reset_consumption_record()
        # Random
        run(env, random_agent, episode)
        rand.append(env.consumption_record)
        env.reset_consumption_record()
    return smooth(sum(loc, []), episode), smooth(sum(rand, []), episode), smooth(sum(rl, []), episode)


def smooth(record, n):
    # 对结果进行平滑处理
    res = []
    length = len(record)
    for i in range(int(length/n)):
        average = sum(record[n*i:n*(i+1)]) / n
        res.append(average)
    return record


def extract_last_n(record, n):
    # 取最后n次结果的平均值
    result = record[-1*n:]
    average = sum(result)/n
    return [average]*n


# 时延对比实验
if __name__ == "__main__":
    # 用户数为20的卸载收敛性分析
    w1 = np.array([1]*20)
    w2 = np.array([0]*20)
    f_ue = np.array([2]*20)
    p_ue = np.array([0.3]*20)
    # 查看分配的均衡情况
    convergence = OffloadingV1(user_n=20, task_t=40, f_mec=40, f_unit=1, f_ue=f_ue, p_ue=p_ue, bandwidth=20, w1=w1, w2=w2)
    dqn = DeepQNetwork(convergence.action_n, convergence.observation_n, learning_rate=0.001, reward_decay=0.9,
                   e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
    train(convergence, dqn, 1, 2000, 500)
    # 损失分析

    # 回报分析
