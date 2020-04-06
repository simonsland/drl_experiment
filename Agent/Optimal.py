import numpy as np
import math


class OptimalPolicy:
    def __init__(self, request, observation_n, action_n, f_mec, f_unit, f_ue, bandwidth):
        self.request = request
        self.observation_n = observation_n
        self.optimal_decision = np.zeros(observation_n, dtype=float)
        self.optimal_delay = 1000
        self.action_n = action_n
        self.f_mec = f_mec
        self.f_unit = f_unit
        self.f_ue = f_ue
        self.bandwidth = bandwidth

    def run(self):
        decision = np.zeros(self.observation_n, dtype=float)
        self.find_optimal_decision(decision, idx=0)

    def find_optimal_decision(self, decision, idx):
        print(decision)
        if idx == self.observation_n:  # 资源分配完成
            # 计算该decision下的delay
            delay = 0
            if sum(decision > 0):
                trans_rate = self.bandwidth / sum(decision > 0)
                for i in range(0, self.observation_n):
                    if decision[i] > 0:
                        delay += self.request[i * 2] / trans_rate + self.request[i * 2 + 1] / decision[i]
                    else:
                        delay += self.request[i * 2 + 1] / self.f_ue
            else:
                for i in range(0, self.observation_n):
                    delay += self.request[i * 2 + 1] / self.f_ue
            if delay < self.optimal_delay:
                self.optimal_delay = delay
                self.optimal_decision = decision
        else:  # 资源分配未完成
            mec_remain = self.f_mec - sum(decision)
            sub_step = (mec_remain - self.f_ue) / self.f_unit + 1
            for i in range(0, int(sub_step)):
                if i == 0:
                    next_ = decision.copy()
                    self.find_optimal_decision(next_, idx+1)
                else:
                    next_ = decision.copy()
                    next_[idx] = self.f_ue + i * self.f_unit
                    self.find_optimal_decision(next_, idx+1)

    def print(self):
        print("optimal decision: ", self.optimal_decision)
        print("optimal delay: ", self.optimal_delay)



