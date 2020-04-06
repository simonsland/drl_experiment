import numpy as np


class RequestGenerator:
    def __init__(self):
        self.network_tasks = self.task_generate()

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