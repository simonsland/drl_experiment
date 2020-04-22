# ----------------------收敛分析-----------------------
# 用户数为20的卸载收敛性分析
#     w1 = np.array([1]*20)
#     w2 = np.array([0]*20)
#     f_ue = np.array([2]*20)
#     p_ue = np.array([0.3]*20)
#     # 查看分配的均衡情况
#     convergence = OffloadingV1(user_n=20, task_t=40, f_mec=40, f_unit=1, f_ue=f_ue, p_ue=p_ue, bandwidth=20, w1=w1, w2=w2)
#     dqn = DeepQNetwork(convergence.action_n, convergence.observation_n, learning_rate=0.001, reward_decay=0.9,
#                    e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
#     train(convergence, dqn, 1, 2000, 500)
#     loss = dqn.loss_record
#     # 损失分析
#     # 设置中文字体
#     ch = font_manager.FontProperties(fname="/Library/Fonts/Arial Unicode.ttf")
#     sns.set(font=ch.get_name())
#     data = {
#         "迭代次数": np.linspace(0, len(loss), len(loss)),
#         "损失变化": loss
#     }
#     df = pd.DataFrame(data)
#     sns.lineplot(x="迭代次数", y="损失变化", data=df)
#     plt.show()
# 累积奖励分析
#     reward = smooth(convergence.consumption_record, 10)
#     # 设置中文字体
#     ch = font_manager.FontProperties(fname="/Library/Fonts/Arial Unicode.ttf")
#     sns.set(font=ch.get_name())
#     data = {
#         "迭代次数": np.linspace(0, len(reward), len(reward)),
#         "累积奖励变化": reward
#     }
#     df = pd.DataFrame(data)
#     sns.lineplot(x="迭代次数", y="累积奖励变化", data=df)
#     plt.show()

# -----------------------------卸载算法对比分析-----------------------------
# 用户数为20的动作设计效果对比实验
#     w1 = np.array([1]*20)
#     w2 = np.array([0]*20)
#     f_ue = np.array([2]*20)
#     p_ue = np.array([0.3]*20)
#     gen = RequestGenerator(task_t=10, user_n=20)
#     # 算法I-Proposed Alogorithm
#     proposed = OffloadingV1(request=gen.request, user_n=20, f_mec=40, f_unit=1,
#                                   f_ue=f_ue, p_ue=p_ue, bandwidth=20, w1=w1, w2=w2, reward_function="Proposed")
#     dqn_v1 = DeepQNetwork(proposed.action_n, proposed.observation_n, learning_rate=0.001, reward_decay=0.9,
#                    e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
#     train(proposed, dqn_v1, 1, 2000, 500)
#     consumption_v1 = smooth(proposed.consumption_record, 10)
#     # 算法II-SAQ-learning
#     saq_learning = OffloadingV1(request=gen.request, user_n=20, f_mec=40, f_unit=1,
#                                   f_ue=f_ue, p_ue=p_ue, bandwidth=20, w1=w1, w2=w2, reward_function="SAQ-learning")
#     dqn_v2 = DeepQNetwork(saq_learning.action_n, saq_learning.observation_n, learning_rate=0.001, reward_decay=0.9,
#                           e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
#     train(saq_learning, dqn_v2, 1, 2000, 500)
#     consumption_v2 = smooth(saq_learning.consumption_record, 10)
#     # 算法III-JTOBA
#     JTOBA = OffloadingV1(request=gen.request, user_n=20, f_mec=40, f_unit=1,
#                                 f_ue=f_ue, p_ue=p_ue, bandwidth=20, w1=w1, w2=w2, reward_function="JTOBA")
#     dqn_v3 = DeepQNetwork(JTOBA.action_n, JTOBA.observation_n, learning_rate=0.001, reward_decay=0.9,
#                           e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
#     train(JTOBA, dqn_v3, 1, 2000, 500)
#     consumption_v3 = smooth(JTOBA.consumption_record, 10)
#     # 算法IV-Random
#     random = OffloadingV1(request=gen.request, user_n=20, f_mec=40, f_unit=1,
#                             f_ue=f_ue, p_ue=p_ue, bandwidth=20, w1=w1, w2=w2)
#     random_agent = Random(random.action_n)
#     run(random, random_agent, 2000)  # random算法无须训练
#     consumption_v4 = smooth(random.consumption_record, 10)
#     # 设置中文字体
#     ch = font_manager.FontProperties(fname="/Library/Fonts/Arial Unicode.ttf")
#     sns.set(style='whitegrid', font=ch.get_name())
#     data = {
#         "Proposed": consumption_v1,
#         "SAQ-learning": consumption_v2,
#         "JTOBA": consumption_v3,
#         "Random": consumption_v4
#     }
#     df = pd.DataFrame(data, index=np.linspace(0, len(consumption_v1), len(consumption_v1)), columns=["Proposed", "SAQ-learning", "JTOBA", "Random"])
#     sns.lineplot(data=df)
#     plt.xlabel("迭代次数")
#     plt.ylabel("系统消耗")
#     plt.show()

# -----------------------------奖励辅助系统消耗变化---------------------------
# 用户数为20的辅助奖励效果实验
#     user_n = 20
#     w1 = np.array([1]*user_n)
#     w2 = np.array([0]*user_n)
#     f_ue = np.array([2]*user_n)
#     p_ue = np.array([0.3]*user_n)
#     gen = RequestGenerator(task_t=10, user_n=user_n)
#     # 算法I-Proposed with aside reward Alogorithm
#     aside = OffloadingV1(request=gen.request, user_n=user_n, f_mec=2*user_n, f_unit=1,
#                                   f_ue=f_ue, p_ue=p_ue, bandwidth=user_n, w1=w1, w2=w2, reward_function="Proposed_aside")
#     dqn_v1 = DeepQNetwork(aside.action_n, aside.observation_n, learning_rate=0.001, reward_decay=0.9,
#                    e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
#     train(aside, dqn_v1, 1, 2000, 500)
#     consumption_v1 = smooth_min(aside.consumption_record, 10)
#     # 算法II-without aside reward
#     without_aside = OffloadingV1(request=gen.request, user_n=user_n, f_mec=2*user_n, f_unit=1,
#                                   f_ue=f_ue, p_ue=p_ue, bandwidth=user_n, w1=w1, w2=w2, reward_function="Proposed")
#     dqn_v2 = DeepQNetwork(without_aside.action_n, without_aside.observation_n, learning_rate=0.001, reward_decay=0.9,
#                           e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
#     train(without_aside, dqn_v2, 1, 2000, 500)
#     consumption_v2 = smooth_min(without_aside.consumption_record, 10)
#     # 设置中文字体
#     ch = font_manager.FontProperties(fname="/Library/Fonts/Arial Unicode.ttf")
#     sns.set(font=ch.get_name())
#     data = {
#         "带辅助奖励": consumption_v1,
#         "无辅助奖励": consumption_v2,
#     }
#     df = pd.DataFrame(data, index=np.linspace(0, len(consumption_v1), len(consumption_v1)), columns=["带辅助奖励", "无辅助奖励"])
#     sns.lineplot(data=df)
#     plt.xlabel("迭代次数")
#     plt.ylabel("系统消耗")
#     plt.show()

# -----------------------------卸载覆盖率实验--------------------------------
# # 用户数为[8, 12, 16, 20]的辅助奖励效果实验
#     # 系统消耗
#     c_aside = []
#     c_none = []
#     # 卸载人数
#     u_aside = []
#     u_none = []
#     for user_n in [8, 12, 16, 20]:
#         w1 = np.array([1] * user_n)
#         w2 = np.array([0] * user_n)
#         f_ue = np.array([2] * user_n)
#         p_ue = np.array([0.3] * user_n)
#         gen = RequestGenerator(task_t=10, user_n=user_n)
#         # 算法I-Proposed with aside reward Alogorithm
#         aside = OffloadingV1(request=gen.request, user_n=user_n, f_mec=2*user_n, f_unit=1,
#                              f_ue=f_ue, p_ue=p_ue, bandwidth=user_n, w1=w1, w2=w2, reward_function="Proposed_aside")
#         dqn_v1 = DeepQNetwork(aside.action_n, aside.observation_n, learning_rate=0.001, reward_decay=0.9,
#                               e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
#         train(aside, dqn_v1, 1, 2000, 500)
#         # 提取最后10次结果中的统计量
#         consumption = smooth_average(aside.consumption_record, 10)
#         offload = smooth_average(aside.offload_record, 10)
#         c_aside.append(extract_last_n_min(consumption, 10))
#         u_aside.append(extract_last_n_max(offload, 10))
#         # 算法II-without aside reward
#         without_aside = OffloadingV1(request=gen.request, user_n=user_n, f_mec=2*user_n, f_unit=1,
#                                      f_ue=f_ue, p_ue=p_ue, bandwidth=user_n, w1=w1, w2=w2, reward_function="Proposed")
#         dqn_v2 = DeepQNetwork(without_aside.action_n, without_aside.observation_n, learning_rate=0.001,
#                               reward_decay=0.9,
#                               e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
#         train(without_aside, dqn_v2, 1, 2000, 500)
#         # 提取最后10次结果中的统计量
#         consumption = smooth_average(without_aside.consumption_record, 10)
#         offload = smooth_average(without_aside.offload_record, 10)
#         c_none.append(extract_last_n_min(consumption, 10))
#         u_none.append(extract_last_n_min(offload, 10))
#     # 设置中文字体
#     ch = font_manager.FontProperties(fname="/Library/Fonts/Arial Unicode.ttf")
#     sns.set(style='whitegrid', font=ch.get_name())
#     plt.subplot(1, 2, 1)
#     data = {
#         "带辅助奖励": c_aside,
#         "无辅助奖励": c_none,
#     }
#     df = pd.DataFrame(data, index=[8, 12, 16, 20], columns=["带辅助奖励", "无辅助奖励"])
#     sns.lineplot(data=df, markers=True)
#     plt.xlabel("小区用户数")
#     plt.ylabel("系统消耗")
#     # 条形图
#     plt.subplot(1, 2, 2)
#     label = [8, 12, 16, 20]
#     x = range(len(u_aside))
#     b1 = plt.bar(x=x, height=u_aside, width=0.4, label="带辅助奖励")
#     b2 = plt.bar(x=[i + 0.4 for i in x], height=u_none, width=0.4, label="无辅助奖励")
#     plt.legend([b1, b2], ["带辅助奖励", "无辅助奖励"], loc='upper left')
#     plt.xticks([i + 0.2 for i in x], label)
#     plt.xlabel("小区用户数")
#     plt.ylabel("卸载人数")
#     plt.show()

# ------------------------------用户数影响----------------------------------
# # 用户数为[8, 12, 16，20]
#     # 系统消耗
#     s_local = []
#     s_proposed = []
#     # s_jtoba = []
#     s_random = []
#     f_mec = 30
#     bandwidth = 20
#     for user_n in [8, 12, 16, 20]:
#         w1 = np.array([1] * user_n)
#         w2 = np.array([0] * user_n)
#         f_ue = np.array([2] * user_n)
#         p_ue = np.array([0.3] * user_n)
#         gen = RequestGenerator(task_t=10, user_n=user_n)
#         # 算法I-Proposed Algorithm
#         proposed = OffloadingV1(request=gen.request, user_n=user_n, f_mec=f_mec, f_unit=1,
#                              f_ue=f_ue, p_ue=p_ue, bandwidth=bandwidth, w1=w1, w2=w2, reward_function="Proposed")
#         dqn_v1 = DeepQNetwork(proposed.action_n, proposed.observation_n, learning_rate=0.001, reward_decay=0.9,
#                               e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
#         train(proposed, dqn_v1, 1, 500, 500)
#         # 提取最后10次结果中的统计量
#         s_local.append(proposed.consumption_record[0])
#         consumption = smooth_average(proposed.consumption_record, 10)
#         s_proposed.append(extract_last_n_min(consumption, 10))
#         # # 算法II-SAQ-learning
#         # saq_learning = OffloadingV1(request=gen.request, user_n=user_n, f_mec=f_mec, f_unit=1,
#         #                               f_ue=f_ue, p_ue=p_ue, bandwidth=bandwidth, w1=w1, w2=w2, reward_function="SAQ-learning")
#         # dqn_v2 = DeepQNetwork(saq_learning.action_n, saq_learning.observation_n, learning_rate=0.001, reward_decay=0.9,
#         #                       e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
#         # train(saq_learning, dqn_v2, 1, 1000, 500)
#         # consumption = smooth_average(saq_learning.consumption_record, 10)
#         # s_saq.append(extract_last_n_min(consumption, 10))
#         # 算法III-JTOBA
#         # JTOBA = OffloadingV1(request=gen.request, user_n=user_n, f_mec=f_mec, f_unit=1,
#         #                             f_ue=f_ue, p_ue=p_ue, bandwidth=bandwidth, w1=w1, w2=w2, reward_function="JTOBA")
#         # dqn_v3 = DeepQNetwork(JTOBA.action_n, JTOBA.observation_n, learning_rate=0.001, reward_decay=0.9,
#         #                       e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
#         # train(JTOBA, dqn_v3, 1, 500, 500)
#         # consumption = smooth_average(JTOBA.consumption_record, 10)
#         # s_jtoba.append(extract_last_n_average(consumption, 100))
#         # 算法IV-Random
#         random = OffloadingV1(request=gen.request, user_n=user_n, f_mec=f_mec, f_unit=1,
#                                 f_ue=f_ue, p_ue=p_ue, bandwidth=bandwidth, w1=w1, w2=w2)
#         random_agent = Random(random.action_n)
#         run(random, random_agent, 500)  # random算法无须训练
#         consumption = smooth_average(random.consumption_record, 10)
#         s_random.append(extract_last_n_average(consumption, 10))
#         print("Proposed:", s_proposed)
#         print("Local:", s_local)
#         # print("Proposed:", s_jtoba)
#         print("Random:", s_random)
#     # 设置中文字体
#     ch = font_manager.FontProperties(fname="/Library/Fonts/Arial Unicode.ttf")
#     sns.set(style='whitegrid', font=ch.get_name())
#     data = {
#         "Proposed": s_proposed,
#         "Local": s_local,
#         # "JTOBA": s_jtoba,
#         "Random": s_random,
#     }
#     print("Proposed:", s_proposed)
#     print("Local:", s_local)
#     # print("Proposed:", s_jtoba)
#     print("Random:", s_random)
#     df = pd.DataFrame(data, index=[8, 12, 16, 20], columns=["Proposed", "Local", "Random"])
#     sns.lineplot(data=df, markers=True)
#     plt.xlabel("小区用户数")
#     plt.ylabel("系统消耗")
#     plt.show()

# -----------------------------边缘云容量的影响-------------------------------
# # 用户数为10时的边缘云计算容量的影响
#     # 系统消耗
#     s_local = []
#     s_proposed = []
#     # s_saq = []
#     # s_jtoba = []
#     s_random = []
#     user_n = 10
#     gen = RequestGenerator(task_t=10, user_n=user_n)
#     for f_mec in [10, 20, 30, 40]:
#         w1 = np.array([1] * user_n)
#         w2 = np.array([0] * user_n)
#         f_ue = np.array([2] * user_n)
#         p_ue = np.array([0.3] * user_n)
#         # 算法I-Proposed Algorithm
#         proposed = OffloadingV1(request=gen.request, user_n=user_n, f_mec=f_mec, f_unit=1,
#                              f_ue=f_ue, p_ue=p_ue, bandwidth=user_n, w1=w1, w2=w2, reward_function="Proposed")
#         dqn_v1 = DeepQNetwork(proposed.action_n, proposed.observation_n, learning_rate=0.001, reward_decay=0.9,
#                               e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
#         train(proposed, dqn_v1, 1, 500, 500)
#         # 提取最后10次结果中的统计量
#         s_local.append(proposed.consumption_record[0])  # 全本地计算
#         consumption = extract_last_n_average(proposed.consumption_record, 10)
#         s_proposed.append(consumption)
#         # # 算法II-SAQ-learning
#         # saq_learning = OffloadingV1(request=gen.request, user_n=user_n, f_mec=f_mec, f_unit=1,
#         #                               f_ue=f_ue, p_ue=p_ue, bandwidth=user_n, w1=w1, w2=w2, reward_function="SAQ-learning")
#         # dqn_v2 = DeepQNetwork(saq_learning.action_n, saq_learning.observation_n, learning_rate=0.001, reward_decay=0.9,
#         #                       e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
#         # train(saq_learning, dqn_v2, 1, 2000, 500)
#         # consumption = extract_last_n_min(smooth_average(saq_learning.consumption_record, 10), 10)
#         # s_saq.append(consumption)
#         # # 算法III-JTOBA
#         # JTOBA = OffloadingV1(request=gen.request, user_n=user_n, f_mec=f_mec, f_unit=1,
#         #                             f_ue=f_ue, p_ue=p_ue, bandwidth=user_n, w1=w1, w2=w2, reward_function="JTOBA")
#         # dqn_v3 = DeepQNetwork(JTOBA.action_n, JTOBA.observation_n, learning_rate=0.001, reward_decay=0.9,
#         #                       e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
#         # train(JTOBA, dqn_v3, 1, 2000, 500)
#         # consumption = extract_last_n_min(smooth_average(JTOBA.consumption_record, 10), 5)
#         # s_jtoba.append(consumption)
#         # 算法IV-Random
#         random = OffloadingV1(request=gen.request, user_n=user_n, f_mec=f_mec, f_unit=1,
#                                 f_ue=f_ue, p_ue=p_ue, bandwidth=user_n, w1=w1, w2=w2)
#         random_agent = Random(random.action_n)
#         run(random, random_agent, 10)  # random算法无须训练
#         consumption = extract_last_n_average(random.consumption_record, 10)
#         s_random.append(consumption)
#     # 设置中文字体
#     ch = font_manager.FontProperties(fname="/Library/Fonts/Arial Unicode.ttf")
#     sns.set(style='whitegrid', font=ch.get_name())
#     data = {
#         "Proposed": s_proposed,
#         # "SAQ-learning": s_saq,
#         # "JTOBA": s_jtoba,
#         "Random": s_random,
#         "Local": s_local
#     }
#     df = pd.DataFrame(data, index=[10, 20, 30, 40], columns=["Proposed", "Local", "Random"])
#     sns.lineplot(data=df, markers=True)
#     plt.xlabel("边缘云容量")
#     plt.ylabel("系统消耗")
#     plt.show()

# -----------------------------信道容量的影响-------------------------------
# # 用户数为10时信道容量的影响
#     # 系统消耗
#     # s_local = []
#     s_proposed = []
#     s_saq = []
#     s_jtoba = []
#     s_random = []
#     user_n = 10
#     f_mec = 20
#     f_unit = 1
#     # bandwidth = 10
#     gen = RequestGenerator(task_t=10, user_n=user_n)
#     for bandwidth in [10, 20, 30, 40]:
#         w1 = np.array([1] * user_n)
#         w2 = np.array([0] * user_n)
#         f_ue = np.array([2] * user_n)
#         p_ue = np.array([0.3] * user_n)
#         # 算法I-Proposed Algorithm
#         proposed = OffloadingV1(request=gen.request, user_n=user_n, f_mec=f_mec, f_unit=f_unit,
#                              f_ue=f_ue, p_ue=p_ue, bandwidth=bandwidth, w1=w1, w2=w2, reward_function="Proposed")
#         dqn_v1 = DeepQNetwork(proposed.action_n, proposed.observation_n, learning_rate=0.001, reward_decay=0.9,
#                               e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
#         train(proposed, dqn_v1, 1, 2000, 500)
#         # 提取最后10次结果中的统计量
#         # s_local.append(proposed.consumption_record[0])  # 全本地计算
#         consumption = extract_last_n_average(proposed.consumption_record, 10)
#         s_proposed.append(consumption)
#         # 算法II-SAQ-learning
#         saq_learning = OffloadingV1(request=gen.request, user_n=user_n, f_mec=f_mec, f_unit=f_unit,
#                                       f_ue=f_ue, p_ue=p_ue, bandwidth=bandwidth, w1=w1, w2=w2, reward_function="SAQ-learning")
#         dqn_v2 = DeepQNetwork(saq_learning.action_n, saq_learning.observation_n, learning_rate=0.001, reward_decay=0.9,
#                               e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
#         train(saq_learning, dqn_v2, 1, 2000, 500)
#         consumption = extract_last_n_average(saq_learning.consumption_record, 100)
#         s_saq.append(consumption)
#         # 算法III-JTOBA
#         JTOBA = OffloadingV1(request=gen.request, user_n=user_n, f_mec=f_mec, f_unit=f_unit,
#                                     f_ue=f_ue, p_ue=p_ue, bandwidth=bandwidth, w1=w1, w2=w2, reward_function="JTOBA")
#         dqn_v3 = DeepQNetwork(JTOBA.action_n, JTOBA.observation_n, learning_rate=0.001, reward_decay=0.9,
#                               e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
#         train(JTOBA, dqn_v3, 1, 2000, 500)
#         consumption = extract_last_n_average(JTOBA.consumption_record, 100)
#         s_jtoba.append(consumption)
#         # 算法IV-Random
#         random = OffloadingV1(request=gen.request, user_n=user_n, f_mec=f_mec, f_unit=f_unit,
#                                 f_ue=f_ue, p_ue=p_ue, bandwidth=bandwidth, w1=w1, w2=w2)
#         random_agent = Random(random.action_n)
#         run(random, random_agent, 10)  # random算法无须训练
#         consumption = extract_last_n_average(random.consumption_record, 10)
#         s_random.append(consumption)
#     # 设置中文字体
#     ch = font_manager.FontProperties(fname="/Library/Fonts/Arial Unicode.ttf")
#     sns.set(style='whitegrid', font=ch.get_name())
#     data = {
#         "Proposed": s_proposed,
#         "SAQ-learning": s_saq,
#         "JTOBA": s_jtoba,
#         "Random": s_random,
#         # "Local": s_local
#     }
#     df = pd.DataFrame(data, index=[10, 20, 30, 40], columns=["Proposed", "SAQ-learning", "JTOBA", "Random"])
#     sns.lineplot(data=df, markers=True)
#     plt.xlabel("信道容量")
#     plt.ylabel("系统消耗")
#     plt.show()

# ------------------------------任务模型-----------------------------------
# import matplotlib.pyplot as plt
# import matplotlib.font_manager as font_manager
# import seaborn as sns
# import pandas as pd
#
# ch = font_manager.FontProperties(fname="/Library/Fonts/Arial Unicode.ttf")
# sns.set(style='whitegrid', font=ch.get_name())
#
# task_t=4
# time_slot=100
# x = np.linspace(0, time_slot, time_slot)
# cycle = time_slot / 2
# c = 2 * math.pi / cycle
# f = np.zeros([task_t, time_slot], dtype=float)
# sns.set(style='whitegrid', font=ch.get_name())
# # 任务流行度演化
# plt.subplot(1,2,1)
# for i in range(task_t):
#     y = 0.3 * np.sin(c*(x+i*cycle/task_t)) + 0.5
#     f[i, :] = y
# data = {
#         "0": f[0,:],
#         "1": f[1,:],
#         "2": f[2,:],
#         "3": f[3,:]
#     }
# df = pd.DataFrame(data, index=x, columns=["0", "1", "2", "3"])
# sns.lineplot(data=df)
# plt.xlabel("时隙")
# plt.ylabel("流行度演化")
# # 任务请求次数
# # time_slot = 100
# # task_t = 10
# user_n = 10
# x = np.linspace(0, time_slot, time_slot)
# cycle = time_slot / 2
# c = 2 * math.pi / cycle
# f = np.zeros([task_t, time_slot], dtype=float)
# r = np.zeros([task_t, time_slot], dtype=int)
# for i in range(task_t):
#     y = 0.3 * np.sin(c*(x+i*cycle/task_t)) + 0.5
#     f[i, :] = y
#     # plt.plot(x, y)
# for j in range(time_slot):
#     for i in range(task_t):
#         r[i, j] = round(f[i, j] / sum(f[:, j]) * user_n)
# plt.subplot(1,2,2)
# data = {
#         "0": r[0,:]
#     }
# df = pd.DataFrame(data, index=x, columns=["0"])
# sns.lineplot(data=df)
# plt.xlabel("时隙")
# plt.ylabel("任务请求数量")
# plt.show()

# --------------------------基于请求预测的缓存、基于经验流行度的缓存、无缓存对比实验------------------------
# # 用户数为20的缓存辅助效果实验
#     w1 = np.array([1]*20)
#     w2 = np.array([0]*20)
#     f_ue = np.array([2]*20)
#     p_ue = np.array([0.3]*20)
#     user_n = 20
#     task_t = 10
#     gen = LSTMRequestGenerator(task_t=task_t, user_n=user_n, time_slot=100)
#     cache_v1 = Cache(cache_capacity=150, task=gen.task, task_t=task_t, user_n=user_n)  # 所提缓存策略
#     cache_v2 = Cache(cache_capacity=150, task=gen.task, task_t=10, user_n=20)  # 只考虑任务量的缓存策略
#     # cache_v3 = Cache(cache_capacity=150, task=gen.task, task_t=task_t, user_n=user_n)  # 考虑经验流行度的缓存策略
#     # 10个时隙
#     s_cache_v1 = []
#     s_cache_v2 = []
#     # s_cache_v3 = []
#     # s_cache_none = []
#     time_slot = 10
#     for t in range(time_slot):
#         print("时隙开始")
#         request_v = gen.request_v_t(t)
#         request = gen.request_t(t)
#         print("request_v: ", request_v)
#         print("cache_v1: ", cache_v1.cache)
#         print("cache_v2: ", cache_v2.cache)
#         # print("cache_v3: ", cache_v3.cache)
#         # ---------------基于请求次数和任务量-------------
#         cache_aside_v1 = OffloadingV5(request_v=request_v, request=request, cache=cache_v1.cache, user_n=user_n, f_mec=2*user_n, f_unit=1,
#                               f_ue=f_ue, p_ue=p_ue, bandwidth=user_n, w1=w1, w2=w2)
#         dqn_v1 = DeepQNetwork(cache_aside_v1.action_n, cache_aside_v1.observation_n, learning_rate=0.001, reward_decay=0.9,
#                        e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
#         train(cache_aside_v1, dqn_v1, 1, 500, 500)
#         # 更新缓存
#         # 最小消耗位置索引
#         consumption = extract_last_n(cache_aside_v1.consumption_record, 10)
#         idx = consumption.index(min(consumption))
#         # 最小消耗卸载决策向量
#         offload_v = cache_aside_v1.offload_v[idx]
#         offload_t = []
#         for i in range(user_n):
#             if offload_v[i] > 0:
#                 offload_t.append(request_v[i])
#         # 更新缓存
#         print("offload_v1:", offload_t)
#         cache_v1.cache_update_size(offload_t, gen.request_v_t(t+1))
#         s_cache_v1.append(consumption[idx])
#         # hit_rate_v1.append(extract_last_n_max(cache_aside_v1.hit_rate, 10))
#         # ----------基于请求次数的缓存----------
#         cache_aside_v2 = OffloadingV5(request_v=request_v, request=request, cache=cache_v2.cache, user_n=user_n, f_mec=2*user_n, f_unit=1,
#                               f_ue=f_ue, p_ue=p_ue, bandwidth=user_n, w1=w1, w2=w2)
#         dqn_v2 = DeepQNetwork(cache_aside_v2.action_n, cache_aside_v2.observation_n, learning_rate=0.001, reward_decay=0.9,
#                        e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
#         train(cache_aside_v2, dqn_v2, 1, 500, 500)
#         # 更新缓存
#         # 最小消耗位置索引
#         consumption = extract_last_n(cache_aside_v2.consumption_record, 10)
#         idx = consumption.index(min(consumption))
#         # 最小消耗卸载决策向量
#         offload_v = cache_aside_v2.offload_v[idx]
#         offload_t = []
#         for i in range(20):
#             if offload_v[i] > 0:
#                 offload_t.append(request_v[i])
#         # 更新缓存
#         print("offload_v2:", offload_t)
#         cache_v2.cache_update(offload_t, gen.request_v_t(t + 1))
#         s_cache_v2.append(consumption[idx])
#         # hit_rate_v2.append(extract_last_n_max(cache_aside_v2.hit_rate, 10))
#         # # ------基于经验流行度---------
#         # cache_aside_v3 = OffloadingV5(request_v=request_v, request=request, cache=cache_v3.cache, user_n=user_n, f_mec=2*user_n,
#         #                               f_unit=1,
#         #                               f_ue=f_ue, p_ue=p_ue, bandwidth=user_n, w1=w1, w2=w2)
#         # dqn_v3 = DeepQNetwork(cache_aside_v3.action_n, cache_aside_v3.observation_n, learning_rate=0.001,
#         #                       reward_decay=0.9,
#         #                       e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
#         # train(cache_aside_v3, dqn_v3, 1, 500, 500)
#         # # 更新缓存
#         # # 最小消耗位置索引
#         # consumption = extract_last_n(cache_aside_v3.consumption_record, 10)
#         # idx = consumption.index(min(consumption))
#         # # 最小消耗卸载决策向量
#         # offload_v = cache_aside_v3.offload_v[idx]
#         # offload_t = []
#         # for i in range(user_n):
#         #     if offload_v[i] > 0:
#         #         offload_t.append(request_v[i])
#         # # 更新缓存
#         # print("offload_v3:", offload_t)
#         # cache_v3.cache_update_experience(offload_t, gen.popularity_t(t))
#         # s_cache_v3.append(consumption[idx])
#         # # hit_rate_v3.append(extract_last_n_max(cache_aside_v3.hit_rate, 10))
#         # ----------------无缓存--------------
#         # without_cache = OffloadingV1(request=request, user_n=20, f_mec=40, f_unit=1, f_ue=f_ue, p_ue=p_ue, bandwidth=20, w1=w1, w2=w2)
#         # dqn_v3 = DeepQNetwork(without_cache.action_n, without_cache.observation_n, learning_rate=0.001, reward_decay=0.9,
#         #                    e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
#         # train(without_cache, dqn_v3, 1, 500, 500)
#         # # 取最后10次中最小系统消耗
#         # s_cache_none.append(extract_last_n_min(without_cache.consumption_record, 10))
#         print("s_cache_v1: ", s_cache_v1)
#         print("s_cache_v2: ", s_cache_v2)
#         # #print("s_cache_v3: ", s_cache_v3)
#         # print("s_cache_none: ", s_cache_none)
#     # 设置中文字体
#     ch = font_manager.FontProperties(fname="/Library/Fonts/Arial Unicode.ttf")
#     sns.set(style="whitegrid",font=ch.get_name())
#     x = [1,2,3,4,5,6,7,8,9,10]
#     # plt.figure('系统消耗')
#     # b1 = plt.bar(x=x, height=s_cache_v1, width=0.4)
#     # b3 = plt.bar(x=x, height=list(np.array(s_cache_v2) - np.array(s_cache_v1)), width=0.4, bottom=s_cache_v1)
#     # b4 = plt.bar(x=x, height=np.array(s_cache_none) - np.array(s_cache_v2), width=0.4, bottom=s_cache_v2)
#     # plt.legend([b1,b3,b4], ["size-aware", "popularity-only", "none-cache"], loc='upper right')
#     # plt.ylim(600, 850)
#     # plt.xlabel("时间步")
#     # plt.ylabel("系统消耗")
#     # plt.figure('命中率变化')
#     # 条形图
#     label = [1,2,3,4,5,6,7,8,9,10]
#     b5 = plt.bar(x=x, height=s_cache_v1, width=0.2)
#     b6 = plt.bar(x=[i + 0.2 for i in x], height=s_cache_v2, width=0.2)
#     plt.xticks([i + 0.2 for i in x], label)
#     plt.legend([b5, b6], ["size-aware", "popularity-only"], loc='upper right')
#     plt.ylim(600, 850)
#     plt.xlabel("时间步")
#     plt.ylabel("系统消耗")
#     plt.show()

# 缓存对比数据（流行度预测）
# s_cache_v1:  [743.8202380952382, 689.7999999999998, 697.5666666666666, 724.0166666666668, 718.5166666666668, 717.0, 718.55, 692.4833333333332, 666.9999999999999, 672.05]
# s_cache_v3:  [757.5595238095237, 695.6666666666666, 722.1666666666667, 733.85, 736.6166666666667, 738.4833333333333, 726.5333333333333, 704.5833333333334, 679.0, 679.5]
# s_cache_none:  [755.2234126984126, 748.8357142857144, 779.225, 798.4833333333333, 794.5666666666666, 794.5095238095238, 790.6738095238095, 754.1666666666667, 730.6190476190476, 756.0285714285715]

# 任务量感知对比数据
# s_cache_v1:  [734.25, 680.3999999999999, 696.95, 724.7, 730.25, 721.8666666666667, 708.75, 686.85, 669.1166666666664, 680.4333333333334]
# s_cache_v2:  [738.7476190476192, 721.7166666666666, 721.9666666666666, 740.0333333333334, 733.9000000000001, 730.7666666666667, 730.7500000000001, 690.05, 676.2166666666666, 677.25]

# -------------------- 任务模型参数的影响 ----------------
# 用户数为20的缓存辅助效果实验
#     w1 = np.array([1]*20)
#     w2 = np.array([0]*20)
#     f_ue = np.array([2]*20)
#     p_ue = np.array([0.3]*20)
#     user_n = 20
#     task_t = 10
#     gen = LSTMRequestGenerator(task_t=task_t, user_n=user_n, time_slot=100)
#     cache_v1 = Cache(cache_capacity=150, task=gen.task, task_t=task_t, user_n=user_n)  # 所提缓存策略
#     cache_v2 = Cache(cache_capacity=150, task=gen.task, task_t=10, user_n=20)  # 只考虑流行度的缓存策略
#     s_cache_v1 = []
#     s_cache_v2 = []
#     for sigma in [5, 10, 15, 20]:
#         print("sigma开始: ", sigma)
#         # 修改任务生成器中的任务上传数据量
#         gen.task = gen.task_generate(sigma=sigma)
#         time_slot = 2
#         consumption_v1 = 0
#         consumption_v2 = 0
#         for t in range(time_slot):
#             print("时隙开始")
#             request_v = gen.request_v_t(t)
#             request = gen.request_t(t)
#             print("request_v: ", request_v)
#             print("cache_v1: ", cache_v1.cache)
#             print("cache_v2: ", cache_v2.cache)
#             # ---------------基于请求次数和任务量-------------
#             cache_aside_v1 = OffloadingV5(request_v=request_v, request=request, cache=cache_v1.cache, user_n=user_n, f_mec=2*user_n, f_unit=1,
#                                   f_ue=f_ue, p_ue=p_ue, bandwidth=user_n, w1=w1, w2=w2)
#             dqn_v1 = DeepQNetwork(cache_aside_v1.action_n, cache_aside_v1.observation_n, learning_rate=0.001, reward_decay=0.9,
#                            e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
#             train(cache_aside_v1, dqn_v1, 1, 500, 500)
#             # 更新缓存
#             # 最小消耗位置索引
#             consumption = extract_last_n(cache_aside_v1.consumption_record, 10)
#             idx = consumption.index(min(consumption))
#             # 最小消耗卸载决策向量
#             offload_v = cache_aside_v1.offload_v[idx]
#             offload_t = []
#             for i in range(user_n):
#                 if offload_v[i] > 0:
#                     offload_t.append(request_v[i])
#             # 更新缓存
#             print("offload_v1:", offload_t)
#             cache_v1.cache_update_size(offload_t, gen.request_v_t(t+1))
#             consumption_v1 = consumption[idx]
#             # ----------基于请求次数的缓存----------
#             cache_aside_v2 = OffloadingV5(request_v=request_v, request=request, cache=cache_v2.cache, user_n=user_n, f_mec=2*user_n, f_unit=1,
#                                   f_ue=f_ue, p_ue=p_ue, bandwidth=user_n, w1=w1, w2=w2)
#             dqn_v2 = DeepQNetwork(cache_aside_v2.action_n, cache_aside_v2.observation_n, learning_rate=0.001, reward_decay=0.9,
#                            e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
#             train(cache_aside_v2, dqn_v2, 1, 500, 500)
#             # 更新缓存
#             # 最小消耗位置索引
#             consumption = extract_last_n(cache_aside_v2.consumption_record, 10)
#             idx = consumption.index(min(consumption))
#             # 最小消耗卸载决策向量
#             offload_v = cache_aside_v2.offload_v[idx]
#             offload_t = []
#             for i in range(20):
#                 if offload_v[i] > 0:
#                     offload_t.append(request_v[i])
#             # 更新缓存
#             print("offload_v2:", offload_t)
#             cache_v2.cache_update(offload_t, gen.request_v_t(t + 1))
#             consumption_v2 = consumption[idx]
#             print("consumption_v1: ", consumption_v1)
#             print("consumption_v2: ", consumption_v2)
#             print("时隙结束")
#         s_cache_v1.append(consumption_v1)
#         s_cache_v2.append(consumption_v2)
#         print("sigma结束: ", sigma)
#     # 设置中文字体
#     ch = font_manager.FontProperties(fname="/Library/Fonts/Arial Unicode.ttf")
#     sns.set(style="whitegrid",font=ch.get_name())
#     x = [5,10,15,20]
#     data = {
#         "proposed": s_cache_v1,
#         "prediction-only": s_cache_v2
#     }
#     df = pd.DataFrame(data, index=x, columns=["proposed", "prediction-only"])
#     sns.lineplot(data=df, markers=True)
#     plt.xlabel("数据量方差")
#     plt.ylabel("系统消耗")
#     plt.show()
