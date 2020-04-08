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
#     sns.set(font=ch.get_name())
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

# -----------------------------缓存辅助系统消耗变化---------------------------
# 用户数为20的辅助奖励效果实验
#     w1 = np.array([1]*20)
#     w2 = np.array([0]*20)
#     f_ue = np.array([2]*20)
#     p_ue = np.array([0.3]*20)
#     gen = RequestGenerator(task_t=10, user_n=20)
#     # 算法I-Proposed with aside reward Alogorithm
#     aside = OffloadingV1(request=gen.request, user_n=20, f_mec=40, f_unit=1,
#                                   f_ue=f_ue, p_ue=p_ue, bandwidth=20, w1=w1, w2=w2, reward_function="Proposed_aside")
#     dqn_v1 = DeepQNetwork(aside.action_n, aside.observation_n, learning_rate=0.001, reward_decay=0.9,
#                    e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
#     train(aside, dqn_v1, 1, 2000, 500)
#     consumption_v1 = smooth_min(aside.consumption_record, 10)
#     # 算法II-without aside reward
#     without_aside = OffloadingV1(request=gen.request, user_n=20, f_mec=40, f_unit=1,
#                                   f_ue=f_ue, p_ue=p_ue, bandwidth=20, w1=w1, w2=w2, reward_function="Proposed")
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
# 用户数为[10,15,20]的辅助奖励效果实验
#     # 系统消耗
#     c_aside = []
#     c_none = []
#     # 卸载人数
#     u_aside = []
#     u_none = []
#     for user_n in [10, 15, 20]:
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
#     sns.set(font=ch.get_name())
#     plt.subplot(1, 2, 1)
#     data = {
#         "带辅助奖励": c_aside,
#         "无辅助奖励": c_none,
#     }
#     df = pd.DataFrame(data, index=[10, 15, 20], columns=["带辅助奖励", "无辅助奖励"])
#     sns.lineplot(data=df)
#     plt.xlabel("小区用户数")
#     plt.ylabel("系统消耗")
#     # 条形图
#     plt.subplot(1, 2, 2)
#     label = [10, 15, 20]
#     x = range(len(u_aside))
#     plt.bar(x=x, height=u_aside, width=0.4, label="带辅助奖励")
#     plt.bar(x=[i + 0.4 for i in x], height=u_none, width=0.4, label="无辅助奖励")
#     plt.xticks([i + 0.2 for i in x], label)
#     plt.xlabel("小区用户数")
#     plt.ylabel("卸载人数")
#     plt.show()

# ------------------------------用户数影响----------------------------------
# 用户数为[10, 15, 20，25，30]的辅助奖励效果实验
#     # 系统消耗
#     s_local = []
#     s_proposed = []
#     s_saq = []
#     s_jtoba = []
#     s_random = []
#     for user_n in [10, 12, 14, 16, 20]:
#         w1 = np.array([1] * user_n)
#         w2 = np.array([0] * user_n)
#         f_ue = np.array([2] * user_n)
#         p_ue = np.array([0.3] * user_n)
#         gen = RequestGenerator(task_t=10, user_n=user_n)
#         # 算法I-Proposed Algorithm
#         proposed = OffloadingV1(request=gen.request, user_n=user_n, f_mec=2*user_n, f_unit=1,
#                              f_ue=f_ue, p_ue=p_ue, bandwidth=user_n, w1=w1, w2=w2, reward_function="Proposed")
#         dqn_v1 = DeepQNetwork(proposed.action_n, proposed.observation_n, learning_rate=0.001, reward_decay=0.9,
#                               e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
#         train(proposed, dqn_v1, 1, 2000, 500)
#         # 提取最后10次结果中的统计量
#         s_local.append(proposed.consumption_record[0])
#         consumption = smooth_average(proposed.consumption_record, 10)
#         s_proposed.append(extract_last_n_min(consumption, 10))
#         # 算法II-SAQ-learning
#         saq_learning = OffloadingV1(request=gen.request, user_n=user_n, f_mec=2*user_n, f_unit=1,
#                                       f_ue=f_ue, p_ue=p_ue, bandwidth=user_n, w1=w1, w2=w2, reward_function="SAQ-learning")
#         dqn_v2 = DeepQNetwork(saq_learning.action_n, saq_learning.observation_n, learning_rate=0.001, reward_decay=0.9,
#                               e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
#         train(saq_learning, dqn_v2, 1, 2000, 500)
#         consumption = smooth_average(saq_learning.consumption_record, 10)
#         s_saq.append(extract_last_n_min(consumption, 10))
#         # 算法III-JTOBA
#         JTOBA = OffloadingV1(request=gen.request, user_n=user_n, f_mec=2*user_n, f_unit=1,
#                                     f_ue=f_ue, p_ue=p_ue, bandwidth=user_n, w1=w1, w2=w2, reward_function="JTOBA")
#         dqn_v3 = DeepQNetwork(JTOBA.action_n, JTOBA.observation_n, learning_rate=0.001, reward_decay=0.9,
#                               e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
#         train(JTOBA, dqn_v3, 1, 2000, 500)
#         consumption = smooth_average(JTOBA.consumption_record, 10)
#         s_jtoba.append(extract_last_n_min(consumption, 10))
#         # 算法IV-Random
#         random = OffloadingV1(request=gen.request, user_n=user_n, f_mec=2*user_n, f_unit=1,
#                                 f_ue=f_ue, p_ue=p_ue, bandwidth=user_n, w1=w1, w2=w2)
#         random_agent = Random(random.action_n)
#         run(random, random_agent, 2000)  # random算法无须训练
#         consumption = smooth_average(random.consumption_record, 10)
#         s_random.append(extract_last_n_min(consumption, 10))
#     # 设置中文字体
#     ch = font_manager.FontProperties(fname="/Library/Fonts/Arial Unicode.ttf")
#     sns.set(font=ch.get_name())
#     data = {
#         "Proposed": s_proposed,
#         "SAQ-learning": s_saq,
#         "JTOBA": s_jtoba,
#         "Random": s_random,
#         "Local": s_local
#     }
#     df = pd.DataFrame(data, index=[10, 12, 14, 16, 20], columns=["Proposed", "SAQ-learning", "JTOBA", "Random", "Local"])
#     sns.lineplot(data=df)
#     plt.xlabel("小区用户数")
#     plt.ylabel("系统消耗")
#     plt.show()

# ------------------------------缓存命中率实验-------------------------------
# 用户数为20的缓存辅助命中实验
#     w1 = np.array([1]*20)
#     w2 = np.array([0]*20)
#     f_ue = np.array([2]*20)
#     p_ue = np.array([0.3]*20)
#     gen = LSTMRequestGenerator(task_t=10, user_n=20, time_slot=100)
#     cache = Cache(cache_capacity=150, task=gen.task, task_t=10, user_n=20)
#     # 5个时隙
#     for t in range(5):
#         print("时隙开始")
#         request_v = gen.request_v_t(t)
#         request = gen.request_t(t)
#         cache_aside = OffloadingV5(request_v=request_v, request=request, cache=cache.cache, user_n=20, f_mec=40, f_unit=1,
#                               f_ue=f_ue, p_ue=p_ue, bandwidth=20, w1=w1, w2=w2)
#         dqn = DeepQNetwork(cache_aside.action_n, cache_aside.observation_n, learning_rate=0.001, reward_decay=0.9,
#                        e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
#         train(cache_aside, dqn, 1, 20, 5)
#         cache_aside.reset_consumption_record()
#         cache_aside.reset_isTest(isTest=True)  # 切换到测试环境
#         run(cache_aside, dqn, 10)
#         # 最小消耗位置索引
#         consumption = cache_aside.consumption_record
#         idx = consumption.index(min(consumption))
#         # 最小消耗卸载决策向量
#         offload_v = cache_aside.offload_v[idx-1]
#         offload_t = []
#         for i in range(20):
#             if offload_v[i] > 0:
#                 offload_t.append(request_v[i])
#         print("请求：", request_v)
#         print("缓存状态：", cache.cache)
#         print("系统消耗：", consumption, "卸载决策", offload_t)
#         cache.cache_update(offload_t, gen.request_v_t(t+1))
#         print("时隙结束")

# -------------------------------缓存效果对比实验---------------------------------------
# 用户数为20的缓存辅助效果实验
#     w1 = np.array([1]*20)
#     w2 = np.array([0]*20)
#     f_ue = np.array([2]*20)
#     p_ue = np.array([0.3]*20)
#     gen = LSTMRequestGenerator(task_t=10, user_n=20, time_slot=100)
#     cache_v1 = Cache(cache_capacity=150, task=gen.task, task_t=10, user_n=20)  # 考虑流行度与任务量的缓存策略
#     cache_v2 = Cache(cache_capacity=150, task=gen.task, task_t=10, user_n=20)  # 只考虑任务量的缓存策略
#     # 10个时隙
#     s_cache_v1 = []
#     s_cache_v2 = []
#     s_cache_none = []
#     hit_rate_v1 = []
#     hit_rate_v2 = []
#     for t in range(10):
#         print("时隙开始")
#         request_v = gen.request_v_t(t)
#         request = gen.request_t(t)
#         # 考虑流行度与任务量的主动缓存策略
#         cache_aside_v1 = OffloadingV5(request_v=request_v, request=request, cache=cache_v1.cache, user_n=20, f_mec=40, f_unit=1,
#                               f_ue=f_ue, p_ue=p_ue, bandwidth=20, w1=w1, w2=w2)
#         dqn_v1 = DeepQNetwork(cache_aside_v1.action_n, cache_aside_v1.observation_n, learning_rate=0.001, reward_decay=0.9,
#                        e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
#         train(cache_aside_v1, dqn_v1, 1, 100, 100)
#         # 更新缓存
#         # 最小消耗位置索引
#         consumption = extract_last_n(smooth(cache_aside_v1.consumption_record, 10), 10)
#         hit_rate = extract_last_n(smooth(cache_aside_v1.hit_rate, 10), 10)
#         idx = consumption.index(min(consumption))
#         # 最小消耗卸载决策向量
#         offload_v = cache_aside_v1.offload_v[idx]
#         offload_t = []
#         for i in range(20):
#             if offload_v[i] > 0:
#                 offload_t.append(request_v[i])
#         # 更新缓存
#         cache_v1.cache_update_size(offload_t, gen.request_v_t(t+1))
#         s_cache_v1.append(consumption[idx])
#         hit_rate_v1.append(hit_rate[idx])
#         # 考虑流行度的主动缓存策略
#         cache_aside_v2 = OffloadingV5(request_v=request_v, request=request, cache=cache_v2.cache, user_n=20, f_mec=40, f_unit=1,
#                               f_ue=f_ue, p_ue=p_ue, bandwidth=20, w1=w1, w2=w2)
#         dqn_v2 = DeepQNetwork(cache_aside_v2.action_n, cache_aside_v2.observation_n, learning_rate=0.001, reward_decay=0.9,
#                        e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
#         train(cache_aside_v2, dqn_v1, 1, 100, 100)
#         # 更新缓存
#         # 最小消耗位置索引
#         consumption = extract_last_n(smooth(cache_aside_v2.consumption_record, 10), 10)
#         hit_rate = extract_last_n(smooth(cache_aside_v2.hit_rate, 10), 10)
#         idx = consumption.index(min(consumption))
#         # 最小消耗卸载决策向量
#         offload_v = cache_aside_v2.offload_v[idx]
#         offload_t = []
#         for i in range(20):
#             if offload_v[i] > 0:
#                 offload_t.append(request_v[i])
#         # 更新缓存
#         cache_v2.cache_update(offload_t, gen.request_v_t(t+1))
#         s_cache_v2.append(consumption[idx])
#         hit_rate_v2.append(hit_rate[idx])
#         # 无缓存
#         without_cache = OffloadingV1(request=request, user_n=20, f_mec=40, f_unit=1, f_ue=f_ue, p_ue=p_ue, bandwidth=20, w1=w1, w2=w2)
#         dqn_v2 = DeepQNetwork(without_cache.action_n, without_cache.observation_n, learning_rate=0.001, reward_decay=0.9,
#                            e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
#         train(without_cache, dqn_v2, 1, 100, 100)
#         # 取最后10次中最小系统消耗
#         cache_none.append(extract_last_n_min(without_cache.consumption_record, 10))
#     # 设置中文字体
#     ch = font_manager.FontProperties(fname="/Library/Fonts/Arial Unicode.ttf")
#     sns.set(font=ch.get_name())
#     data = {
#         "缓存辅助": with_cache,
#         "无缓存": cache_none
#     }
#     df = pd.DataFrame(data, index=np.linspace(0, 10, 10), columns=["缓存辅助", "无缓存"])
#     sns.lineplot(data=df)
#     plt.xlabel("时间步")
#     plt.ylabel("系统消耗")
#     plt.show()

