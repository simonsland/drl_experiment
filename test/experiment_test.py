# # 预训练参数对结果的影响
# DQN结构：三层全连接层，隐藏层神经元个数为50
# # 用户数为10的卸载时延实验环境
# w1 = np.array([1]*10)
# w2 = np.array([0]*10)
# f_ue = np.array([2]*10)
# p_ue = np.array([0.3]*10)
# delay_u10 = Offloading(user_n=10, task_t=20, f_mec=15, f_unit=1, f_ue=f_ue, p_ue=p_ue, bandwidth=20, w1=w1, w2=w2)
# # 1000步预训练
# dqn_pre_1000 = DeepQNetwork(delay_u10.action_n, delay_u10.observation_n, learning_rate=0.001, reward_decay=0.9,
#                             e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
# run(env=delay_u10, agent=dqn_pre_1000, episode=2000, pre_train=1000)
# dqn_pre_1000.sess.close()
# consumption_pre_1000 = delay_u10.consumption_record
#
# delay_u10.reset_record()
#
# # 500步预训练
# dqn_pre_500 = DeepQNetwork(delay_u10.action_n, delay_u10.observation_n, learning_rate=0.001, reward_decay=0.9,
#                            e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
# run(env=delay_u10, agent=dqn_pre_500, episode=2000, pre_train=500)
# consumption_pre_500 = delay_u10.consumption_record
# dqn_pre_500.sess.close()
# # 打印结果
# plt.ylabel('consumption')
# plt.xlabel('training steps')
# plot_consumption(consumption_pre_500, 'b-')
# plot_consumption(consumption_pre_1000, 'r-')
# plt.show()


# # 泛化能力测试
# dqn = DeepQNetwork(delay_u10_train.action_n, delay_u10_train.observation_n, learning_rate=0.001, reward_decay=0.9,
#                    e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
# train(env=delay_u10_train, agent=dqn, episode=2000, pre_train=100)
# consumption_train = smooth_consumption(delay_u10_train.consumption_record)
# # dqn.sess.close()
#
# delay_u10_test = Offloading(user_n=10, task_t=20, f_mec=15, f_unit=1, f_ue=f_ue, p_ue=p_ue, bandwidth=20, w1=w1, w2=w2)
# run(delay_u10_test, dqn, 100)
# consumption_test = delay_u10_test.consumption_record
#
# # 打印结果
# plt.subplot(2, 1, 1)
# plt.plot(np.arange(len(consumption_train)), consumption_train, 'b-')
# plt.ylabel('consumption')
# plt.xlabel('training steps')
#
# plt.subplot(2, 1, 2)
# plt.plot(np.arange(len(consumption_test)), consumption_test, 'b-')
# plt.ylabel('consumption')
# plt.xlabel('test episode')
#
# plt.show()



# # 用户数为10的卸载时延实验环境
# w1 = np.array([1]*10)
# w2 = np.array([0]*10)
# f_ue = np.array([2]*10)
# p_ue = np.array([0.3]*10)
# # 随计算容量增加的变化
# mec = 10
# delay_u10_mec10 = Offloading(user_n=10, task_t=20, f_mec=10, f_unit=1, f_ue=f_ue, p_ue=p_ue, bandwidth=20, w1=w1, w2=w2)
# dqn_u10_mec10 = DeepQNetwork(delay_u10_mec10.action_n, delay_u10_mec10.observation_n, learning_rate=0.001, reward_decay=0.9,
#                    e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
# train(delay_u10_mec10, dqn_u10_mec10, 2000, 500)
# record = smooth_consumption(delay_u10_mec10.consumption_record)
# plt.plot(np.arange(len(record)), record, 'b-')
# # mec = 20
# delay_u10_mec20 = Offloading(user_n=10, task_t=20, f_mec=20, f_unit=2, f_ue=f_ue, p_ue=p_ue, bandwidth=20,
#                                    w1=w1, w2=w2)
# dqn_u10_mec_20 = DeepQNetwork(delay_u10_mec20.action_n, delay_u10_mec20.observation_n, learning_rate=0.001,
#                    reward_decay=0.9,
#                    e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
# train(delay_u10_mec20, dqn_u10_mec_20, 2000, 500)
# record = smooth_consumption(delay_u10_mec20.consumption_record)
# plt.plot(np.arange(len(record)), record, 'r-')
# # mec = 40
# delay_u10_mec40 = Offloading(user_n=10, task_t=20, f_mec=40, f_unit=2, f_ue=f_ue, p_ue=p_ue, bandwidth=20,
#                                    w1=w1, w2=w2)
# dqn_u10_mec40 = DeepQNetwork(delay_u10_mec40.action_n, delay_u10_mec40.observation_n, learning_rate=0.001,
#                    reward_decay=0.9,
#                    e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
# train(delay_u10_mec40, dqn_u10_mec40, 2000, 500)
# record = smooth_consumption(delay_u10_mec40.consumption_record)
# plt.plot(np.arange(len(record)), record, 'g-')
# # mec = 60
# delay_u10_mec60 = Offloading(user_n=10, task_t=20, f_mec=60, f_unit=2, f_ue=f_ue, p_ue=p_ue, bandwidth=20,
#                                    w1=w1, w2=w2)
# dqn_u10_mec60 = DeepQNetwork(delay_u10_mec60.action_n, delay_u10_mec60.observation_n, learning_rate=0.001,
#                    reward_decay=0.9,
#                    e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
# train(delay_u10_mec60, dqn_u10_mec60, 2000, 500)
# record = smooth_consumption(delay_u10_mec60.consumption_record)
# plt.plot(np.arange(len(record)), record, 'k-')
# # # optimal
# # optimal_u10_mec10 = OptimalPolicy(request=delay_u10_mec10.request,
# #                                   observation_n=delay_u10_mec10.observation_n,
# #                                   action_n=delay_u10_mec10.action_n, f_mec=delay_u10_mec10.f_mec,
# #                                   f_unit=delay_u10_mec10.f_unit, f_ue=2,
# #                                   bandwidth=delay_u10_mec10.bandwidth)
# # optimal_u10_mec10.run()
# # print(optimal_u10_mec10.optimal_delay, optimal_u10_mec10.optimal_decision)
# # optimal_u10_mec20 = OptimalPolicy(request=delay_u10_mec20.request,
# #                                   observation_n=delay_u10_mec20.observation_n,
# #                                   action_n=delay_u10_mec20.action_n, f_mec=delay_u10_mec20.f_mec,
# #                                   f_unit=delay_u10_mec20.f_unit, f_ue=2,
# #                                   bandwidth=delay_u10_mec20.bandwidth)
# plt.ylabel('consumption')
# plt.xlabel('training steps')
# plt.show()


# 用户数为20的卸载时延实验环境
# w1 = np.array([1]*20)
# w2 = np.array([0]*20)
# f_ue = np.array([2]*20)
# p_ue = np.array([0.3]*20)
# # 随用户增加的变化
# # 查看分配的均衡情况
# delay_u20 = Offloading(user_n=20, task_t=40, f_mec=60, f_unit=2, f_ue=f_ue, p_ue=p_ue, bandwidth=20, w1=w1, w2=w2)
# dqn_u10_mec10 = DeepQNetwork(delay_u20.action_n, delay_u20.observation_n, learning_rate=0.001, reward_decay=0.9,
#                    e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=True)
# train(delay_u20, dqn_u10_mec10, 2000, 500)
# record = smooth_consumption(delay_u20.consumption_record)
# plt.plot(np.arange(len(record)), record, 'b-')


# 用户数为20的卸载时延实验环境
# w1 = np.array([1]*20)
# w2 = np.array([0]*20)
# f_ue = np.array([2]*20)
# p_ue = np.array([0.3]*20)
# # 分配范围的优化
# delay_u20 = Offloading(user_n=20, task_t=40, f_mec=40, f_unit=1, f_ue=f_ue, p_ue=p_ue, bandwidth=20, w1=w1, w2=w2)
# dqn_u20_v1 = DeepQNetwork(delay_u20.action_n, delay_u20.observation_n, learning_rate=0.001, reward_decay=0.9,
#                    e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=False)
# train(delay_u20, dqn_u20_v1, 2000, 500)
# consumption_v1 = smooth_consumption(delay_u20.consumption_record)
# plt.subplot(2, 1, 1)
# plt.plot(np.arange(len(consumption_v1)), consumption_v1, 'b-')
# plt.subplot(2, 1, 2)
# plt.plot(np.arange(20), delay_u20.offload_record[-20:], 'b-')
#
# delay_u20.reset_consumption_record()
# delay_u20.reset_offload_record()
# delay_u20.reset_reward_function('v_5')
# dqn_u20_v2 = DeepQNetwork(delay_u20.action_n, delay_u20.observation_n, learning_rate=0.001, reward_decay=0.9,
#                           e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=False)
# train(delay_u20, dqn_u20_v2, 2000, 500)
# consumption_v2 = smooth_consumption(delay_u20.consumption_record)
# plt.subplot(2, 1, 1)
# plt.plot(np.arange(len(consumption_v2)), consumption_v2, 'r-')
# plt.subplot(2, 1, 2)
# plt.plot(np.arange(20), delay_u20.offload_record[-20:], 'r-')
# # plt.ylabel('consumption')
# # plt.xlabel('training steps')
# plt.show()

# 用户数为10的卸载时延实验环境
# w1 = np.array([1]*10)
# w2 = np.array([0]*10)
# f_ue = np.array([2]*10)
# p_ue = np.array([0.3]*10)
# # 多回合模型
# delay_u10_multi = OffloadingV2(user_n=10, task_t=15, f_mec=40, f_unit=1, f_ue=f_ue, p_ue=p_ue, bandwidth=20, w1=w1, w2=w2)
# dqn_u10_multi = DeepQNetwork(delay_u10_multi.action_n, delay_u10_multi.observation_n, learning_rate=0.001, reward_decay=0.9,
#                    e_greedy=0.9, replace_target_iter=100, memory_size=2000, output_graph=False)
# train(delay_u10_multi, dqn_u10_multi, 4000, 1, 500)  # 训练模型
# delay_u10_multi.reset_consumption_record()
# # 随机模型
# random_u10_multi = Random(delay_u10_multi.action_n)
# local, random, dqn = compare(delay_u10_multi, dqn_u10_multi, random_u10_multi, 20, 10)
# plt.plot(np.arange(len(local)), local, 'b-')
# plt.plot(np.arange(len(dqn)), dqn, 'r-')
# plt.plot(np.arange(len(random)), random, 'k-')
# plt.ylabel('consumption')
# plt.xlabel('steps')
# plt.show()
