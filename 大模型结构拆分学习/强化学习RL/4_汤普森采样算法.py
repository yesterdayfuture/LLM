
import random
import numpy as np

#每个老虎机的中奖概率,0-1之间的均匀分布
probs = np.random.uniform(size=10)

#记录每个老虎机的收益，即 期望或 返回值
rewards = [[1] for _ in range(10)]


'''
np.random.beta 用于从‌Beta分布‌中生成随机样本。
Beta分布是定义在区间 (0, 1) 上的连续概率分布，其形状由两个正参数 α（alpha）和 β（beta）控制
'''
#beta分布测试
# print('当数字小的时候,beta分布的概率有很大的随机性')
# for _ in range(5):
#     print(np.random.beta(1, 1))

# print('当数字大时,beta分布逐渐稳定')
# for _ in range(5):
#     print(np.random.beta(1e5, 1e5))



#选择一个老虎机
def choose_one():
    #求出每个老虎机出1的次数+1
    count_1 = [sum(i) + 1 for i in rewards]

    #求出每个老虎机出0的次数+1
    count_0 = [sum(1 - np.array(i)) + 1 for i in rewards]

    #按照beta分布计算奖励分布,这可以认为是每一台老虎机中奖的概率
    beta = np.random.beta(count_1, count_0)

    return beta.argmax()


def try_and_play():
    i = choose_one()

    #玩老虎机,得到结果
    reward = 0
    if random.random() < probs[i]:
        reward = 1

    #记录玩的结果
    rewards[i].append(reward)


def get_result():
    #玩N次
    for _ in range(5000):
        try_and_play()

    #期望的最好结果
    target = probs.max() * 5000

    #实际玩出的结果
    result = sum([sum(i) for i in rewards])

    return target, result


print(get_result())
