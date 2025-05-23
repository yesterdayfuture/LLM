
import random
import numpy as np

#每个老虎机的中奖概率,0-1之间的均匀分布
probs = np.random.uniform(size=10)

#记录每个老虎机的收益，即 期望或 返回值
rewards = [[1] for _ in range(10)]


#随机选择的概率递减的贪婪算法
def choose_one():
    #求出每个老虎机各玩了多少次
    played_count = [len(i) for i in rewards]
    played_count = np.array(played_count)

    #求出上置信界
    #分子是总共玩了多少次,取根号后让他的增长速度变慢
    #分母是每台老虎机玩的次数,乘以2让他的增长速度变快
    #随着玩的次数增加,分母会很快超过分子的增长速度,导致分数越来越小
    #具体到每一台老虎机,则是玩的次数越多,分数就越小,也就是ucb的加权越小
    #所以ucb衡量了每一台老虎机的不确定性,不确定性越大,探索的价值越大
    fenzi = played_count.sum()**0.5
    fenmu = played_count * 2
    ucb = fenzi / fenmu

    #ucb本身取根号
    #大于1的数会被缩小,小于1的数会被放大,这样保持ucb恒定在一定的数值范围内
    ucb = ucb**0.5

    #计算每个老虎机的奖励平均
    rewards_mean = [np.mean(i) for i in rewards]
    rewards_mean = np.array(rewards_mean)

    #ucb和期望求和
    ucb += rewards_mean

    return ucb.argmax()


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


