
import random
import numpy as np

#每个老虎机的中奖概率,0-1之间的均匀分布
probs = np.random.uniform(size=10)

#记录每个老虎机的收益，即 期望或 返回值
rewords = [[1] for _ in range(10)]


#贪婪算法
def choose_one():

    #统计 玩老虎机的总次数
    run_counts = sum([len(i) for i in rewords])

    #随机选择的概率逐渐下降
    if random.random() < 1/run_counts:
        return random.randint(0,9)
    
    #计算每个老虎机的期望收益
    reword_mean = [np.mean(i) for i in rewords]
    #选择收益最大的老虎机
    return np.argmax(reword_mean)


# for _ in range(10):
#     #选择老虎机
#     index = choose_one()
#     #老虎机中奖
#     print(index)


def try_and_play():
    i = choose_one()

    #玩老虎机,得到结果
    reward = 0
    if random.random() < probs[i]:
        reward = 1

    #记录玩的结果
    rewords[i].append(reward)

# try_and_play()
# print(rewords)


def get_result():
    #玩N次
    for _ in range(5000):
        try_and_play()

    #期望的最好结果
    target = probs.max() * 5000

    #实际玩出的结果
    result = sum([sum(i) for i in rewords])

    return target, result


print(get_result())

