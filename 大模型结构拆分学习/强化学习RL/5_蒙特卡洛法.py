
import numpy as np

#状态转移概率矩阵
#很显然,状态4(第5行)就是重点了,要进入状态4,只能从状态2,3进入
#[5, 5]
P = np.array([
    [0.5, 0.5, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.5, 0.5],
    [0.0, 0.1, 0.2, 0.2, 0.5],
    [0.0, 0.0, 0.0, 0.0, 0.0],
])

#反馈矩阵，-100的位置是不可能走到的
#[5, 5]
R = np.array([
    [-1.0, 0.0, -100.0, -100.0, -100.0],
    [-1.0, -100.0, -2.0, -100.0, -100.0],
    [-100.0, -100.0, -100.0, -2.0, 0.0],
    [-100.0, 1.0, 1.0, 1.0, 10.0],
    [-100.0, -100.0, -100.0, -100.0, -100.0],
])

import numpy as np
import random


#生成一个chain
def get_chain(max_lens):
    #采样结果
    ss = []
    rs = []

    #随机选择一个除4以外的状态作为起点
    s = random.choice(range(4)) #从0-3选择一个
    ss.append(s)

    for _ in range(max_lens):
        #按照P的概率，找到下一个状态
        s_next = np.random.choice(np.arange(5), p=P[s])

        #取到r
        r = R[s, s_next]

        #s_next变成当前状态,开始接下来的循环
        s = s_next

        ss.append(s)
        rs.append(r)

        #如果状态到了4则结束
        if s == 4:
            break

    return ss, rs


#生成N个chain
def get_chains(N, max_lens):
    ss = []
    rs = []
    for _ in range(N):
        s, r = get_chain(max_lens)
        ss.append(s)
        rs.append(r)

    return ss, rs


ss, rs = get_chains(100, 20)

#给定一条链,计算回报
def get_value(rs):
    sum = 0
    for i, r in enumerate(rs):
        #给每一步的反馈做一个系数,随着步数往后衰减,也就是说,越早的动作影响越大
        sum += 0.5**i * r

    #最终的反馈是所有步数衰减后的求和
    return sum

#蒙特卡洛法评估每个状态的价值
def get_values_by_monte_carlo(ss, rs):
    #记录5个不同开头的价值
    #其实只有4个,因为状态4是不可能作为开头状态的
    values = [[] for _ in range(5)]

    #遍历所有链
    for s, r in zip(ss, rs):
        #计算不同开头的价值
        values[s[0]].append(get_value(r))

    #求每个开头的平均价值
    return [np.mean(i) for i in values]



#计算状态动作对(s,a)出现的频率,以此来估算策略的占用度量
def occupancy(ss, rs, s, a):
    rho = 0

    count_by_time = np.zeros(max_time)
    count_by_s_a = np.zeros(max_time)

    for si, ri in zip(ss, rs):
        for i in range(len(ri)):
            s_opt = si[i]
            a_opt = si[i + 1]

            #统计每个时间步的次数
            count_by_time[i] += 1

            #统计s，a出现的次数
            if s == s_opt and a == a_opt:
                count_by_s_a[i] += 1

    #i -> [999 - 0]
    for i in reversed(range(max_time)):
        if count_by_time[i] == 0:
            continue

        #以时间逐渐衰减
        rho += 0.5**i * count_by_s_a[i] / count_by_time[i]

    return (1 - 0.5) * rho


max_time = 1000
ss, rs = get_chains(max_time, 2000)


print(occupancy(ss, rs, 3, 1) + occupancy(ss, rs, 3, 2) + occupancy(ss, rs, 3, 3))