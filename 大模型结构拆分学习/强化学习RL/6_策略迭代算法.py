

#获取一个格子的状态 ground 前进,trap 陷阱,terminal 终止
def get_state(row, col):
    if row != 3:
        return 'ground'

    if row == 3 and col == 0:
        return 'ground'

    if row == 3 and col == 11:
        return 'terminal'

    return 'trap'


print(get_state(0, 0))



#在一个格子里做一个动作
def move(row, col, action):
    #如果当前已经在陷阱或者终点，则不能执行任何动作，反馈都是0
    if get_state(row, col) in ['trap', 'terminal']:
        return row, col, 0

    #↑
    if action == 0:
        row -= 1

    #↓
    if action == 1:
        row += 1

    #←
    if action == 2:
        col -= 1

    #→
    if action == 3:
        col += 1

    #不允许走到地图外面去
    row = max(0, row)
    row = min(3, row)
    col = max(0, col)
    col = min(11, col)

    #是陷阱的话，奖励是-100，否则都是-1
    #这样强迫了机器尽快结束游戏,因为每走一步都要扣一分
    #结束最好是以走到终点的形式,避免被扣100分
    reward = -1
    if get_state(row, col) == 'trap':
        reward = -100

    return row, col, reward


print(move(3, 1, 0))


import numpy as np

#初始化每个格子的价值
values = np.zeros([4, 12])

#初始化每个格子下采用动作的概率
pi = np.ones([4, 12, 4]) * 0.25



#计算在一个状态下执行动作的分数
def get_qsa(row, col, action):
    #在当前状态下执行动作,得到下一个状态和reward
    next_row, next_col, reward = move(row, col, action)

    #计算下一个状态的分数,取values当中记录的分数即可,0.9是折扣因子
    value = values[next_row, next_col] * 0.9

    #如果下个状态是终点或者陷阱,则下一个状态的分数是0
    if get_state(next_row, next_col) in ['trap', 'terminal']:
        value = 0

    #动作的分数本身就是reward,加上下一个状态的分数
    return value + reward


print(get_qsa(0, 0, 0))


#策略评估
def get_values():

    #初始化一个新的values,重新评估所有格子的分数
    new_values = np.zeros([4, 12])

    #遍历所有格子
    for row in range(4):
        for col in range(12):

            #计算当前格子4个动作分别的分数
            action_value = np.zeros(4)

            #遍历所有动作
            for action in range(4):
                action_value[action] = get_qsa(row, col, action)

            #每个动作的分数和它的概率相乘
            action_value *= pi[row, col]

            #最后这个格子的分数,等于该格子下所有动作的分数求和
            new_values[row, col] = action_value.sum()

    return new_values


# print(get_values())


#策略提升
def get_pi():
    #重新初始化每个格子下采用动作的概率,重新评估
    new_pi = np.zeros([4, 12, 4])

    #遍历所有格子
    for row in range(4):
        for col in range(12):

            #计算当前格子4个动作分别的分数
            action_value = np.zeros(4)

            #遍历所有动作
            for action in range(4):
                action_value[action] = get_qsa(row, col, action)

            #计算当前状态下，达到最大分数的动作有几个
            count = (action_value == action_value.max()).sum()

            #让这些动作均分概率
            for action in range(4):
                if action_value[action] == action_value.max():
                    new_pi[row, col, action] = 1 / count
                else:
                    new_pi[row, col, action] = 0

    return new_pi


# print(get_pi())


# for _ in range(2):
#     values = get_values()
#     print(values)


#循环迭代策略评估和策略提升,寻找最优解
for _ in range(10):
    for _ in range(100):
        values = get_values()
    pi = get_pi()


#打印游戏，方便测试
def show(row, col, action):
    graph = [
        '□', '□', '□', '□', '□', '□', '□', '□', '□', '□', '□', '□', '□', '□',
        '□', '□', '□', '□', '□', '□', '□', '□', '□', '□', '□', '□', '□', '□',
        '□', '□', '□', '□', '□', '□', '□', '□', '□', '○', '○', '○', '○', '○',
        '○', '○', '○', '○', '○', '❤'
    ]

    action = {0: '↑', 1: '↓', 2: '←', 3: '→'}[action]

    graph[row * 12 + col] = action

    graph = ''.join(graph)

    for i in range(0, 4 * 12, 12):
        print(graph[i:i + 12])


# show(1, 1, 0)



from IPython import display
import time


def test():
    #起点在0,0
    row = 0
    col = 0

    #最多玩N步
    for _ in range(200):

        #选择一个动作
        action = np.random.choice(np.arange(4), size=1, p=pi[row, col])[0]

        #打印这个动作
        display.clear_output(wait=True)
        time.sleep(0.1)
        show(row, col, action)

        #执行动作
        row, col, reward = move(row, col, action)

        #获取当前状态，如果状态是终点或者掉陷阱则终止
        if get_state(row, col) in ['trap', 'terminal']:
            break


test()

