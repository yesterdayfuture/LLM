

'''
绘制 向量中 tensor 的 3d 可视化图
'''

import torch
import numpy as np  # 导入numpy库，用于数值计算
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot库，用于绘图
from mpl_toolkits.mplot3d import Axes3D  # 从mpl_toolkits.mplot3d导入Axes3D，用于3D绘图

# 定义函数visualize_tensor，用于可视化张量（tensor），参数tensor为待可视化的张量，batch_spacing为批次之间的间隔，默认为3
def visualize_tensor(tensor, batch_spacing=3):
    # 创建一个新的图形
    fig = plt.figure()
    # 在图形中添加一个3D子图，111表示1行1列第1个子图，projection='3d'表示3D投影
    ax = fig.add_subplot(111, projection='3d')

    # 遍历tensor的每一个batch（批次）
    for batch in range(tensor.shape[0]):
        # 遍历tensor的每一个channel（通道）
        for channel in range(tensor.shape[1]):
            # 遍历tensor的每一个height（高度）
            for i in range(tensor.shape[2]):
                # 遍历tensor的每一个width（宽度）
                for j in range(tensor.shape[3]):
                    # 计算当前元素的x坐标，j为宽度索引，batch * (tensor.shape[3] + batch_spacing)用于在批次之间添加间隔
                    x = j + (batch * (tensor.shape[3] + batch_spacing))
                    # y坐标为高度索引i（注意：原代码中y和z的位置写反了，这里已更正）
                    y = i
                    # z坐标为通道索引channel
                    z = channel
                    # 如果tensor中当前位置的值为0，则颜色设为灰色，否则为红色
                    color = 'red' if tensor[batch, channel, i, j] == 0 else 'gray'
                    # 使用ax.bar3d绘制3D柱状图，参数依次为x, y, z坐标，dx, dy, dz为柱体的长宽高，shade=True表示有阴影，color为颜色，edgecolor为边框颜色，alpha为透明度
                    ax.bar3d(x, y, z, 1, 1, 1, shade=True, color=color, edgecolor="black", alpha=0.9)  # 注意：原代码中alpha值写为0.0可能是一个错误，这里更正为0.9以便更好地观察

    # 设置x轴标签为'Width'（宽度）
    ax.set_xlabel('Width')
    # 设置y轴标签为'Height'（高度），原代码中错误地写为了'B & C'，这里已更正
    ax.set_ylabel('Height')
    # 设置z轴标签为'Channel'（通道），原代码中写为了'Height'，但根据上下文，这里应该表示通道，因此已更正
    ax.set_zlabel('Channel')
    # 设置x轴的范围，使用[::-1]是为了反转x轴的方向，可能是为了适应特定的数据展示需求
    # 注意：原代码中此处理的是z轴（但标签写为x轴），且注释有误，这里根据上下文和代码逻辑进行更正，但保留原注释以说明可能的误解来源
    # ax.set_xlim(ax.get_xlim()[::-1])  # 原代码，已更正为处理x轴本身
    ax.set_xlim(0, tensor.shape[3] * (tensor.shape[0] + batch_spacing) if tensor.shape[0] > 0 else 1)  # 更正后的x轴范围设置
    # 设置z轴标签的位置，以便更好地显示
    ax.zaxis.labelpad = 15

    # 显示图形
    plt.show()


tensor_data = torch.randn(1,3, 4, 4)  # 创建一个随机的4D张量，形状为(2,3,4,4)
tensor_data[0,1] = 0
visualize_tensor(tensor_data)

