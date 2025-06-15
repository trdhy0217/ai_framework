import os
import time
import argparse
import numpy as np
from scipy.io import savemat, loadmat

from mindspore import nn, ops, load_checkpoint, load_param_into_net, set_seed
from mindflow.utils import load_yaml_config

from src import my_test_dataset, AEnet, save_loss_curve

np.random.seed(0)
set_seed(0)


def prediction():
    # 加载配置文件，准备参数
    config = load_yaml_config(args.config_file_path)  # 从 YAML 配置文件中读取所有设置
    data_params = config["data"]                      # 数据相关参数字典
    model_params = config["model"]                    # 模型相关参数字典
    prediction_params = config["prediction"]          # 推理相关参数字典
    prediction_result_dir = prediction_params["prediction_result_dir"]  # 保存一次性预测结果的目录
    pred_continue_dir = prediction_params["pred_continue_dir"]          # 保存循环预测结果的目录

    # 构建网络模型
    net = AEnet(
        in_channels=model_params["in_channels"],          # 输入通道数
        num_layers=model_params["num_layers"],            # 网络层数
        kernel_size=model_params["kernel_size"],          # 卷积核大小
        num_convlstm_layers=model_params["num_convlstm_layers"]  # ConvLSTM 层数
    )
    # 加载模型参数
    m_state_dict = load_checkpoint(prediction_params["ckpt_path"])  # 从指定路径加载 checkpoint
    load_param_into_net(net, m_state_dict)  # 将参数加载到网络中

    # 准备测试数据集
    data_set = my_test_dataset(prediction_params["data_dir"], data_params["time_steps"])  # 创建测试数据迭代器
    # 若保存结果的目录不存在，则创建
    if not os.path.exists(prediction_result_dir):
        os.mkdir(prediction_result_dir)
    if not os.path.exists(pred_continue_dir):
        os.mkdir(pred_continue_dir)

    # 定义损失函数：均方误差 MSE
    loss_func = nn.MSELoss()

    # 保存各步的测试损失
    test_losses = []

    # 根据推理模式执行不同流程
    # 单步预测模式：predict next one-step flow field
    if args.infer_mode == "one":
        for i, (input_1, velocity, label, matrix_01) in enumerate(data_set):
            pred = net(input_1, velocity)                   # 前向推理，获取预测
            pred = ops.mul(pred, matrix_01)                  # 对预测结果应用掩码或变换矩阵
            loss = ops.sqrt(loss_func(pred, label))          # 计算预测值与真实值的均方根误差
            test_losses.append(loss)                         # 记录损失张量
            print(f"test loss: {(loss.asnumpy().item()):.6f}")  # 打印当前损失
            # 保存预测结果与真实值到 mat 文件
            savemat(f"{prediction_result_dir}/prediction_data{i}.mat", {
                'prediction': pred.asnumpy(),
                'real': label.asnumpy(),
            })

    # 循环预测模式：predict a complete periodic flow field
    elif args.infer_mode == "cycle":
        for i, (inputvar, velocityvar, targetvar, matrix_01) in enumerate(data_set):
            if i == 0:
                inputs = inputvar  # 第一帧输入
            label = targetvar      # 真值
            velocity = velocityvar # 流场速度
            # 前向推理
            pred = net(inputs, velocity)
            pred = ops.mul(pred, matrix_01)                   # 应用矩阵变换
            loss = ops.sqrt(loss_func(pred, label))           # 计算均方根误差
            loss_aver = loss.asnumpy().item()                  # 转为 Python 数值

            # 记录并打印损失
            test_losses.append(loss_aver)
            print(f"test loss: {loss_aver:.6f}")
            # 保存循环预测结果
            savemat(f"{pred_continue_dir}/prediction_data{i}.mat", {
                'prediction': pred.asnumpy(),
                'real': label.asnumpy(),
            })
            # 将当前预测结果拼接为下一步的输入
            pred = ops.operations.ExpandDims()(pred, 1)       # 扩展维度以匹配输入格式
            cat = ops.concat((inputs, pred), axis=1)          # 在时间维度上连接张量
            inputs = cat[:, 1:, :, :, :]                     # 去掉最老的一帧，保留最新序列

    # 绘制并保存测试损失曲线
    save_loss_curve(
        test_losses,        # 损失列表
        'Epoch',            # 横轴标签
        'test_losses',      # 纵轴标签
        'Test_losses Curve',# 图表标题
        'Test_losses.png'   # 输出文件名
    )


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="GRAPH")           # 运行模式：GRAPH/其他
    parser.add_argument("--device_target", type=str, default="Ascend")# 设备目标：Ascend/GPU/CPU
    parser.add_argument("--device_id", type=int, default=0)              # 设备 ID
    parser.add_argument("--config_file_path", type=str, default="./config.yaml")  # 配置文件路径
    parser.add_argument("--infer_mode", type=str, default="one")       # 推理模式：one 或 cycle

    args = parser.parse_args()
    start_time = time.time()  # 记录开始时间
    prediction()              # 调用预测函数
    print(f"total time: {(time.time() - start_time):.2f}s")  # 打印总耗时
