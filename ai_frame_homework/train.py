import os
import time
import argparse
import numpy as np

from mindspore import nn, ops, context, save_checkpoint, set_seed, data_sink, jit

from mindflow.utils import load_yaml_config

from src import my_train_dataset, AEnet, save_loss_curve

np.random.seed(0)
set_seed(0)

def train():
    # 加载配置文件，准备参数
    config = load_yaml_config(args.config_file_path)  # 从 YAML 配置文件中读取所有设置
    data_params = config["data"]                      # 数据相关参数字典
    model_params = config["model"]                    # 模型相关参数字典
    optimizer_params = config["optimizer"]            # 优化器相关参数字典

    # 准备保存模型检查点的目录
    ckpt_dir = optimizer_params["ckpt_dir"]
    if not os.path.exists(ckpt_dir):                  # 若目录不存在则创建
        os.mkdir(ckpt_dir)

    # 构建待训练模型
    model = AEnet(
        in_channels=model_params["in_channels"],      # 输入通道数
        num_layers=model_params["num_layers"],        # 网络层数
        kernel_size=model_params["kernel_size"],      # 卷积核大小
        num_convlstm_layers=model_params["num_convlstm_layers"]  # ConvLSTM 层数
    )
    # 定义损失函数：均方误差 MSE
    loss_func = nn.MSELoss()
    # 定义优化器：Adam
    optimizer = nn.Adam(
        params=model.trainable_params(),               # 可训练参数列表
        learning_rate=optimizer_params["lr"]          # 学习率
    )

    # 如果使用 Ascend 设备，启用动态损失缩放和自动混合精度
    if use_ascend:
        from mindspore.amp import DynamicLossScaler, auto_mixed_precision, all_finite
        loss_scaler = DynamicLossScaler(1024, 2, 100)   # 初始化动态损失缩放器
        auto_mixed_precision(model, 'O1')               # 开启 O1 级别混合精度
    else:
        loss_scaler = None

    # 前向计算函数，返回损失
    def forward_fn(inputs, velocity, label):
        pred = model(inputs, velocity)                # 前向推理
        loss = loss_func(pred, label)                  # 计算 MSE 损失

        if use_ascend:
            loss = loss_scaler.scale(loss)             # 缩放损失
        return loss

    # 计算损失和梯度的函数
    grad_fn = ops.value_and_grad(
        forward_fn, None, optimizer.parameters, has_aux=False
    )

    # 准备训练和验证数据集
    dataset_train, dataset_eval = my_train_dataset(
        data_params["data_dir"],                     # 数据路径
        data_params["time_steps"],                   # 时间步长
        args.data_list,                                # 数据列表（如不同参数设置）
        batch_size=args.batch_size                     # 批大小
    )

    # 定义带 @jit 加速的训练步骤函数
    @jit
    def train_step(inputs, velocity, label):
        loss, grads = grad_fn(inputs, velocity, label)  # 计算损失和梯度
        if use_ascend:
            loss = loss_scaler.unscale(loss)            # 反缩放损失
            if all_finite(grads):                        # 若梯度有效，则反缩放梯度
                grads = loss_scaler.unscale(grads)
        # 应用更新并返回损失
        loss = ops.depend(loss, optimizer(grads))
        return loss

    # 定义带 @jit 加速的验证步骤函数，计算并返回 RMSE 损失
    @jit
    def eval_step(inputs, velocity, label):
        loss = forward_fn(inputs, velocity, label)     # 计算 MSE 损失
        loss = ops.sqrt(loss)                          # 转为 RMSE
        return loss

    # 使用数据下沉机制提升性能
    train_sink_process = data_sink(train_step, dataset_train, sink_size=1)
    eval_sink_process = data_sink(eval_step, dataset_eval, sink_size=1)
    train_data_size = dataset_train.get_dataset_size()  # 训练集批数
    eval_data_size = dataset_eval.get_dataset_size()    # 验证集批数

    # 用于记录每个 epoch 的平均损失
    avg_train_losses = []
    avg_valid_losses = []

    # 开始循环训练
    for epoch in range(1, optimizer_params["epochs"] + 1):
        train_losses = 0.0
        valid_losses = 0.0

        # 记录训练开始时间
        local_time_beg = time.time()
        model.set_train(True)                            # 切换到训练模式

        # 遍历所有训练批次
        for _ in range(train_data_size):
            step_train_loss = ops.squeeze(train_sink_process(), axis=())  # 获取标量损失
            step_train_loss = step_train_loss.asnumpy().item()            # 转为 Python 数值
            train_losses += step_train_loss

        # 计算并记录平均训练损失
        train_loss = train_losses / train_data_size
        avg_train_losses.append(train_loss)
        print(f"epoch: {epoch}, epoch average train loss: {train_loss:.6f}, "
              f"epoch time: {(time.time() - local_time_beg):.2f}s")

        # 每隔一定间隔进行验证
        if epoch % optimizer_params["eval_interval"] == 0:
            eval_time_beg = time.time()
            model.set_train(False)                       # 切换到评估模式
            for _ in range(eval_data_size):
                step_eval_loss = ops.squeeze(eval_sink_process(), axis=())
                step_eval_loss = step_eval_loss.asnumpy().item()
                valid_losses += step_eval_loss

            valid_loss = valid_losses / eval_data_size
            avg_valid_losses.append(valid_loss)
            print(f"epoch: {epoch}, epoch average valid loss: {valid_loss:.6f}, "
                  f"epoch time: {(time.time() - eval_time_beg):.2f}s")

        # 每隔一定间隔保存模型
        if epoch % optimizer_params["save_ckpt_interval"] == 0:
            save_checkpoint(model, f"{ckpt_dir}/net_{epoch}.ckpt")

    # 绘制并保存训练与验证损失曲线
    save_loss_curve(
        avg_train_losses, 'Epoch', 'avg_train_losses', 'Avg_train_losses Curve', 'Avg_train_losses.png'
    )
    save_loss_curve(
        avg_valid_losses, 'Epoch', 'avg_valid_losses', 'Avg_valid_losses Curve', 'Avg_valid_losses.png'
    )


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="GRAPH")
    parser.add_argument("--save_graphs", type=bool, default=False)
    parser.add_argument("--save_graphs_path", type=str, default="./summary")
    parser.add_argument("--device_target", type=str, default="Ascend")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--data_list", type=list, default=['0.25', '0.30', '0.35', '0.40'])
    parser.add_argument('--batch_size', type=int, default=16, help="batch size")
    parser.add_argument("--config_file_path", type=str, default="./config.yaml")

    args = parser.parse_args()

    # 设置运行上下文
    context.set_context(
        mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
        save_graphs=args.save_graphs,
        save_graphs_path=args.save_graphs_path,
        device_target=args.device_target,
        device_id=args.device_id
    )
    # 判断是否使用 Ascend 设备
    use_ascend = context.get_context(attr_key='device_target') == "Ascend"

    # 记录开始时间并启动训练
    start_time = time.time()
    train()
    print(f"total time: {(time.time() - start_time):.2f}s")