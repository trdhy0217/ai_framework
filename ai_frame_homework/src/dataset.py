from collections import namedtuple

import numpy as np
import h5py

import mindspore.dataset as ds


class TrainDatasetSource:
    """
    数据集源类：负责从文件中加载数据并按训练/验证集比例划分
    """
    def __init__(self, data_dir, dataset_list, ratio=0.8):
        self.data_dir = data_dir            # 数据根目录
        self.dataset_list = dataset_list    # 不同子数据集的标识列表
        self.ratio = ratio                  # 训练集所占比例（其余为验证集）

    def train_data(self):
        """ 读取并划分训练集和验证集数据 """
        train_dataset = []   # 存放各子集的训练流场数据
        valid_dataset = []   # 存放各子集的验证流场数据
        train_velocity = []  # 存放各子集的训练速度场数据
        valid_velocity = []  # 存放各子集的验证速度场数据

        # 遍历每个子数据集
        for i in self.dataset_list:
            # 加载投影后的速度场数据 total_puv
            data_path = f"{self.data_dir}/f0.90h{i}/project/total_puv_project.mat"
            data_source = h5py.File(data_path, 'r')
            data_sample = data_source['total_puv'][:]  # 原始 shape: (T, H, W, C)
            # 转换为 (T, C, H, W) 并转换为 float32
            data_sample = np.array(data_sample.transpose([0, 3, 1, 2]), np.float32)

            # 根据比例划分训练/验证集
            data_length = data_sample.shape[0]
            split_idx = int(data_length * self.ratio)
            train_dataset.append(data_sample[:split_idx])
            valid_dataset.append(data_sample[split_idx:])

            # 加载对应的速度场数据 velocity
            vel_path = f"{self.data_dir}/f0.90h{i}/project/velocity.mat"
            vel_source = h5py.File(vel_path, 'r')
            data_velocity = vel_source['velocity'][:]  # shape: (T, ...)
            data_velocity = np.array(data_velocity, np.float32)

            train_velocity.append(data_velocity[:split_idx])
            valid_velocity.append(data_velocity[split_idx:])

        # 使用 namedtuple 统一返回格式
        DatasetResult = namedtuple('DatasetResult',
                                   ['train_dataset', 'train_velocity', 'valid_dataset', 'valid_velocity'])

        return DatasetResult(train_dataset, train_velocity, valid_dataset, valid_velocity)


class TrainDatasetMake:
    """
    训练/验证数据集生成器：根据时间步长构造输入与标签
    """
    def __init__(self, dataset, velocity, time_steps, dataset_list):
        self.dataset = dataset            # 列表，每个元素是一个子集的 ndarray，shape (T, C, H, W)
        self.velocity = velocity          # 列表，每个元素是对应的速度场数据 ndarray
        self.time_steps = time_steps      # 使用多少个连续时间步作为输入
        self.dataset_numbers = len(dataset_list)  # 子集数量，用于计算 __len__

    def __len__(self):
        # 每个子集可生成的样本数 = T - time_steps
        return (len(self.dataset[0]) - self.time_steps) * self.dataset_numbers

    def __getitem__(self, idx):
        # 根据全局索引计算属于哪个子集及在该子集内的偏移
        per_set = len(self.dataset[0]) - self.time_steps
        idx_dataset = idx // per_set
        idx_in_set = idx % per_set

        # 构造输入序列 (time_steps 步) 和标签（下一步）
        input_seq = self.dataset[idx_dataset][idx_in_set:idx_in_set + self.time_steps]
        vel_seq = self.velocity[idx_dataset][idx_in_set:idx_in_set + self.time_steps]
        label = self.dataset[idx_dataset][idx_in_set + self.time_steps]

        return input_seq, vel_seq, label


def my_train_dataset(data_dir, time_steps, dataset_list, batch_size):
    """
    构建 MindSpore 训练与验证 Dataset
    """
    print("batch_size", batch_size)
    # 获取划分好的原始数据
    train_data, train_velocity, valid_data, valid_velocity = \
        TrainDatasetSource(data_dir, dataset_list).train_data()

    # 构造训练集 Dataset
    train_dataset = TrainDatasetMake(train_data, train_velocity, time_steps, dataset_list)
    train_dataset = ds.GeneratorDataset(train_dataset, ["inputs", "v", "labels"], shuffle=True)
    train_dataset = train_dataset.batch(batch_size=batch_size, drop_remainder=True)

    # 构造验证集 Dataset
    valid_dataset = TrainDatasetMake(valid_data, valid_velocity, time_steps, dataset_list)
    valid_dataset = ds.GeneratorDataset(valid_dataset, ["inputs", "v", "labels"], shuffle=False)
    valid_dataset = valid_dataset.batch(batch_size=batch_size, drop_remainder=True)

    return train_dataset, valid_dataset


class TestDatasetMake:
    """
    测试数据集生成器：用于预测/推理阶段的单帧或循环预测
    """
    def __init__(self, dataset, velocity, matrix_01, time_steps):
        self.dataset = dataset            # ndarray, shape (T, C, H, W)
        self.velocity = velocity          # ndarray, shape (T, ...)
        self.matrix_01 = matrix_01        # 掩码矩阵 ndarray, shape (T, C, H, W)
        self.time_steps = time_steps

    def __len__(self):
        return len(self.dataset) - self.time_steps  # 可生成样本数

    def __getitem__(self, idx):
        # 构造测试输入、速度、真实标签及掩码矩阵
        test_input = self.dataset[idx:idx + self.time_steps]
        test_velocity = self.velocity[idx:idx + self.time_steps]
        test_label = self.dataset[idx + self.time_steps]
        test_matrix_01 = self.matrix_01[idx + self.time_steps]

        TestDatasetResult = namedtuple('TestDatasetResult',
                                       ['test_input', 'test_velocity', 'test_label', 'test_matrix_01'])

        return TestDatasetResult(test_input, test_velocity, test_label, test_matrix_01)


def my_test_dataset(data_dir, time_steps):
    """
    构造 MindSpore 测试 Dataset，仅读取前 10 帧进行示例
    """
    # 加载并预处理测试流场数据
    ds_path = f"{data_dir}/project/total_puv_project.mat"
    with h5py.File(ds_path, 'r') as f:
        data_sample = f['total_puv'][0:10]
    test_data = np.array(data_sample.transpose([0, 3, 1, 2]), np.float32)

    # 加载并预处理速度场数据
    vel_path = f"{data_dir}/project/velocity.mat"
    with h5py.File(vel_path, 'r') as f:
        vel_sample = f['velocity'][0:10]
    test_velocity = np.array(vel_sample, np.float32)

    # 加载并预处理掩码矩阵
    mat_path = f"{data_dir}/project/Matrix_01.mat"
    with h5py.File(mat_path, 'r') as f:
        mat_sample = f['Matrix'][0:10]
    test_matrix_01 = np.array(mat_sample.transpose([0, 3, 1, 2]), np.float32)

    # 创建测试集并批处理
    test_dataset = TestDatasetMake(test_data, test_velocity, test_matrix_01, time_steps)
    test_dataset = ds.GeneratorDataset(test_dataset,
                                       ["input", "velocity", "label", "matrix_01"],
                                       shuffle=False)
    test_dataset = test_dataset.batch(batch_size=1, drop_remainder=True)

    return test_dataset
