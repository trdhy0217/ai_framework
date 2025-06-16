from mindspore import nn, ops, numpy, float32
from mindspore.ops import operations as P


class ConvLSTMCell(nn.Cell):
    """
    单个 ConvLSTM 单元
    实现类似 LSTM 的门控机制，但在时空域使用卷积运算
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        # 输入通道数
        self.input_dim = input_dim
        # 隐藏状态通道数
        self.hidden_dim = hidden_dim
        # 卷积核大小，支持 (k_h, k_w)
        self.kernel_size = kernel_size
        # 是否使用偏置
        self.bias = bias
        # 用于在通道维度上拼接张量
        self.concat = P.Concat(axis=1)
        # 用于将卷积输出分割为四部分（i, f, o, g）
        self.split = P.Split(axis=1, output_num=4)

        # 定义卷积操作：输入拼接后通道为 input_dim + hidden_dim，输出为 4*hidden_dim
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              stride=1,
                              pad_mode="same",      # 保持输入输出尺寸一致
                              padding=0,
                              has_bias=self.bias,
                              data_format="NCHW")
        # 对卷积结果进行规范化，提高收敛稳定性
        self.norm = nn.BatchNorm2d(4 * self.hidden_dim)

    def construct(self, input_tensor, cur_state):
        """
        前向计算：接收当前输入和前一时刻的隐藏状态 h_cur, c_cur，输出新的 (h_next, c_next)
        input_tensor:    [batch, input_dim, H, W]
        cur_state: tuple([h_cur, c_cur])，均为 [batch, hidden_dim, H, W]
        """
        h_cur, c_cur = cur_state
        # 在通道维度上拼接当前输入和上次隐藏状态
        combined = ops.concat((input_tensor, h_cur), 1)
        # 卷积并归一化
        combined_conv = self.conv(combined)
        combined_conv = self.norm(combined_conv)
        # 分割成 4 部分，对应 i, f, o, g
        cc_i, cc_f, cc_o, cc_g = self.split(combined_conv)

        # 门控操作
        i = ops.sigmoid(cc_i)    # 输入门
        f = ops.sigmoid(cc_f)    # 遗忘门
        o = ops.sigmoid(cc_o)    # 输出门
        g = ops.tanh(cc_g)       # 候选记忆

        # 更新细胞状态和隐藏状态
        c_next = f * c_cur + i * g
        h_next = o * ops.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size, h_ini):
        """
        初始化隐藏状态 h 和记忆状态 c
        h_ini: 初始值标量或可广播张量，用于 h 的初始填充
        image_size: (height, width)
        返回: tuple(init_h, init_c)
        """
        height, width = image_size
        # 将标量 h_ini 重塑并广播到 [batch, hidden_dim, H, W]
        h_ini = numpy.reshape(h_ini, (batch_size, 1, 1, 1))
        h_ini = numpy.broadcast_to(h_ini, (batch_size, self.hidden_dim, height, width))
        # 初始化 h 为 h_ini 填充的全1张量，c 为全0张量
        init_h = h_ini * numpy.ones(shape=(batch_size, self.hidden_dim, height, width)).astype(float32)
        init_c = numpy.zeros(shape=(batch_size, self.hidden_dim, height, width)).astype(float32)
        return (init_h, init_c)


class ConvLSTM(nn.Cell):
    """
    多层 ConvLSTM 网络
    支持多层堆叠，每层输出作为下一层输入
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=True):
        super(ConvLSTM, self).__init__()
        # 校验 kernel_size 参数格式
        self._check_kernel_size_consistency(kernel_size)
        # 将 kernel_size, hidden_dim 扩展为与 num_layers 长度一致的列表
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)

        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim          # 输入特征通道数
        self.hidden_dim = hidden_dim        # 各层隐藏状态通道数列表
        self.kernel_size = kernel_size      # 各层卷积核大小列表
        self.num_layers = num_layers        # 网络层数
        self.batch_first = batch_first      # 输入维度顺序 (batch, seq, ...)
        self.bias = bias

        # 创建多个 ConvLSTMCell
        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
        # 使用 MindSpore 的 CellList 管理多个子模块
        self.cell_list = nn.CellList(cell_list)

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        """
        确保 kernel_size 为 tuple 或 list of tuples
        """
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        """
        若 param 不是 list，则复制为 num_layers 长度的列表
        """
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

    def construct(self, input_tensor, h0):
        """
        前向执行多层 ConvLSTM
        input_tensor: 5D 张量, 形状可为 [batch, seq_len, C, H, W] 或 [seq_len, batch, C, H, W]
        h0: 初始 hidden state 标量或张量
        返回: (layer_output_list, last_state_list)
        """
        # 若非 batch_first, 将形状转为 (batch, seq, ...)
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        batch_size, seq_len, _, height, width = input_tensor.shape
        # 初始化每层的 (h, c)
        hidden_state = self._init_hidden(batch_size=batch_size, image_size=(height, width), h_ini=h0)

        layer_output_list = []   # 保存每层按时间堆叠的输出
        last_state_list = []     # 保存最后时刻每层的 (h, c)

        cur_layer_input = input_tensor
        # 逐层处理
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            # 在时间维度上循环
            for t in range(seq_len):
                # 单步 ConvLSTMCell 计算
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                  cur_state=[h, c])
                output_inner.append(h)
            # 将当前层所有时间步输出堆叠
            layer_output = ops.stack(output_inner, axis=1)
            # 下一层的输入为当前层的输出
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size, h_ini):
        """
        为所有层生成初始隐藏状态列表
        """
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size, h_ini))
        return init_states
