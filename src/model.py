from mindspore import nn, ops

from .conv_lstm import ConvLSTM


class Encoder(nn.Cell):
    """
    编码器部分：多层卷积网络，用于提取时空特征并降维
    """
    def __init__(self, in_channels, num_layers, kernel_size, has_bias=True, weight_init='XavierUniform',
                 activation=nn.LeakyReLU()):
        super(Encoder, self).__init__()
        # 构建多层卷积操作列表
        layers = []
        for num in range(1, num_layers + 1):
            if num == 1:
                # 第一层：将输入通道映射到 2^(num+1) 通道，并降采样一半
                layers.extend([
                    nn.Conv2d(in_channels, 2 ** (num + 1), kernel_size,
                              stride=2, pad_mode='same', has_bias=has_bias,
                              weight_init=weight_init, data_format='NCHW'),
                    activation
                ])
            elif num % 2 == 0:
                # 偶数层：保持特征图尺寸不变，通道数不变
                channels = int(2 ** (num / 2 + 1))
                layers.extend([
                    nn.Conv2d(channels, channels, kernel_size - 1,
                              stride=1, pad_mode='same', has_bias=has_bias,
                              weight_init=weight_init, data_format='NCHW'),
                    activation
                ])
            elif num % 2 == 1:
                # 奇数层（除第一层）：进一步降采样并增加通道数
                in_ch = int(2 ** ((num + 1) / 2))
                out_ch = int(2 ** ((num + 3) / 2))
                layers.extend([
                    nn.Conv2d(in_ch, out_ch, kernel_size,
                              stride=2, pad_mode='same', has_bias=has_bias,
                              weight_init=weight_init, data_format='NCHW'),
                    activation
                ])
        # 将所有层封装为顺序容器
        self.convlayers = nn.SequentialCell(layers)

    def construct(self, x):
        """
        前向计算
        x: [batch*seq, C, H, W]
        返回: 特征图 [batch*seq, hidden_C, H', W']
        """
        x = self.convlayers(x)
        return x


class Decoder(nn.Cell):
    """
    解码器部分：多层反卷积与卷积网络，用于从隐状态恢复原始分辨率流场
    """
    def __init__(self, in_channels, num_layers, kernel_size, weight_init='XavierUniform', activation=nn.LeakyReLU()):
        super(Decoder, self).__init__()
        layers = []
        for num in range(1, num_layers + 1):
            if num == num_layers:
                # 最后一层：使用常规卷积恢复到输出通道数
                layers.extend([
                    nn.Conv2d(in_channels, in_channels, kernel_size + 1,
                              stride=1, pad_mode='same', padding=0,
                              weight_init=weight_init),
                    activation
                ])
            elif num == num_layers - 1:
                # 倒数第二层：使用反卷积上采样一次
                layers.extend([
                    nn.Conv2dTranspose(in_channels + 1, in_channels, kernel_size,
                                       stride=2, pad_mode='same', padding=0),
                    activation
                ])
            elif num % 2 == 1:
                # 其他奇数层：反卷积上采样并递减通道
                in_ch = int(2 ** ((15 - num) / 2))
                out_ch = int(2 ** ((13 - num) / 2))
                layers.extend([
                    nn.Conv2dTranspose(in_ch, out_ch, kernel_size,
                                       stride=2, pad_mode='same', weight_init=weight_init),
                    activation
                ])
            elif num % 2 == 0:
                # 偶数层：常规卷积，保持特征图大小和通道
                ch = int(2 ** ((14 - num) / 2))
                layers.extend([
                    nn.Conv2d(ch, ch, kernel_size - 1,
                              stride=1, pad_mode='same', weight_init=weight_init),
                    activation
                ])
        self.deconv_layers = nn.SequentialCell(layers)

    def construct(self, x):
        """
        前向计算
        x: [batch, hidden_C, H', W']
        返回: [batch, in_channels, H, W]
        """
        x = self.deconv_layers(x)
        return x


class AEnet(nn.Cell):
    """
    完整自编码器网络：Encoder + ConvLSTM + Decoder
    """
    def __init__(self,
                 in_channels,
                 num_layers,
                 kernel_size,
                 num_convlstm_layers):
        super(AEnet, self).__init__()
        # 编码器：提取特征并降采样
        self.encoder = Encoder(in_channels=in_channels,
                               num_layers=num_layers,
                               kernel_size=kernel_size)
        # ConvLSTM：时序特征建模，隐藏状态维度固定为 128
        self.convlstm = ConvLSTM(input_dim=128,
                                 hidden_dim=128,
                                 kernel_size=(3, 3),
                                 num_layers=num_convlstm_layers,
                                 batch_first=True,
                                 bias=True)
        # 解码器：恢复分辨率并生成预测
        self.decoder = Decoder(in_channels=in_channels,
                               num_layers=num_layers,
                               kernel_size=kernel_size)

    def construct(self, x, velocity):
        """
        前向计算流程：
        x: [batch, seq_len, C, H, W] 输入流场序列
        velocity: 用作 ConvLSTM 的初始隐藏状态或其他辅助信息
        返回: one-step 预测 [batch, C, H, W]
        """
        b, t, c, h, w = x.shape

        # 合并 batch 与时间维度，送入 Encoder
        con_in = ops.reshape(x, (b * t, c, h, w))
        con_out = self.encoder(con_in)  # [b*t, 128, h', w']

        # 还原为 5D 张量，作为 ConvLSTM 输入
        lstm_in = ops.reshape(con_out, (b, t,
                                        con_out.shape[1],
                                        con_out.shape[2],
                                        con_out.shape[3]))
        # 使用 velocity 作为隐藏状态初始值 h0
        _, last_states = self.convlstm(lstm_in, velocity)
        # 取第一层最后时刻的 h
        lstm_out = last_states[0][0]

        # 解码器重建到原始空间
        out = self.decoder(lstm_out)  # [b, C, H, W]

        return out