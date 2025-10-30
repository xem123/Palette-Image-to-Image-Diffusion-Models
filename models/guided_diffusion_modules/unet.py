from abc import abstractmethod
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import (
    checkpoint,
    zero_module,
    normalization,
    count_flops_attn,
    gamma_embedding
)
# 实现 SiLU 激活函数（Swish），比 ReLU 更平滑，有助于梯度传播
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# 抽象基类，定义接收嵌入作为条件的模块接口
class EmbedBlock(nn.Module):
    """
    Any module where forward() takes embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        应用模块到输入x，条件是emb嵌入
        Apply the module to `x` given `emb` embeddings.
        """
# 扩展nn.Sequential，允许将嵌入传递给支持它的子模块
class EmbedSequential(nn.Sequential, EmbedBlock):
    """
    A sequential module that passes embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, EmbedBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x
# 通过插值上采样，可选卷积调整通道数
class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.

    """
    def __init__(self, channels, use_conv, out_channel=None):
        super().__init__()
        self.channels = channels
        self.out_channel = out_channel or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channel, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x
# 通过卷积或平均池化下采样
class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """

    def __init__(self, channels, use_conv, out_channel=None):
        super().__init__()
        self.channels = channels
        self.out_channel = out_channel or channels
        self.use_conv = use_conv
        stride = 2
        if use_conv:
            self.op = nn.Conv2d(
                self.channels, self.out_channel, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channel
            self.op = nn.AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(EmbedBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of embedding channels.
    :param dropout: the rate of dropout.
    :param out_channel: if specified, the number of out channels.
    :param use_conv: if True and out_channel is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channel=None,
        use_conv=False,
        use_scale_shift_norm=False,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channel = out_channel or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            nn.Conv2d(channels, self.out_channel, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False)
            self.x_upd = Upsample(channels, False)
        elif down:
            self.h_upd = Downsample(channels, False)
            self.x_upd = Downsample(channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channel if use_scale_shift_norm else self.out_channel,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channel),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv2d(self.out_channel, self.out_channel, 3, padding=1)
            ),
        )

        if self.out_channel == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(
                channels, self.out_channel, 3, padding=1
            )
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channel, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)

class UNet(nn.Module):
    """
    The full UNet model with attention and embedding.
    :param in_channel: channels in the input Tensor, for image colorization : Y_channels + X_channels .
    :param inner_channel: base channel count for the model.
    :param out_channel: channels in the output Tensor.
    :param res_blocks: number of residual blocks per downsample.
    :param attn_res: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mults: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    带注意力机制和嵌入层的完整UNet模型
    :param in_channel: 输入张量的通道数，如图像着色时为Y通道+X通道
    :param inner_channel: 模型的基础通道数
    :param out_channel: 输出张量的通道数
    :param res_blocks: 每次下采样的残差块数量
    :param attn_res: 应用注意力机制的下采样率集合，如包含16则在16x下采样时使用注意力
    :param dropout: Dropout概率
    :param channel_mults: UNet各层的通道乘数
    :param conv_resample: 是否使用可学习卷积进行上/下采样
    :param use_checkpoint: 是否使用梯度检查点减少内存占用
    :param num_heads: 每个注意力层的注意力头数
    :param num_head_channels: 每个注意力头的固定通道数，若指定则忽略num_heads
    :param num_heads_upsample: 上采样时的注意力头数，已弃用
    :param use_scale_shift_norm: 是否使用类似FiLM的条件归一化机制
    :param resblock_updown: 是否使用残差块进行上/下采样
    :param use_new_attention_order: 是否使用新的注意力计算顺序以提高效率
    """

    # def __init__(
    #     self,
    #     image_size,  # 256 (图像尺寸)
    #     in_channel,  # 6 (输入通道数)
    #     inner_channel,  # 64 (基础通道数)
    #     out_channel,  # 3 (输出通道数)
    #     res_blocks,  # 2 (每层残差块数量)
    #     attn_res,  # 16 (应用注意力的分辨率)
    #     dropout=0,  # 0.2 (丢弃率)
    #     channel_mults=(1, 2, 4, 8),  # 通道乘数
    #     conv_resample=True,  # 是否使用卷积上/下采样
    #     use_checkpoint=False,  # 是否使用梯度检查点
    #     use_fp16=False,  # 是否使用半精度
    #     num_heads=1,  # 注意力头数
    #     num_head_channels=-1,  # 每个头的通道数
    #     num_heads_upsample=-1,  # 上采样头数
    #     use_scale_shift_norm=True,  # 是否使用缩放移位归一化
    #     resblock_updown=True,  # 是否用残差块上/下采样
    #     use_new_attention_order=False,  # 是否使用新的注意力顺序
    # ):
    #
    #     super().__init__()
    #     # 如果未指定上采样头数，则使用与下采样相同的头数
    #     if num_heads_upsample == -1:
    #         num_heads_upsample = num_heads
    #     # 保存所有模型参数
    #     self.image_size = image_size
    #     self.in_channel = in_channel
    #     self.inner_channel = inner_channel
    #     self.out_channel = out_channel
    #     self.res_blocks = res_blocks
    #     self.attn_res = attn_res
    #     self.dropout = dropout
    #     self.channel_mults = channel_mults
    #     self.conv_resample = conv_resample
    #     self.use_checkpoint = use_checkpoint
    #     self.dtype = torch.float16 if use_fp16 else torch.float32
    #     self.num_heads = num_heads
    #     self.num_head_channels = num_head_channels
    #     self.num_heads_upsample = num_heads_upsample
    #
    #     # 创建条件嵌入网络，将时间步转换为256维向量
    #     cond_embed_dim = inner_channel * 4 # 64*4 = 256
    #     self.cond_embed = nn.Sequential(
    #         nn.Linear(inner_channel, cond_embed_dim),  # 线性变换：64→256
    #         SiLU(),  # SiLU激活函数
    #         nn.Linear(cond_embed_dim, cond_embed_dim),  # 线性变换：256→256
    #     )
    #     # 设置初始通道数并创建输入处理模块
    #     ch = input_ch = int(channel_mults[0] * inner_channel) #64
    #     self.input_blocks = nn.ModuleList(
    #         [EmbedSequential(nn.Conv2d(in_channel, ch, 3, padding=1))]# 初始卷积层
    #     )
    #     self._feature_size = ch
    #     input_block_chans = [ch]
    #     ds = 1# 下采样率
    #     # 构建编码器路径（下采样部分）
    #
    #     for level, mult in enumerate(channel_mults):
    #         # 为每个分辨率级别添加残差块
    #         for _ in range(res_blocks):
    #             layers = [
    #                 ResBlock(
    #                     ch,
    #                     cond_embed_dim,
    #                     dropout,
    #                     out_channel=int(mult * inner_channel),
    #                     use_checkpoint=use_checkpoint,
    #                     use_scale_shift_norm=use_scale_shift_norm,
    #                 )
    #             ]
    #             ch = int(mult * inner_channel)
    #             if ds in attn_res:
    #                 layers.append(
    #                     AttentionBlock(
    #                         ch,
    #                         use_checkpoint=use_checkpoint,
    #                         num_heads=num_heads,
    #                         num_head_channels=num_head_channels,
    #                         use_new_attention_order=use_new_attention_order,
    #                     )
    #                 )
    #             self.input_blocks.append(EmbedSequential(*layers))
    #             self._feature_size += ch
    #             input_block_chans.append(ch)
    #         if level != len(channel_mults) - 1:
    #             out_ch = ch
    #             self.input_blocks.append(
    #                 EmbedSequential(
    #                     ResBlock(
    #                         ch,
    #                         cond_embed_dim,
    #                         dropout,
    #                         out_channel=out_ch,
    #                         use_checkpoint=use_checkpoint,
    #                         use_scale_shift_norm=use_scale_shift_norm,
    #                         down=True,
    #                     )
    #                     if resblock_updown
    #                     else Downsample(
    #                         ch, conv_resample, out_channel=out_ch
    #                     )
    #                 )
    #             )
    #             ch = out_ch
    #             input_block_chans.append(ch)
    #             ds *= 2
    #             self._feature_size += ch
    #
    #     self.middle_block = EmbedSequential(
    #         ResBlock(
    #             ch,
    #             cond_embed_dim,
    #             dropout,
    #             use_checkpoint=use_checkpoint,
    #             use_scale_shift_norm=use_scale_shift_norm,
    #         ),
    #         AttentionBlock(
    #             ch,
    #             use_checkpoint=use_checkpoint,
    #             num_heads=num_heads,
    #             num_head_channels=num_head_channels,
    #             use_new_attention_order=use_new_attention_order,
    #         ),
    #         ResBlock(
    #             ch,
    #             cond_embed_dim,
    #             dropout,
    #             use_checkpoint=use_checkpoint,
    #             use_scale_shift_norm=use_scale_shift_norm,
    #         ),
    #     )
    #     self._feature_size += ch
    #
    #     self.output_blocks = nn.ModuleList([])
    #     for level, mult in list(enumerate(channel_mults))[::-1]:
    #         for i in range(res_blocks + 1):
    #             ich = input_block_chans.pop()
    #             layers = [
    #                 ResBlock(
    #                     ch + ich,
    #                     cond_embed_dim,
    #                     dropout,
    #                     out_channel=int(inner_channel * mult),
    #                     use_checkpoint=use_checkpoint,
    #                     use_scale_shift_norm=use_scale_shift_norm,
    #                 )
    #             ]
    #             ch = int(inner_channel * mult)
    #             if ds in attn_res:
    #                 layers.append(
    #                     AttentionBlock(
    #                         ch,
    #                         use_checkpoint=use_checkpoint,
    #                         num_heads=num_heads_upsample,
    #                         num_head_channels=num_head_channels,
    #                         use_new_attention_order=use_new_attention_order,
    #                     )
    #                 )
    #             if level and i == res_blocks:
    #                 out_ch = ch
    #                 layers.append(
    #                     ResBlock(
    #                         ch,
    #                         cond_embed_dim,
    #                         dropout,
    #                         out_channel=out_ch,
    #                         use_checkpoint=use_checkpoint,
    #                         use_scale_shift_norm=use_scale_shift_norm,
    #                         up=True,
    #                     )
    #                     if resblock_updown
    #                     else Upsample(ch, conv_resample, out_channel=out_ch)
    #                 )
    #                 ds //= 2
    #             self.output_blocks.append(EmbedSequential(*layers))
    #             self._feature_size += ch
    #
    #     self.out = nn.Sequential(
    #         normalization(ch),
    #         SiLU(),
    #         zero_module(nn.Conv2d(input_ch, out_channel, 3, padding=1)),
    #     )

    def __init__(
            self,
            image_size,  # 256 (图像尺寸)
            in_channel,  # 6 (输入通道数)
            inner_channel,  # 64 (基础通道数)
            out_channel,  # 3 (输出通道数)
            res_blocks,  # 2 (每层残差块数量)
            attn_res,  # 16 (应用注意力的分辨率)
            dropout=0,  # 0.2 (丢弃率)
            channel_mults=(1, 2, 4, 8),  # 通道乘数
            conv_resample=True,  # 是否使用卷积上/下采样
            use_checkpoint=False,  # 是否使用梯度检查点
            use_fp16=False,  # 是否使用半精度
            num_heads=1,  # 注意力头数
            num_head_channels=-1,  # 每个头的通道数
            num_heads_upsample=-1,  # 上采样头数
            use_scale_shift_norm=True,  # 是否使用缩放移位归一化
            resblock_updown=True,  # 是否用残差块上/下采样
            use_new_attention_order=False,  # 是否使用新的注意力顺序
    ):

        super().__init__()

        # 如果未指定上采样头数，则使用与下采样相同的头数
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        # 保存所有模型参数
        self.image_size = image_size
        self.in_channel = in_channel
        self.inner_channel = inner_channel
        self.out_channel = out_channel
        self.res_blocks = res_blocks
        self.attn_res = attn_res
        self.dropout = dropout
        self.channel_mults = channel_mults
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        # 创建条件嵌入网络，将时间步转换为256维向量
        # 维度计算：inner_channel通常设为 64，因此cond_embed_dim=64×4=256。这一维度与论文中提到的条件扩散模型的时间步嵌入维度一致。
        # 论文对应关系：该模块对应论文中条件扩散模型的时间步条件处理，通过γ（噪声水平指示器）嵌入指导去噪过程。
        cond_embed_dim = inner_channel * 4  # 64*4 = 256
        self.cond_embed = nn.Sequential(
            nn.Linear(inner_channel, cond_embed_dim),  # 线性变换：第一个线性层将时间步的初始嵌入（如 64 维）映射到 256 维，增强特征表达能力
            SiLU(),  # SiLU激活函数：引入非线性，提升模型拟合能力
            nn.Linear(cond_embed_dim, cond_embed_dim),  # 线性变换：第二个线性层保持 256 维，确保条件嵌入与后续残差块的维度匹配
        )


        # 设置初始通道数并创建输入处理模块
        ch = input_ch = int(channel_mults[0] * inner_channel)  # 1*64，其中 "channel_mults": [1,2,4,8],
        # 初始卷积层： 3×3 卷积层
        self.input_blocks = nn.ModuleList(
            [EmbedSequential(nn.Conv2d(in_channel, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1  # 下采样率

        # 构建编码器路径（下采样部分）
        for level, mult in enumerate(channel_mults):# 通道数扩展：其中 "channel_mults": [1,2,4,8]定义各层通道数倍数，随下采样逐步增加通道数，捕捉更抽象特征。
            # 为每个分辨率级别添加残差块
            # 残差块（ResBlock）：每个分辨率级别包含"res_blocks": 2个残差块，用于提取层次化特征，同时通过cond_embed_dim融入时间步条件
            for _ in range(res_blocks): # 其中 "res_blocks": 2
                layers = [
                    ResBlock(
                        ch,  # 当前通道数
                        cond_embed_dim,  # 条件嵌入维度
                        dropout,  # Dropout率
                        out_channel=int(mult * inner_channel),  # 输出通道数
                        use_checkpoint=use_checkpoint,  # 是否使用梯度检查点
                        use_scale_shift_norm=use_scale_shift_norm,  # 是否使用缩放移位归一化
                    )
                ]
                ch = int(mult * inner_channel)  # 更新当前通道数

                # 当下采样率ds在注意力attn_res列表中时（"attn_res": [16]），添加自注意力层，增强全局依赖建模能力，对应论文中 “自注意力对性能至关重要” 的结论
                if ds in attn_res:# 其中"attn_res": [16]
                    layers.append(
                        AttentionBlock(
                            ch,  # 当前通道数
                            use_checkpoint=use_checkpoint,  # 是否使用梯度检查点
                            num_heads=num_heads,  # 注意力头数
                            num_head_channels=num_head_channels,  # 每个头的通道数
                            use_new_attention_order=use_new_attention_order,  # 是否使用新的注意力顺序
                        )
                    )
                # 将构建的层添加到输入模块列表
                self.input_blocks.append(EmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)  # 保存通道数以用于解码器

            # 如果不是最后一个级别，添加下采样层
            if level != len(channel_mults) - 1:
                out_ch = ch
                self.input_blocks.append(
                    EmbedSequential(
                        # 使用残差块进行下采样或普通下采样层
                        ResBlock(
                            ch,
                            cond_embed_dim,
                            dropout,
                            out_channel=out_ch,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,  # 下采样标志
                        )
                        if resblock_updown # 默认True，当resblock_updown=True时，用带down=True标志的残差块实现下采样
                        # 否则用Downsample模块（如步长 2 的卷积或平均池化）
                        else Downsample(
                            ch, conv_resample, out_channel=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2  # 下采样率ds每次翻倍（如 1→2→4→8），对应图像尺寸减半（256×256→128×128→…）
                self._feature_size += ch

        # 构建中间层（瓶颈部分）：两个残差块包裹一个自注意力块，形成瓶颈层，确保特征提取的完整性
        # 自注意力块在此处捕捉最高维度特征的全局依赖（如 8×8 分辨率下的全局关系）
        # 维度保持：通道数ch与编码器最后一层一致，维持特征维度不变，为解码器上采样做准备
        self.middle_block = EmbedSequential(
            ResBlock(
                ch,
                cond_embed_dim,
                dropout,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                cond_embed_dim,
                dropout,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        # 构建解码器路径（上采样部分）
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mults))[::-1]:
            for i in range(res_blocks + 1):
                ich = input_block_chans.pop()  # 跳层连接，从编码器保存的通道列表中取出对应层特征，与当前解码器特征拼接（ch + ich），恢复底层空间信息，提升生成细节保真度
                layers = [
                    ResBlock(
                        ch + ich,  # 当前特征与编码器特征拼接后的通道数
                        cond_embed_dim,
                        dropout,
                        out_channel=int(inner_channel * mult),
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(inner_channel * mult)  # 更新当前通道数

                # 如果当前采样率在注意力列表中，添加注意力块
                if ds in attn_res:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )

                # 上采样层（非最后一层且处理完残差块时）
                if level and i == res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            cond_embed_dim,
                            dropout,
                            out_channel=out_ch,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,  # 上采样标志
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, out_channel=out_ch)
                    )
                    ds //= 2  # 上采样率减半
                self.output_blocks.append(EmbedSequential(*layers))
                self._feature_size += ch

        # 构建输出层
        self.out = nn.Sequential(
            normalization(ch),  # 归一化层
            SiLU(),  # SiLU激活函数
            zero_module(nn.Conv2d(input_ch, out_channel, 3, padding=1)),  # 零初始化卷积层
        )

    def forward(self, x, gammas):
        """
        Apply the model to an input batch.
        :param x: an [N x 2 x ...] Tensor of inputs (B&W)
        :param gammas: a 1-D batch of gammas.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        gammas = gammas.view(-1, )
        emb = self.cond_embed(gamma_embedding(gammas, self.inner_channel))

        h = x.type(torch.float32)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)

if __name__ == '__main__':
    b, c, h, w = 3, 6, 64, 64
    timsteps = 100
    model = UNet(
        image_size=h,
        in_channel=c,
        inner_channel=64,
        out_channel=3,
        res_blocks=2,
        attn_res=[8]
    )
    x = torch.randn((b, c, h, w))
    emb = torch.ones((b, ))
    out = model(x, emb)