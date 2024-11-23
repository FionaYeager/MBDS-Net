import torch.utils.data
import torch
import torch.nn as nn
import torch.nn.functional as F
from guided_diffusion.utils import dice_score
import math
from abc import abstractmethod
from guided_diffusion.con_unet import ConUNet
from guided_diffusion.DCA import DCA
# torch.autograd.set_detect_anomaly(True)

def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs

class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0, dilation, groups=inplanes, bias=bias)
        self.bn = nn.BatchNorm2d(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, reps=3, stride=1, dilation=1, start_with_relu=False, grow_first=True):
        super(Block, self).__init__()

        if out_channels != in_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            self.skipbn = nn.BatchNorm2d(out_channels)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=False)
        rep = []

        filters = in_channels
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_channels, out_channels, 3, 1, dilation))
            rep.append(nn.BatchNorm2d(out_channels))
            filters = out_channels

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, 1, dilation))
            rep.append(nn.BatchNorm2d(filters))
            rep.append(nn.ReLU(inplace=False))

        if not start_with_relu:
            rep = rep[1:]

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x = x + skip

        return x

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def normalization(channels):
    return GroupNorm32(16, channels)

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, emb_channels):
        super(conv_block, self).__init__()

        self.inconv = nn.Sequential(
            # normalization(in_ch),
            # nn.SiLU(),
            # ODConv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),#True
            # SeparableConv2d(in_ch, out_ch, kernel_size=3, stride=1, bias=True),
            # nn.BatchNorm2d(out_ch),
            normalization(out_ch),
            nn.ReLU(inplace=True),
        )

        self.outconv = nn.Sequential(
            # normalization(out_ch),
            # nn.SiLU(),
            # nn.Dropout(p=0.1),
            # ODConv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),#True
            # SeparableConv2d(out_ch, out_ch, kernel_size=3, stride=1, bias=True),
            # nn.BatchNorm2d(out_ch),
            normalization(out_ch),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.2)
            )

        self.emb_layers = nn.Sequential(
            # nn.SiLU(),
            nn.ReLU(),
            nn.Linear(
                emb_channels,
                out_ch,
            )
        )
        # self.SCA = CA(out_ch)
        # self.PEE = PEE(out_ch, out_ch, [3, 5, 7])
        if in_ch == out_ch:
            # self.skip_connection = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)
            self.skip_connection = nn.Identity()
        else:
            # self.skip_connection = SeparableConv2d(in_ch, out_ch, kernel_size=3, stride=1, bias=True)
            self.skip_connection = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)


    def forward(self, x, emb):
        h = self.inconv(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        h = h + emb_out
        h = self.outconv(h)
        # h = self.SCA(h)
        h = h + self.skip_connection(x)
        # h = self.PEE(h)
        # h = self.SCA(h)
        return h

class conv_block2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block2, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        h = self.conv(x)
        return h

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    # print("half:", half)#64
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        # self.uu = nn.Upsample(scale_factor=2)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            # normalization(in_ch),
            # nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),#True
            # nn.BatchNorm2d(out_ch),
            normalization(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x = self.uu(x)
        h = self.up(x)
        # h = self.outconv(h)
        # h = h + self.skip_connection(x)
        return h

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x, emb)
        return x

class conv_block1(nn.Module):
    def __init__(self, in_ch, out_ch, emb_channels):
        super(conv_block1, self).__init__()

        self.inconv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False, groups=in_ch),#, bias=True,group,输出in
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),#, bias=True,group,输出in
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.outconv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False, groups=out_ch),
            # , bias=True,group,输出in
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),  # , bias=True,group,输出in
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
            )

        self.emb_layers = nn.Sequential(
            # nn.SiLU(),
            nn.ReLU(),
            nn.Linear(
                emb_channels,
                out_ch,
            )
        )
        if in_ch == out_ch:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False, groups=in_ch),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),  # , bias=True,group,输出in
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

    def forward(self, x, emb):
        # h = self.SCA(x)
        h = self.inconv(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        h = h + emb_out
        h = self.outconv(h)
        h = h + self.skip_connection(x)
        return h

class UnetDsv3(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(UnetDsv3, self).__init__()
        self.dsv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.Upsample(size=scale_factor)
        )

    def forward(self, input):
        return self.dsv(input)

class UNet(nn.Module):

    def __init__(self, n_channels, out_channels):
        super(UNet, self).__init__()
        self.ConUNet = ConUNet(3, 1)
        self.n_channels = n_channels
        self.out_channels = out_channels
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]#, n1 * 16
        self.DCA = DCA(n=1,
                       features=[int(32), int(64), int(128), int(256)],
                       strides=[8, 8 // 2, 8 // 4, 8 // 8],
                       patch=32,
                       )
        self.filter = filters[0]
        time_embed_dim = filters[0] * 4
        self.time_embed = nn.Sequential(
            nn.Linear(filters[0], time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv0 = nn.Conv2d(n_channels, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.Conv1 = TimestepEmbedSequential(conv_block(16, filters[0], time_embed_dim))
        self.Conv1_2 = TimestepEmbedSequential(conv_block(filters[0], filters[0], time_embed_dim))
        self.Conv1_3 = TimestepEmbedSequential(conv_block(filters[0], filters[0], time_embed_dim))

        self.Conv2 = TimestepEmbedSequential(conv_block(filters[0], filters[1], time_embed_dim))#2*
        self.Conv2_2 = TimestepEmbedSequential(conv_block(filters[1], filters[1], time_embed_dim))
        self.Conv2_3 = TimestepEmbedSequential(conv_block(filters[1], filters[1], time_embed_dim))

        self.Conv3 = TimestepEmbedSequential(conv_block(filters[1], filters[2], time_embed_dim))#2*
        self.Conv3_2 = TimestepEmbedSequential(conv_block(filters[2], filters[2], time_embed_dim))
        self.Conv3_3 = TimestepEmbedSequential(conv_block(filters[2], filters[2], time_embed_dim))

        self.Conv4 = TimestepEmbedSequential(conv_block(filters[2], filters[3], time_embed_dim))#2*
        self.Conv4_2 = TimestepEmbedSequential(conv_block(filters[3], filters[3], time_embed_dim))
        self.Conv4_3 = TimestepEmbedSequential(conv_block(filters[3], filters[3], time_embed_dim))

        self.Conv5 = TimestepEmbedSequential(conv_block(filters[3], filters[4], time_embed_dim))#2* 去掉第五层
        self.Conv5_2 = TimestepEmbedSequential(conv_block(filters[4], filters[4], time_embed_dim))
        self.Conv5_3 = TimestepEmbedSequential(conv_block(filters[4], filters[4], time_embed_dim))

        self.Up5 = up_conv(filters[4], filters[3])#2*
        self.Up_conv5 = TimestepEmbedSequential(conv_block(2*filters[3], filters[3], time_embed_dim))#decoder 3*
        self.Up_conv5_2 = TimestepEmbedSequential(conv_block(filters[3], filters[3], time_embed_dim))
        self.Up_conv5_3 = TimestepEmbedSequential(conv_block(filters[3], filters[3], time_embed_dim))

        self.Up4 = up_conv(filters[3], filters[2])#2*
        self.Up_conv4 = TimestepEmbedSequential(conv_block(2*filters[2], filters[2], time_embed_dim))#3*
        self.Up_conv4_2 = TimestepEmbedSequential(conv_block(filters[2], filters[2], time_embed_dim))
        self.Up_conv4_3 = TimestepEmbedSequential(conv_block(filters[2], filters[2], time_embed_dim))

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = TimestepEmbedSequential(conv_block(2*filters[1], filters[1], time_embed_dim))#3*
        self.Up_conv3_2 = TimestepEmbedSequential(conv_block(filters[1], filters[1], time_embed_dim))
        self.Up_conv3_3 = TimestepEmbedSequential(conv_block(filters[1], filters[1], time_embed_dim))

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = TimestepEmbedSequential(conv_block(2*filters[0], filters[0], time_embed_dim))#3*
        self.Up_conv2_2 = TimestepEmbedSequential(conv_block(filters[0], filters[0], time_embed_dim))
        self.Up_conv2_3 = TimestepEmbedSequential(conv_block(filters[0], filters[0], time_embed_dim))

        self.finalup = nn.Upsample(size=(256, 256))
        self.x5 = nn.Conv2d(filters[3], out_channels, kernel_size=1, stride=1, padding=0)#conv_block2(filters[3], filters[0])
        self.x4 = nn.Conv2d(filters[2], out_channels, kernel_size=1, stride=1, padding=0)#conv_block2(filters[2], filters[0])
        self.x3 = nn.Conv2d(filters[1], out_channels, kernel_size=1, stride=1, padding=0)#conv_block2(filters[1], filters[0])

        self.ou = nn.Conv2d(filters[0], out_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x, timesteps):
        emb = self.time_embed(timestep_embedding(timesteps, self.filter))
        c = x[:, :-1, ...]
        c1, c2, c3, c4 = self.ConUNet(c)#, c5
        x = self.conv0(x)
        e1 = self.Conv1(x, emb)
        e1 = self.Conv1_2(e1, emb)
        # r1_2 = e1
        e1 = self.Conv1_3(e1, emb)
        # e1 = torch.cat((c1, e1), dim=1)
        r1_3 = e1

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2, emb)
        e2 = self.Conv2_2(e2, emb)
        # r2_2 = e2
        e2 = self.Conv2_3(e2, emb)
        # e2 = torch.cat((c2, e2), dim=1)
        r2_3 = e2

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3, emb)
        e3 = self.Conv3_2(e3, emb)
        # r3_2 = e3
        e3 = self.Conv3_3(e3, emb)
        # e3 = torch.cat((c3, e3), dim=1)
        r3_3 = e3

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4, emb)
        e4 = self.Conv4_2(e4, emb)
        # r4_2 = e4
        e4 = self.Conv4_3(e4, emb)
        # e4 = torch.cat((c4, e4), dim=1)
        e4 = c4 + e4
        r4_3 = e4

        x1, x2, x3, x4 = self.DCA([r1_3, r2_3, r3_3, r4_3])

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5, emb)
        e5 = self.Conv5_2(e5, emb)
        e5 = self.Conv5_3(e5, emb)

        d5 = self.Up5(e5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5, emb)
        # d5 = torch.cat((r4_2, d5), dim=1)
        d5 = self.Up_conv5_2(d5, emb)
        d5 = self.Up_conv5_3(d5, emb)

        d4 = self.Up4(d5)#d5
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4, emb)
        # d4 = torch.cat((r3_2, d4), dim=1)
        d4 = self.Up_conv4_2(d4, emb)
        d4 = self.Up_conv4_3(d4, emb)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3, emb)
        # d3 = torch.cat((r2_2, d3), dim=1)
        d3 = self.Up_conv3_2(d3, emb)
        d3 = self.Up_conv3_3(d3, emb)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2, emb)
        # d2 = torch.cat((r1_2, d2), dim=1)
        d2 = self.Up_conv2_2(d2, emb)
        d2 = self.Up_conv2_3(d2, emb)

        x5 = self.x5(self.finalup(d5))
        x4 = self.x4(self.finalup(d4))
        x3 = self.x3(self.finalup(d3))
        out = self.ou(d2)

        return out, x5, x4, x3


if __name__ == "__main__":
    import torch
    model = UNet(4, 1)
    # model.eval()
    # total = sum([param.nelement() for param in model.parameters()])
    # print('Number of parameter: % .2fM' % (total / 1e6))
    input = torch.rand(8,4,256,256)
    input = input[:, :, 1:, :] - input[:, :, :-1, :]
    # t = torch.rand(8)
    # output, out2 = model(input, t)
    print("\n",input.shape)
    # print(model)
