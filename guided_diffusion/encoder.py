import torch.utils.data
import torch
import torch.nn as nn
import torch.nn.functional as F
# tensor=torch.ones(size=(2,1280,32,32))
from guided_diffusion.utils import dice_score
# print(tensor)
# from guided_diffusion.diff_unet import ODConv2d

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

class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


class ODConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=1):
        super(ODConv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = Attention(in_planes, out_planes, kernel_size, groups=groups,
                                   reduction=reduction, kernel_num=kernel_num)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes//groups, kernel_size, kernel_size),
                                   requires_grad=True)
        self._initialize_weights()

        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def _forward_impl_common(self, x):
        # Multiplying channel attention (or filter attention) to weights and feature maps are equivalent,
        # while we observe that when using the latter method the models will run faster with less gpu memory cost.
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x * channel_attention
        x = x.reshape(1, -1, height, width)
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = output * filter_attention
        return output

    def _forward_impl_pw1x(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x):
        return self._forward_impl(x)

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            # nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False, groups=in_ch),#去掉bias=True,加上group,输出变为in_ch
            # nn.BatchNorm2d(in_ch),
            # nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            #一个深度可分离卷积；一个普通卷积
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.PEE = PEE(out_ch, out_ch, [3, 5, 7])
        self.fuse = nn.Sequential(nn.Conv2d(out_ch * 4, out_ch, kernel_size=1, stride=1, padding=0),#True
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False, groups=out_ch),
                                       # nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
                                       nn.BatchNorm2d(out_ch),
                                       nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0))#True
        # self.fuse = nn.Conv2d(out_ch*4, out_ch, kernel_size=1, stride=1, padding=0, bias=True)
        # if in_ch == out_ch:
        #     self.skip_connection = nn.Identity()
        # else:
        #     self.skip_connection = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True)
        #     # self.skip_connection = ODConv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        #     # self.skip_connection = SeparableConv2d(in_ch, out_ch, kernel_size=3, stride=1, bias=True)
    def forward(self, x):
        h = self.conv(x)
        # h = h + self.skip_connection(x)
        edge = self.PEE(h)
        # h = torch.cat((h, edge), dim=1)
        h = self.fuse(edge)
        # h = edge + self.skip_connection(x)
        # h = h + edge
        return h

class conv_block1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block1, self).__init__()

        self.conv = nn.Sequential(
            # nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False, groups=in_ch),#, bias=True,group,输出in
            # nn.BatchNorm2d(in_ch),
            # nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            #一个深度可分离卷积；一个普通卷积
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h = self.conv(x)
        return h

class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),#改为false
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class SE_Block(nn.Module):                         # Squeeze-and-Excitation block
    def __init__(self, in_planes):
        super(SE_Block, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(in_planes, in_planes // 16, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes // 16, in_planes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = x
        x = self.avgpool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        out = self.sigmoid(x)
        # print("x:",x.shape)
        # print("out:",out.shape)
        return y * out.expand_as(y)

class PEE(nn.Module):
    def __init__(self, in_channels, out_channels, edge_poolings):
        super(PEE, self).__init__()

        # self.reduce_conv = nn.Conv2d(in_channels, out_channels, 1, 1, bias=False)

        self.edge_pooling1 = nn.AvgPool2d(edge_poolings[0], stride=1, padding=(edge_poolings[0] - 1) // 2)
        self.edge_pooling2 = nn.AvgPool2d(edge_poolings[1], stride=1, padding=(edge_poolings[1] - 1) // 2)
        self.edge_pooling3 = nn.AvgPool2d(edge_poolings[2], stride=1, padding=(edge_poolings[2] - 1) // 2)

        self.SE = SE_Block(out_channels * 4)
        # self.fuse_conv = nn.Sequential(nn.Conv2d(out_channels * 3, out_channels, 1, 1, bias=False))
        #                                nn.ReLU(inplace=True),
        #                                nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False, groups=out_channels),
        #                                nn.BatchNorm2d(out_channels),
        #                                nn.Conv2d(out_channels, out_channels, 1, 1, bias=False))
        # self.Sig = nn.Sigmoid()

    def forward(self, x):
        # x = self.reduce_conv(x)
        edge1 = x - self.edge_pooling1(x)
        edge2 = x - self.edge_pooling2(x)
        edge3 = x - self.edge_pooling3(x)
        cat = torch.cat((x, edge1, edge2, edge3), dim=1)
        edge = self.SE(cat)
        # cat = torch.cat((edge1, edge2, edge3), dim=1)
        # edge = self.fuse_conv(edge)
        # x = x + edge
        return edge

class ConUNet(nn.Module):

    def __init__(self, n_channels, out_channels):
        super(ConUNet, self).__init__()
        self.n_channels = n_channels
        self.out_channels = out_channels
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]#, n1 * 16

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(n_channels, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        # self.Conv2_2 = conv_block(filters[1], filters[1])
        # self.PEE2 = PEE(filters[1], filters[1], [5, 7])
        self.Conv3 = conv_block(filters[1], filters[2])
        # self.Conv3_2 = conv_block(filters[2], filters[2])
        # self.PEE3 = PEE(filters[2], filters[2], [3, 5])
        self.Conv4 = conv_block(filters[2], filters[3])
        # self.Conv4_2 = conv_block(filters[3], filters[3])
        # self.PEE4 = PEE(filters[3], filters[3], [3, 5])
        # self.Conv5 = conv_block(filters[3], filters[4])#去掉第五层
        # self.Conv5_2 = conv_block(filters[4], filters[4])
        # self.PEE5 = PEE(filters[4], filters[4], [3, 5])


    def forward(self, x):
        e1 = self.Conv1(x)
        # e1_1 = self.Conv1_2(e1)
        # e1 = self.PEE1(e1)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        # e2_1 = self.Conv2_2(e2)
        # e2 = self.PEE2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        # e3_1 = self.Conv3_2(e3)
        # e3 = self.PEE3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        # e4_1 = self.Conv4_2(e4)
        # e4 = self.PEE4(e4)

        # e5 = self.Maxpool4(e4)#去掉第五层
        # e5 = self.Conv5(e5)#去掉第五层
        # e5_1 = self.Conv5_2(e5)
        # e5 = self.PEE5(e5)

        return e1, e2, e3, e4#, e5


if __name__ == "__main__":
    import torch
    model = ConUNet(3, 1)
    total = sum([param.nelement() for param in model.parameters()])
    print('Number of parameter: % .2fM' % (total / 1e6))
    # model.eval()
    # input = torch.sigmoid(torch.rand(8,1,256,256))
    # output = torch.sigmoid(torch.rand(8,1,256,256))
    # output = torch.where(output > 0.5, 1, 0)
    #
    # input = torch.where(input > 0.5, 1, 0)
    # print("dice1:", 2. * (input * output).sum() / (input + output).sum())
    # sum = 0
    # for output, input in zip(output, input):
    #     dice = dice_score(input,output)
    #     sum = sum+dice
    # print("dice2:",sum/8)
    # output = model(input)
    # print("\n",output.shape)
    # print(model)
