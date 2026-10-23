import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math, numbers
import argparse
import numpy as np
from torch import nn
from torch.nn import init
from os.path import join
from lib.pvtv2 import pvt_v2_b4
from einops import rearrange
from pytorch_wavelets import DWTForward, DWTInverse
import fvcore.nn.weight_init as weight_init
from timm.models.layers import DropPath


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x




def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

    def initialize(self):
        weight_init(self)


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

    def initialize(self):
        weight_init(self)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)
        self.act1 = nn.GELU()
        self.conv3 = BasicConv2d(hidden_features * 2, hidden_features * 2, 3, 1, 1)

    def forward(self, x):
        x1 = self.project_in(x)
        x11 = self.act1(x1)
        x11 = self.conv3(x11)
        x3 = self.project_out(x11)
        return x3

    def initialize(self):
        weight_init(self)

def custom_complex_normalization(input_tensor, dim=-1):
    real_part = input_tensor.real
    imag_part = input_tensor.imag
    norm_real = F.softmax(real_part, dim=dim)
    norm_imag = F.softmax(imag_part, dim=dim)

    normalized_tensor = torch.complex(norm_real, norm_imag)

    return normalized_tensor



class FTSA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(FTSA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)
        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

        self.qkv1conv_3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv2conv_3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv3conv_3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.qkv1conv_5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim, bias=bias)
        self.qkv2conv_5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim, bias=bias)
        self.qkv3conv_5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim, bias=bias)

        self.qkv1conv_7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim, bias=bias)
        self.qkv2conv_7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim, bias=bias)
        self.qkv3conv_7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim, bias=bias)

        self.qkv_0 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.ca = ChannelAttention1(dim)

    def forward(self, x):
        xc = torch.abs(torch.fft.ifft2(self.ca(torch.fft.fft2(x.float()).real) * torch.fft.fft2(x.float())))

        x_out = F.interpolate(xc, size=x.size()[2:], mode='bilinear')
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        q = (self.qkv1conv_3(self.qkv_0(q)) + self.qkv1conv_5(self.qkv_0(q)) + self.qkv1conv_7(self.qkv_0(q))) * x_out
        k = (self.qkv2conv_3(self.qkv_1(k)) + self.qkv1conv_5(self.qkv_1(k)) + self.qkv1conv_7(self.qkv_1(k))) * x_out
        v = self.qkv3conv_3(self.qkv_2(v)) + self.qkv1conv_5(self.qkv_2(v)) + self.qkv1conv_7(self.qkv_2(v))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        _, _, C, _ = q.shape
        mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        index = torch.topk(attn, k=int(C / 2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))
        index = torch.topk(attn, k=int(C * 2 / 3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))
        index = torch.topk(attn, k=int(C * 3 / 4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))
        index = torch.topk(attn, k=int(C * 4 / 5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))
        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)
        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)
        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out





class MSA_head(nn.Module):  # Multi-scale transformer block
    def __init__(self, mode='dilation', dim=128, num_heads=8, ffn_expansion_factor=4, bias=False,
                 LayerNorm_type='WithBias'):
        super(MSA_head, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.ftsa = FTSA(dim, num_heads, bias)

    def forward(self, x):
        x = x + self.ftsa(self.norm1(x))

        x1 = x + self.ffn(self.norm2(x))
        # print(x.shape)
        return x1



class FSTB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FSTB, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1), nn.BatchNorm2d(out_channel), nn.ReLU(True)
        )
        self.reduce = nn.Sequential(
            nn.Conv2d(out_channel * 2, out_channel, 1), nn.BatchNorm2d(out_channel), nn.ReLU(True)
        )
        self.relu = nn.ReLU(True)

        self.t1 = MSA_head(out_channel)

    def forward(self, x):
        x0 = self.conv1(x)
        x0 = self.t1(x0)
        return x0




class ChannelAttention1(nn.Module):
    def __init__(self, channels):
        super(ChannelAttention1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sharedMLP = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        avgout = (self.avg_pool(x))
        maxout = (self.max_pool(x))
        return self.sharedMLP(avgout + maxout)





class CurveCALayer(nn.Module):
    def __init__(self, channel):
        super(CurveCALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.n_curve = 3
        self.relu = nn.ReLU(inplace=False)
        self.predict_a = nn.Sequential(
            nn.Conv2d(channel, channel, 5, stride=1, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channel, 3, 1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.avg_pool(x)
        a = self.predict_a(x1)

        x = torch.clamp(x, 0.0, 1.0)
        for i in range(self.n_curve):
            x = x + a[:, i:i + 1] * x * (1 - x)
            x = torch.clamp(x, 0.0, 1.0)  # 确保迭代稳定
        return self.sigmoid(x)


class SCCM(nn.Module):  #
    def __init__(self, channels, in_channels):
        super(SCCM, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, in_channels, 1), nn.BatchNorm2d(in_channels), nn.ReLU(True)
        )
        self.conv11 = nn.Conv2d(channels,1,1)

        self.Dconv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels),

            nn.Conv2d(in_channels, in_channels, 3, padding=3, dilation=3), nn.BatchNorm2d(in_channels), nn.ReLU(True)
        )

        self.Dconv5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels),

            nn.Conv2d(in_channels, in_channels, 3, padding=5, dilation=5), nn.BatchNorm2d(in_channels), nn.ReLU(True)
        )
        self.Dconv7 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels),

            nn.Conv2d(in_channels, in_channels, 3, padding=7, dilation=7), nn.BatchNorm2d(in_channels), nn.ReLU(True)
        )
        self.Dconv9 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels),

            nn.Conv2d(in_channels, in_channels, 3, padding=9, dilation=9), nn.BatchNorm2d(in_channels), nn.ReLU(True)
        )

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels * 5, in_channels, 1), nn.BatchNorm2d(in_channels), nn.ReLU(True)
        )

        self.out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1), nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1)
        )

        self.weight = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, 1, bias=True),
            nn.BatchNorm2d(in_channels // 16),
            nn.ReLU(True),
            nn.Conv2d(in_channels // 16, in_channels, 1, bias=True),
            nn.Sigmoid())

        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(True)
        self.curatt = CurveCALayer(in_channels)
        self.ca = ChannelAttention1(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, F1):
        F1_input = self.conv1(F1)

        F1_3_s = self.Dconv3(F1_input)
        F1_3_f = self.curatt(self.ca(F1_3_s)) * F1_3_s
        F1_3 = torch.add(F1_3_s, F1_3_f)

        F1_5_s = self.Dconv5(F1_input + F1_3)
        F1_5_f = self.curatt(self.ca(F1_5_s)) * F1_5_s
        # self.relu(self.norm(torch.abs(torch.fft.ifft2(self.weight(torch.fft.fft2(F1_5_s.float()).real)*torch.fft.fft2(F1_5_s.float())))))
        F1_5 = torch.add(F1_5_s, F1_5_f)

        F1_7_s = self.Dconv7(F1_input + F1_5)
        F1_7_f = self.curatt(self.ca(F1_7_s)) * F1_7_s
        # self.relu(self.norm(torch.abs(torch.fft.ifft2(self.weight(torch.fft.fft2(F1_7_s.float()).real)*torch.fft.fft2(F1_7_s.float())))))
        F1_7 = torch.add(F1_7_s, F1_7_f)

        F1_9_s = self.Dconv9(F1_input + F1_7)
        F1_9_f = self.curatt(self.ca(F1_9_s)) * F1_9_s
        # self.relu(self.norm(torch.abs(torch.fft.ifft2(self.weight(torch.fft.fft2(F1_9_s.float()).real)*torch.fft.fft2(F1_9_s.float())))))
        F1_9 = torch.add(F1_9_s, F1_9_f)

        F_out = self.out(self.reduce(torch.cat((F1_3, F1_5, F1_7, F1_9, F1_input), 1)) + F1_input)

        return F_out



class Moude2(nn.Module):
    def __init__(self, in_channel):
        super(Moude2, self).__init__()
        self.conv1 = BasicConv2d(in_channel, in_channel, 1)
        self.conv3 = BasicConv2d(in_channel * 3, in_channel, 1)
        self.conv4 = BasicConv2d(in_channel, in_channel, 3, 1, 1)
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]  # 垂直
        y_LH = yH[0][:, :, 1, ::]  # 水平
        y_HH = yH[0][:, :, 2, ::]  # 对角
        xc = self.conv3(torch.cat((y_HL, y_LH, y_HH), 1))
        xc = self.global_att(xc)
        x1 = self.conv1(yL)
        out = self.conv1(x1 + xc)
        out = F.interpolate(out, size=x.size()[2:], mode='bilinear', align_corners=True)
        return out + self.conv4(x)







class TFRD_1(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(TFRD_1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels),
            nn.ReLU(True)
        )

        self.out = nn.Sequential(
            nn.Conv2d(in_channels * 2, mid_channels, kernel_size=3, padding=1), nn.BatchNorm2d(mid_channels),
            nn.ReLU(True),
            nn.Conv2d(mid_channels, 1, kernel_size=1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
        )


        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(in_channels)
        self.m2 = Moude2(in_channels)
        self.conv1 = BasicConv2d(1, in_channels, 1)

    def forward(self, X, prior_cam):
        prior_cam = F.interpolate(prior_cam, size=X.size()[2:], mode='bilinear', align_corners=True)

        FI = X

        yt = self.conv(torch.cat([FI, prior_cam.expand(-1, X.size()[1], -1, -1)], dim=1))

        yt_s = self.conv3(yt)
        yt_f = self.m2(yt)
        yt_out = torch.add(yt_s, yt_f)

        r_prior_cam_f = self.conv1(prior_cam)
        # print(r_prior_cam_f.shape)
        r_prior_cam_f = -1 * (torch.sigmoid(r_prior_cam_f)) + 1

        r_prior_cam_s = -1 * (torch.sigmoid(prior_cam)) + 1
        r_prior_cam = r_prior_cam_s + r_prior_cam_f

        y_ra = r_prior_cam.expand(-1, X.size()[1], -1, -1).mul(FI)

        out = torch.cat([y_ra, yt_out], dim=1)  # 2,128,48,48

        y = self.out(out)
        y = y + prior_cam
        return y


class TFRD_2(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(TFRD_2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
        )

        self.out = nn.Sequential(
            nn.Conv2d(in_channels * 2, mid_channels, kernel_size=3, padding=1), nn.BatchNorm2d(mid_channels),
            nn.ReLU(True),
            nn.Conv2d(mid_channels, 1, kernel_size=1)
        )

        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(True)
        self.m2 = Moude2(in_channels)

        self.conv1 = BasicConv2d(1, in_channels, 1)

    def forward(self, X, x1, prior_cam):
        prior_cam = F.interpolate(prior_cam, size=X.size()[2:], mode='bilinear', align_corners=True)
        x1_prior_cam = F.interpolate(x1, size=X.size()[2:], mode='bilinear', align_corners=True)
        FI = X

        yt = self.conv(
            torch.cat([FI, prior_cam.expand(-1, X.size()[1], -1, -1), x1_prior_cam.expand(-1, X.size()[1], -1, -1)],
                      dim=1))

        yt_s = self.conv3(yt)
        yt_f = self.m2(yt)
        yt_out = torch.add(yt_s, yt_f)

        r_prior_cam_f = self.conv1(prior_cam)
        r_prior_cam_f = -1 * (torch.sigmoid(r_prior_cam_f)) + 1
        r_prior_cam_s = -1 * (torch.sigmoid(prior_cam)) + 1
        r_prior_cam = r_prior_cam_s + r_prior_cam_f

        r1_prior_cam_f = self.conv1(x1_prior_cam)
        r1_prior_cam_f = -1 * (torch.sigmoid(r1_prior_cam_f)) + 1
        r1_prior_cam_s = -1 * (torch.sigmoid(x1_prior_cam)) + 1
        r1_prior_cam = r1_prior_cam_s + r1_prior_cam_f

        r_prior_cam = r_prior_cam + r1_prior_cam

        y_ra = r_prior_cam.expand(-1, X.size()[1], -1, -1).mul(FI)

        out = torch.cat([y_ra, yt_out], dim=1)

        y = self.out(out)
        y = y + prior_cam + x1_prior_cam
        return y


class TFRD_3(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(TFRD_3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
        )

        self.out = nn.Sequential(
            nn.Conv2d(in_channels * 2, mid_channels, kernel_size=3, padding=1), nn.BatchNorm2d(mid_channels),
            nn.ReLU(True),
            nn.Conv2d(mid_channels, 1, kernel_size=1)
        )

        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(True)
        self.conv1 = BasicConv2d(1, in_channels, 1)
        self.m2 = Moude2(in_channels)

    def forward(self, X, x1, x2, prior_cam):
        prior_cam = F.interpolate(prior_cam, size=X.size()[2:], mode='bilinear', align_corners=True)  #
        x1_prior_cam = F.interpolate(x1, size=X.size()[2:], mode='bilinear', align_corners=True)
        x2_prior_cam = F.interpolate(x2, size=X.size()[2:], mode='bilinear', align_corners=True)
        FI = X

        yt = self.conv(torch.cat(
            [FI, prior_cam.expand(-1, X.size()[1], -1, -1), x1_prior_cam.expand(-1, X.size()[1], -1, -1),
             x2_prior_cam.expand(-1, X.size()[1], -1, -1)], dim=1))

        yt_s = self.conv3(yt)
        yt_f = self.m2(yt)
        yt_out = torch.add(yt_s, yt_f)

        r_prior_cam_f = self.conv1(prior_cam)
        r_prior_cam_f = -1 * (torch.sigmoid(r_prior_cam_f)) + 1
        r_prior_cam_s = -1 * (torch.sigmoid(prior_cam)) + 1
        r_prior_cam = r_prior_cam_s + r_prior_cam_f

        r1_prior_cam_f = self.conv1(x1_prior_cam)
        r1_prior_cam_f = -1 * (torch.sigmoid(r1_prior_cam_f)) + 1
        r1_prior_cam_s = -1 * (torch.sigmoid(x1_prior_cam)) + 1
        r1_prior_cam1 = r1_prior_cam_s + r1_prior_cam_f

        r2_prior_cam_f = self.conv1(x2_prior_cam)
        r2_prior_cam_f = -1 * (torch.sigmoid(r2_prior_cam_f)) + 1
        r2_prior_cam_s = -1 * (torch.sigmoid(x2_prior_cam)) + 1
        r1_prior_cam2 = r2_prior_cam_s + r2_prior_cam_f

        r_prior_cam = r_prior_cam + r1_prior_cam1 + r1_prior_cam2

        y_ra = r_prior_cam.expand(-1, X.size()[1], -1, -1).mul(FI)

        out = torch.cat([y_ra, yt_out], dim=1)

        y = self.out(out)

        y = y + prior_cam + x1_prior_cam + x2_prior_cam

        return y



class Network(nn.Module):
    def __init__(self, channels=128, imagenet_pretrained=True):
        super(Network, self).__init__()
        self.backbone = pvt_v2_b4()  # [64, 128, 320, 512]
        path = 'D:\课程资料\论文\FSEL-main\FSEL\lib\pvt_v2_b4.pth'  # './lib/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        self.dePixelShuffle = torch.nn.PixelShuffle(2)
        self.up = nn.Sequential(
            nn.Conv2d(channels // 4, channels, kernel_size=1), nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(True)
        )

        self.FSTB_5 = FSTB(512 + channels, channels)
        self.FSTB_4 = FSTB(320 + channels, channels)
        self.FSTB_3 = FSTB(128 + channels, channels)
        self.FSTB_2 = FSTB(64 + channels, channels)

        self.SCCM = SCCM(512, channels)

        self.TFRD_1 = TFRD_1(channels, channels)
        self.TFRD_2 = TFRD_2(channels, channels)
        self.TFRD_3 = TFRD_3(channels, channels)

    def forward(self, x):
        image = x
        pvt = self.backbone(x)
        x1 = (pvt[0])  # Bx3x96×96
        x2 = (pvt[1])  # Bx3x48×48
        x3 = (pvt[2])  # Bx3x24×24
        x4 = (pvt[3])  # Bx3x12×12

        p1 = self.SCCM(x4)
        x5_4 = p1
        x5_4_1 = x5_4.expand(-1, 128, -1, -1)

        x4 = self.FSTB_5(torch.cat((x4, x5_4_1), 1))
        x4_up = self.up(self.dePixelShuffle(x4))

        x3 = self.FSTB_4(torch.cat((x3, x4_up), 1))
        x3_up = self.up(self.dePixelShuffle(x3))

        x2 = self.FSTB_3(torch.cat((x2, x3_up), 1))
        x2_up = self.up(self.dePixelShuffle(x2))

        x1 = self.FSTB_2(torch.cat((x1, x2_up), 1))

        x4 = self.TFRD_1(x4, x5_4)
        x3 = self.TFRD_1(x3, x4)
        x2 = self.TFRD_2(x2, x3, x4)
        x1 = self.TFRD_3(x1, x2, x3, x4)

        p0 = F.interpolate(p1, size=image.size()[2:], mode='bilinear', align_corners=True)
        f4 = F.interpolate(x4, size=image.size()[2:], mode='bilinear', align_corners=True)
        f3 = F.interpolate(x3, size=image.size()[2:], mode='bilinear', align_corners=True)
        f2 = F.interpolate(x2, size=image.size()[2:], mode='bilinear', align_corners=True)
        f1 = F.interpolate(x1, size=image.size()[2:], mode='bilinear', align_corners=True)

        return p0, f4, f3, f2, f1




