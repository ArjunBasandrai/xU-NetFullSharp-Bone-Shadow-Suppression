import torch 
import torch.nn as nn

class Gaussian(nn.Module):
    def __init__(self):
        super(Gaussian, self).__init__()

    def forward(self, x):
        return torch.exp(-x**2)

class XUnit(nn.Module):
    def __init__(self, in_channels, kernel_size=9, g=True):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2,
            groups=in_channels, 
            bias=True
        )
        self.G = Gaussian() if g else nn.Sigmoid()

    def forward(self, x):
        out = nn.functional.relu(x)
        out = self.depthwise(out)
        out = self.G(out)
        out = out * x
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=1, padding=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            dilation=dilation, 
            padding=dilation,
            bias=True
        )
        self.x_unit = XUnit(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.x_unit(x)
        return x
        
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.conv_block = ConvBlock(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=1, 
            dilation=dilation
        )

    def forward(self, x, skip):
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, level=1):
        super().__init__()
        
        dilation_c1 = 2 if (level == 3) else 1
        dilation_c2c3 = 2**(level-1)
        dilation_bn = 2**(level+1)
        dilation_d1 = 2 if (level == 3) else 1
        
        self.c1 = ConvBlock(in_channels, out_channels, dilation=dilation_c1)
        self.c2 = ConvBlock(out_channels, out_channels, dilation=dilation_c2c3)
        self.c3 = ConvBlock(out_channels, out_channels, dilation=dilation_c2c3)
        
        self.bn = ConvBlock(out_channels, out_channels, dilation=dilation_bn)
        
        self.d3 = UpBlock(2 * out_channels, out_channels, dilation=dilation_c2c3)
        self.d2 = UpBlock(2 * out_channels, out_channels, dilation=dilation_c2c3)
        self.d1 = UpBlock(2 * out_channels, out_channels, dilation=dilation_d1)

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        
        bn = self.bn(c3)
        
        d3 = self.d3(bn, c3)
        d2 = self.d2(d3, c2)
        d1 = self.d1(d2, c1)
        return d1

class xU_NetFullSharp(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.f = [8, 16, 32, 64, 128]
        
        self.encoder_1_1 = EncoderBlock(in_channels, self.f[0], level=1)
        self.encoder_2_1 = EncoderBlock(self.f[0], self.f[1], level=1)
        self.encoder_3_1 = EncoderBlock(self.f[1], self.f[2], level=2)
        self.encoder_4_1 = EncoderBlock(self.f[2], self.f[3], level=2)
        self.encoder_5_1 = EncoderBlock(self.f[3], self.f[4], level=3)
        
        self.maxpool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool_4x4 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.maxpool_8x8 = nn.MaxPool2d(kernel_size=8, stride=8)

        self.conv1_2 = ConvBlock(self.f[1] + self.f[0], self.f[0])
        self.conv2_2 = ConvBlock(self.f[2] + self.f[1] + self.f[0], self.f[1])
        self.conv1_3 = ConvBlock(self.f[1] + self.f[0] * 2 + self.f[2], self.f[0])
        self.conv3_2_1 = ConvBlock(self.f[0] + self.f[1], self.f[2] * 2)
        self.conv3_2 = ConvBlock(self.f[3] + self.f[2] + self.f[2] * 2, self.f[2])
        self.conv2_3 = ConvBlock(self.f[2] + self.f[1] * 2 + self.f[3] + self.f[0], self.f[1])
        self.conv1_4_1 = ConvBlock(self.f[3] + self.f[2], self.f[0] * 2)
        self.conv1_4 = ConvBlock(self.f[1] + self.f[0] * 5, self.f[0])
        self.conv4_2_1 = ConvBlock(self.f[2] + self.f[1] + self.f[0], self.f[3] * 3)
        self.conv4_2 = ConvBlock(self.f[4] + self.f[3] * 4, self.f[3])
        self.conv3_3_1 = ConvBlock(self.f[1] + self.f[0], self.f[2] * 2)
        self.conv3_3 = ConvBlock(self.f[3] + self.f[2] * 2 + self.f[4] + self.f[2] * 2, self.f[2])
        self.conv2_4_1 = ConvBlock(self.f[3] + self.f[4], self.f[1] * 2)
        self.conv2_4 = ConvBlock(self.f[2] + self.f[1] * 5 + self.f[0], self.f[1])
        self.conv1_5_1 = ConvBlock(self.f[2] + self.f[3] + self.f[4], self.f[0] * 3)
        self.conv1_5 = ConvBlock(self.f[1] + self.f[0] * 7, self.f[0])
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.up16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)
        
        self.out_conv = nn.Conv2d(self.f[0], out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        conv1_1 = self.encoder_1_1(x)
        pool1_1_1 = self.maxpool_2x2(conv1_1)
        pool1_1_2 = self.maxpool_4x4(conv1_1)
        pool1_1_3 = self.maxpool_8x8(conv1_1)
        
        conv2_1 = self.encoder_2_1(pool1_1_1)
        pool2_1_1 = self.maxpool_2x2(conv2_1)
        pool2_1_2 = self.maxpool_4x4(conv2_1)
        
        up1_2 = self.up2(conv2_1)
        conv1_2_in = torch.cat([up1_2, conv1_1], dim=1)
        conv1_2 = self.conv1_2(conv1_2_in)
        pool1_2_1 = self.maxpool_2x2(conv1_2)
        pool1_2_2 = self.maxpool_4x4(conv1_2)
        
        conv3_1 = self.encoder_3_1(pool2_1_1)
        pool3 = self.maxpool_2x2(conv3_1)
        conv3_1_1_3 = self.up4(conv3_1)
        
        up2_2 = self.up2(conv3_1)
        conv2_2_in = torch.cat([up2_2, conv2_1, pool1_1_1], dim=1)
        conv2_2 = self.conv2_2(conv2_2_in)
        pool2_2_1 = self.maxpool_2x2(conv2_2)
        
        up1_3 = self.up2(conv2_2)
        conv1_3_in = torch.cat([up1_3, conv1_1, conv1_2, conv3_1_1_3], dim=1)
        conv1_3 = self.conv1_3(conv1_3_in)
        pool1_3_1 = self.maxpool_2x2(conv1_3)
            
        conv4_1 = self.encoder_4_1(pool3)
        conv4_1_2_3 = self.up4(conv4_1)
        conv4_1_1_4 = self.up8(conv4_1)
        pool4 = self.maxpool_2x2(conv4_1)
        
        up3_2 = self.up2(conv4_1)
        conv3_2_1_in = torch.cat([pool1_1_2, pool2_1_1], dim=1)
        conv3_2_1 = self.conv3_2_1(conv3_2_1_in)
        conv3_2_in = torch.cat([up3_2, conv3_1, conv3_2_1], dim=1)
        conv3_2 = self.conv3_2(conv3_2_in)

        conv3_2_1_4 = self.up4(conv3_2)

        up2_3 = self.up2(conv3_2)
        conv2_3_in = torch.cat([up2_3, conv2_1, conv2_2, conv4_1_2_3, pool1_2_1], dim=1)
        conv2_3 = self.conv2_3(conv2_3_in)
        
        up1_4 = self.up2(conv2_3)
        conv1_4_1_in = torch.cat([conv4_1_1_4, conv3_2_1_4], dim=1)
        conv1_4_1 = self.conv1_4_1(conv1_4_1_in)
        conv1_4_in = torch.cat([up1_4, conv1_1, conv1_2, conv1_3, conv1_4_1], dim=1)
        conv1_4 = self.conv1_4(conv1_4_in)
        
        conv5_1 = self.encoder_5_1(pool4)
        conv5_1_3_3 = self.up4(conv5_1)
        conv5_1_2_4 = self.up8(conv5_1)
        conv5_1_1_5 = self.up16(conv5_1)

        up4_2 = self.up2(conv5_1)
        conv4_2_1_in = torch.cat([pool3, pool2_1_2, pool1_1_3], dim=1)
        conv4_2_1 = self.conv4_2_1(conv4_2_1_in)
        conv4_2_in = torch.cat([up4_2, conv4_1, conv4_2_1], dim=1)
        conv4_2 = self.conv4_2(conv4_2_in)
        
        conv4_2_2_4 = self.up4(conv4_2)
        conv4_2_1_5 = self.up8(conv4_2)

        up3_3 = self.up2(conv4_2)
        conv3_3_1_in = torch.cat([pool2_2_1, pool1_2_2], dim=1)
        conv3_3_1 = self.conv3_3_1(conv3_3_1_in)
        conv3_3_in = torch.cat([up3_3, conv3_1, conv3_2, conv5_1_3_3, conv3_3_1], dim=1)
        conv3_3 = self.conv3_3(conv3_3_in)
        
        conv3_3_1_5 = self.up4(conv3_3)

        up2_4 = self.up2(conv3_3)
        conv2_4_1_in = torch.cat([conv4_2_2_4, conv5_1_2_4], dim=1)
        conv2_4_1 = self.conv2_4_1(conv2_4_1_in)
        conv2_4_in = torch.cat([up2_4, conv2_1, conv2_2, conv2_3, conv2_4_1, pool1_3_1], dim=1)
        conv2_4 = self.conv2_4(conv2_4_in)
        
        up1_5 = self.up2(conv2_4)
        conv1_5_1_in = torch.cat([conv3_3_1_5, conv4_2_1_5, conv5_1_1_5], dim=1)
        conv1_5_1 = self.conv1_5_1(conv1_5_1_in)
        conv1_5_in = torch.cat([up1_5, conv1_1, conv1_2, conv1_3, conv1_4, conv1_5_1], dim=1)
        conv1_5 = self.conv1_5(conv1_5_in)
        
        out = torch.sigmoid(self.out_conv(conv1_5))
        return out
        