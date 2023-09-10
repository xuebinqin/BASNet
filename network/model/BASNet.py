import torch
import torch.nn as nn
from torchvision import models

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        # First convolutional layer
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Second convolutional layer
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Downsample layer if exists
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        # First convolutional block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        # Second convolutional block
        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample:
            residual = self.downsample(residual)

        x += residual
        x = self.relu2(x)

        return x

class PredictModule(nn.Module):
    """
    Predict Module (Encoder-Decoder).
    A unet based model.
    """
    def __init__(self):
        super(PredictModule, self).__init__()

        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        self.encoder = self._make_encoder(resnet)
        self.bridge = self._make_bridge()
        self.decoder = self._make_decoder()
        self.side_outputs = self._make_side_outputs()
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def _make_encoder(self, resnet):
        encoder_layers = [
            nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),  
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                # stage 1
                resnet.layer1 #224
            ),
            # stage 2
            resnet.layer2, #112
            # stage 3
            resnet.layer3, #56
            # stage 4
            resnet.layer4, #28
            # stage 5
            nn.Sequential(
                nn.MaxPool2d(2, 2, ceil_mode=True), #14
                BasicBlock(512, 512),
                BasicBlock(512, 512),
                BasicBlock(512, 512)
            ),
            # stage 6
            nn.Sequential(
                nn.MaxPool2d(2,2,ceil_mode=True), #7
                BasicBlock(512, 512),
                BasicBlock(512, 512),
                BasicBlock(512, 512)
            )
        ]

        return nn.ModuleList(encoder_layers)

    def _make_bridge(self):
        bridge_layers = [
            self._conv_block(512, 512, 3, dilation=2, padding=2),
            self._conv_block(512, 512, 3, dilation=2, padding=2),
            self._conv_block(512, 512, 3, dilation=2, padding=2)
        ] #7
        return nn.Sequential(*bridge_layers)

    def _make_decoder(self):
        decoder_layers = [
            # stage 6d
            nn.Sequential(
                self._conv_block(1024, 512, 3, padding=1),  #7
                self._conv_block(512, 512, 3, dilation=2, padding=2),
                self._conv_block(512, 512, 3, dilation=2, padding=2)
            ),
            # stage 5d
            nn.Sequential(
                self._conv_block(1024, 512, 3, padding=1),  #14
                self._conv_block(512, 512, 3, padding=1),
                self._conv_block(512, 512, 3, padding=1)
            ),
            # stage 4d
            nn.Sequential(
                self._conv_block(1024, 512, 3, padding=1),  #28
                self._conv_block(512, 512, 3, padding=1),
                self._conv_block(512, 256, 3, padding=1)
            ),
            # stage 3d
            nn.Sequential(
                self._conv_block(512, 256, 3, padding=1),  #56
                self._conv_block(256, 256, 3, padding=1),
                self._conv_block(256, 128, 3, padding=1)
            ),
            # stage 2d
            nn.Sequential(
                self._conv_block(256, 128, 3, padding=1),  #112
                self._conv_block(128, 128, 3, padding=1),
                self._conv_block(128, 64, 3, padding=1)
            ),
            # stage 1d
            nn.Sequential(
                self._conv_block(128, 64, 3, padding=1),  #224
                self._conv_block(64, 64, 3, padding=1),
                self._conv_block(64, 64, 3, padding=1)
            )
        ]

        return nn.ModuleList(decoder_layers)

    def _make_side_outputs(self):
        
        side_output_layers = []
        channels = [512, 512, 512, 256, 128, 64]  # sup1 -> sup6
        upsample_scales = [32, 32, 16, 8, 4 ,2] # sup1 -> sup6

        for channel, scale_factor in zip(channels, upsample_scales):
            side_output_layers += [
                nn.Sequential(
                    nn.Conv2d(channel, 1, 3, padding=1),
                    nn.Upsample(scale_factor=scale_factor, mode='bilinear')
                )
            ]

        side_output_layers += [nn.Conv2d(64, 1, 3, padding=1)]  # sup7

        return nn.ModuleList(side_output_layers)

    def _conv_block(self, in_channels, out_channels, kernel_size, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        encoder_outs = []
        for encoder_module in self.encoder:
            x = encoder_module(x)
            encoder_outs.append(x)

        # Bridge
        x = self.bridge(x)

        # Decoder and Side Outputs
        side_outputs = [self.side_outputs[0](x)] if self.training else []

        for idx, decoder_module in enumerate(self.decoder, start=1):
            x = torch.cat((x, encoder_outs[-idx]), dim=1)
            x = decoder_module(x)
            if self.training:
                sx = self.side_outputs[idx](x)
                side_outputs.append(sx)
            else:
                if idx == 6:
                    sx = self.side_outputs[idx](x)
                    side_outputs.append(sx)
            
            if idx < 6: # not applying upsample for decoder stage 1d
                x = self.upsample2(x)
        
        return side_outputs

class RRM(nn.Module):
    """
    Residual Refinement Module (RRM).
    A unet based model.
    """
    def __init__(self, in_ch, inc_ch):
        super(RRM, self).__init__()

        # Initial convolution
        self.conv0 = nn.Conv2d(in_ch, inc_ch, 3, padding=1)

        # Encoder
        self.enc1 = self._block(inc_ch, 64)
        self.enc2 = self._block(64, 64)
        self.enc3 = self._block(64, 64)
        self.enc4 = self._block(64, 64)
        
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)
        
        # Bridge
        self.bridge = self._block(64, 64)
        
        # Decoder
        self.dec4 = self._block(128, 64)
        self.dec3 = self._block(128, 64)
        self.dec2 = self._block(128, 64)
        self.dec1 = self._block(128, 64)
        
        self.final = nn.Conv2d(64, 1, 3, padding=1)
        
        # Upsampling
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        hx1 = self.enc1(self.conv0(x))
        hx2 = self.enc2(self.pool(hx1))
        hx3 = self.enc3(self.pool(hx2))
        hx4 = self.enc4(self.pool(hx3))
        
        # Bridge
        hx5 = self.bridge(self.pool(hx4))
        
        # Decoder with skip connections
        d4 = self.dec4(torch.cat((self.upsample2(hx5), hx4), 1))
        d3 = self.dec3(torch.cat((self.upsample2(d4), hx3), 1))
        d2 = self.dec2(torch.cat((self.upsample2(d3), hx2), 1))
        d1 = self.dec1(torch.cat((self.upsample2(d2), hx1), 1))

        # Final layer
        residual = self.final(d1)

        return x + residual

class BASNet(nn.Module):
    def __init__(self):
        super(BASNet, self).__init__()

        self.predict_module = PredictModule()
        self.refine_module = RRM(1, 64)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        side_outputs = self.predict_module(x)
        out = self.refine_module(side_outputs[-1])
        out = self.sigmoid(out)

        if self.training:
            return (out, *[self.sigmoid(x) for x in side_outputs])
        return out
