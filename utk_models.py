import torch
from torch import nn


def up_conv(in_channels, out_channels, kernel_size, stride=1, padding=1,
            scale_factor=2, norm='batch', activ=None):
    """Create a transposed-convolutional layer, with optional normalization."""
    layers = []
    layers.append(nn.Upsample(scale_factor=scale_factor, mode='nearest'))
    layers.append(nn.Conv2d(
        in_channels, out_channels,
        kernel_size, stride, padding, bias=norm is None
    ))
    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_channels))

    if activ == 'relu':
        layers.append(nn.ReLU())
    elif activ == 'leaky':
        layers.append(nn.LeakyReLU())
    elif activ == 'tanh':
        layers.append(nn.Tanh())

    return nn.Sequential(*layers)


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1,
         norm='batch', init_zero_weights=False, activ=None, discrim=True):
    """Create a convolutional layer, with optional normalization."""
    layers = []
    conv_layer = nn.Conv2d(
        in_channels=in_channels, out_channels=out_channels,
        kernel_size=kernel_size, stride=stride, padding=padding,
        bias=norm is None
    )
    if init_zero_weights:
        conv_layer.weight.data = 0.001 * torch.randn(
            out_channels, in_channels, kernel_size, kernel_size
        )
    if discrim:
        conv_layer = torch.nn.utils.spectral_norm(conv_layer)
    layers.append(conv_layer)

    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_channels))

    if activ == 'relu':
        layers.append(nn.ReLU())
    elif activ == 'leaky':
        layers.append(nn.LeakyReLU())
    elif activ == 'tanh':
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class Generator(nn.Module):

    def __init__(self, noise_size, conv_dim=128):
        super().__init__()
        self.up_conv1 = conv(in_channels=noise_size, out_channels=(conv_dim * 4), kernel_size=4, stride=1, padding=3,
                             norm='batch', init_zero_weights=False, activ='relu', discrim=False)
        self.up_conv2 = up_conv(in_channels=(conv_dim * 4), out_channels=(conv_dim * 2), kernel_size=3, stride=1,
                                padding=1, scale_factor=2, norm='batch', activ='relu')
        self.up_conv3 = up_conv(in_channels=(conv_dim * 2), out_channels=conv_dim, kernel_size=3, stride=1, padding=1,
                                scale_factor=2, norm='batch', activ='relu')
        self.up_conv4 = up_conv(in_channels=conv_dim, out_channels=(conv_dim // 2), kernel_size=3, stride=1, padding=1,
                                scale_factor=2, norm='batch', activ='relu')
        self.up_conv5 = up_conv(in_channels=(conv_dim // 2), out_channels=(conv_dim // 4), kernel_size=3, stride=1,
                                padding=1, scale_factor=2, norm='batch', activ='relu')
        self.up_conv6 = up_conv(in_channels=(conv_dim // 4), out_channels=3, kernel_size=3, stride=1,
                                padding=1, scale_factor=2, norm=None, activ='tanh')

    def forward(self, z):
        z = self.up_conv1(z)
        z = self.up_conv2(z)
        z = self.up_conv3(z)
        z = self.up_conv4(z)
        z = self.up_conv5(z)
        z = self.up_conv6(z)
        return z


class Discriminator(nn.Module):

    def __init__(self, conv_dim=64, norm="batch", sigmoid=False):
        super().__init__()
        self.conv1 = conv(in_channels=3, out_channels=(conv_dim // 2), kernel_size=4, stride=2, padding=1, norm=norm,
                          init_zero_weights=False, activ='leaky')
        self.conv2 = conv(in_channels=(conv_dim // 2), out_channels=conv_dim, kernel_size=4, stride=2, padding=1,
                          norm=norm, init_zero_weights=False, activ='leaky')
        self.conv3 = conv(in_channels=conv_dim, out_channels=(conv_dim * 2), kernel_size=4, stride=2, padding=1,
                          norm=norm, init_zero_weights=False, activ='leaky')
        self.conv4 = conv(in_channels=(conv_dim * 2), out_channels=(conv_dim * 4), kernel_size=4, stride=2, padding=1,
                          norm=norm, init_zero_weights=False, activ='leaky')
        self.conv5 = conv(in_channels=(conv_dim * 4), out_channels=(conv_dim * 8), kernel_size=4, stride=2, padding=1,
                          norm=norm, init_zero_weights=False, activ='leaky')
        self.conv6 = conv(in_channels=(conv_dim * 8), out_channels=1, kernel_size=4, stride=2, padding=0, norm=None,
                          init_zero_weights=False, activ=None)
        self.sigmoid = nn.Sigmoid()
        self.apply_sigmoid = sigmoid

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        if self.apply_sigmoid:
            x = self.sigmoid(x)
        return x.squeeze()
