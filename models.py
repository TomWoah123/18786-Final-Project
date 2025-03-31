import torch
from torch import nn


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1,
         norm='batch', init_zero_weights=False, activ=None):
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


class ConditionalBatchNorm(nn.Module):
    def __init__(self, num_features, num_conditions):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm1d(num_features, affine=False)
        self.embed = nn.Embedding(num_conditions, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at 1
        self.embed.weight.data[:, num_features:].zero_()  # Initialise shift at 0

    def forward(self, x, y):
        out = self.bn(x)
        embed = self.embed(y)
        gamma, beta = embed.chunk(2, 1)
        out = gamma * out + beta
        return out


class Generator(nn.Module):
    class Encoder(nn.Module):
        def __init__(self, conv_dim=64):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=(conv_dim // 2),
                                   kernel_size=4, stride=2, padding=1, bias=False)
            self.cond_batch_norm1 = ConditionalBatchNorm(conv_dim // 2, 2)
            self.activation1 = nn.ReLU()
            self.conv2 = nn.Conv2d(in_channels=(conv_dim // 2), out_channels=conv_dim,
                                   kernel_size=4, stride=2, padding=1, bias=False)
            self.cond_batch_norm2 = ConditionalBatchNorm(conv_dim, 2)
            self.activation2 = nn.ReLU()
            self.conv3 = nn.Conv2d(in_channels=conv_dim, out_channels=conv_dim * 2,
                                   kernel_size=4, stride=2, padding=1, bias=False)
            self.cond_batch_norm3 = ConditionalBatchNorm(conv_dim * 2, 2)
            self.activation3 = nn.ReLU()
            self.conv4 = nn.Conv2d(in_channels=conv_dim * 2, out_channels=conv_dim * 4,
                                   kernel_size=4, stride=2, padding=1, bias=False)
            self.cond_batch_norm4 = ConditionalBatchNorm(conv_dim * 4, 2)
            self.activation4 = nn.ReLU()

        def forward(self, x, target_age):
            x = self.conv1(x)
            x = self.cond_batch_norm1(x, target_age)
            x = self.activation1(x)
            x = self.conv2(x)
            x = self.cond_batch_norm2(x, target_age)
            x = self.activation2(x)
            x = self.conv3(x)
            x = self.cond_batch_norm3(x, target_age)
            x = self.activation3(x)
            x = self.conv4(x)
            x = self.cond_batch_norm4(x, target_age)
            x = self.activation4(x)
            return x

    class AgeModulator(nn.Module):

        def __init__(self, conv_dim=64):
            pass

        def forward(self, x, target_age):
            pass

    class Decoder(nn.Module):
        def __init__(self, conv_dim=64):
            super().__init__()
            self.up_conv1 = nn.ConvTranspose2d(in_channels=conv_dim * 4, out_channels=conv_dim * 2,
                                            kernel_size=4, stride=2, padding=1, bias=False)
            self.cond_batch_norm1 = ConditionalBatchNorm(conv_dim * 2, 2)
            self.activation1 = nn.ReLU()
            self.up_conv2 = nn.ConvTranspose2d(in_channels=conv_dim * 2, out_channels=conv_dim,
                                            kernel_size=4, stride=2, padding=1, bias=False)
            self.cond_batch_norm2 = ConditionalBatchNorm(conv_dim, 2)
            self.activation2 = nn.ReLU()
            self.up_conv3 = nn.ConvTranspose2d(in_channels=conv_dim, out_channels=conv_dim // 2,
                                            kernel_size=4, stride=2, padding=1, bias=False)
            self.cond_batch_norm3 = ConditionalBatchNorm(conv_dim // 2, 2)
            self.activation3 = nn.ReLU()
            self.up_conv4 = nn.ConvTranspose2d(in_channels=conv_dim // 2, out_channels=3,
                                            kernel_size=4, stride=2, padding=1, bias=False)
            self.cond_batch_norm4 = ConditionalBatchNorm(3, 2)
            self.activation4 = nn.ReLU()

        def forward(self, x, target_age):
            x = self.up_conv1(x)
            x = self.cond_batch_norm1(x, target_age)
            x = self.activation1(x)
            x = self.up_conv2(x)
            x = self.cond_batch_norm2(x, target_age)
            x = self.activation2(x)
            x = self.up_conv3(x)
            x = self.cond_batch_norm3(x, target_age)
            x = self.activation3(x)
            x = self.up_conv4(x)
            x = self.cond_batch_norm4(x, target_age)
            x = self.activation4(x)
            return x

    def __init__(self, conv_dim=64):
        super().__init__()
        self.encoder = self.Encoder(conv_dim=conv_dim)

    def forward(self, x, target_age):
        identity_features = self.encoder(x, target_age)


class Discriminator(nn.Module):
    def __init__(self, conv_dim=64, norm="instance"):
        super().__init__()
        self.conv1 = conv(in_channels=3, out_channels=(conv_dim // 2), kernel_size=4, stride=2, padding=1, norm=norm,
                          init_zero_weights=False, activ='relu')
        self.conv2 = conv(in_channels=(conv_dim // 2), out_channels=conv_dim, kernel_size=4, stride=2, padding=1,
                          norm=norm, init_zero_weights=False, activ='relu')
        self.conv3 = conv(in_channels=conv_dim, out_channels=(conv_dim * 2), kernel_size=4, stride=2, padding=1,
                          norm=norm, init_zero_weights=False, activ='relu')
        self.conv4 = conv(in_channels=(conv_dim * 2), out_channels=(conv_dim * 4), kernel_size=4, stride=2, padding=1,
                          norm=norm, init_zero_weights=False, activ='relu')
        self.conv5 = conv(in_channels=(conv_dim * 4), out_channels=1, kernel_size=4, stride=2, padding=0, norm=None,
                          init_zero_weights=False, activ=None)
        self.fc = nn.Linear(49, 1)

    def forward(self, x):
        sigmoid = nn.Sigmoid()
        flatten = nn.Flatten()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = flatten(x)
        x = self.fc(x)
        x = sigmoid(x)
        return x.squeeze()
