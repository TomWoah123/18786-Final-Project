import torch
from torch import nn


class DownLayer(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.conv_one = nn.Conv2d(in_channels=input_channels, out_channels=input_channels * 2,
                                  kernel_size=3, padding=1, stride=1)
        self.leaky_relu = nn.LeakyReLU()
        self.conv_two = nn.Conv2d(in_channels=input_channels * 2, out_channels=input_channels * 2,
                                  kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        x = self.max_pool(x)
        x = self.conv_one(x)
        x = self.leaky_relu(x)
        x = self.conv_two(x)
        x = self.leaky_relu(x)
        return x


class UpLayer(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv_one = nn.Conv2d(in_channels=input_channels, out_channels=input_channels // 2,
                                  kernel_size=3, padding=1, stride=1)
        self.leaky_relu = nn.LeakyReLU()
        self.conv_two = nn.Conv2d(in_channels=input_channels // 2, out_channels=input_channels // 2,
                                  kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        x = self.up_sample(x)
        x = self.conv_one(x)
        x = self.leaky_relu(x)
        x = self.conv_two(x)
        x = self.leaky_relu(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_one = nn.Conv2d(in_channels=5, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.batch_norm = nn.BatchNorm2d(num_features=64)
        self.leaky_relu = nn.LeakyReLU()
        self.conv_two = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.down_one = DownLayer(64)
        self.down_two = DownLayer(128)
        self.down_three = DownLayer(256)
        self.down_four = DownLayer(512)
        self.up_four = UpLayer(1024)
        self.up_three = UpLayer(512)
        self.up_two = UpLayer(256)
        self.up_one = UpLayer(128)
        self.conv_three = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)

    def forward(self, x):
        conv_one_output = self.conv_one(x)
        conv_one_output = self.batch_norm(conv_one_output)
        conv_one_output = self.leaky_relu(conv_one_output)
        conv_two_output = self.conv_two(conv_one_output)
        conv_two_output = self.batch_norm(conv_two_output)
        conv_two_output = self.leaky_relu(conv_two_output)
        down_one_output = self.down_one(conv_two_output)
        down_two_output = self.down_two(down_one_output)
        down_three_output = self.down_three(down_two_output)
        down_four_output = self.down_four(down_three_output)
        up_four_output = self.up_four(down_four_output) + down_three_output
        up_three_output = self.up_three(up_four_output) + down_two_output
        up_two_output = self.up_two(up_three_output) + down_one_output
        up_one_output = self.up_one(up_two_output) + conv_two_output
        output = self.conv_three(up_one_output)
        return output


class Discriminator(nn.Module):

    def conv(self, in_channels, out_channels, kernel_size, stride=2, padding=1,
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

    def __init__(self, conv_dim=64, norm="batch"):
        super().__init__()
        self.conv1 = self.conv(in_channels=4, out_channels=(conv_dim // 2), kernel_size=4, stride=2, padding=1,
                               norm=norm, init_zero_weights=False, activ='leaky')
        self.conv2 = self.conv(in_channels=(conv_dim // 2), out_channels=conv_dim, kernel_size=4, stride=2, padding=1,
                               norm=norm, init_zero_weights=False, activ='leaky')
        self.conv3 = self.conv(in_channels=conv_dim, out_channels=(conv_dim * 2), kernel_size=4, stride=2, padding=1,
                               norm=norm, init_zero_weights=False, activ='leaky')
        self.conv4 = self.conv(in_channels=(conv_dim * 2), out_channels=(conv_dim * 4), kernel_size=4, stride=2,
                               padding=1, norm=norm, init_zero_weights=False, activ='leaky')
        self.conv5 = self.conv(in_channels=(conv_dim * 4), out_channels=1, kernel_size=4, stride=2, padding=1,
                               norm=None, init_zero_weights=False, activ=None)
        self.fc = nn.Linear(256, 1)

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


input_image = torch.randn(size=(16, 4, 512, 512))
discriminator = Discriminator()
result = discriminator(input_image)
