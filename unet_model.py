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
        return x.squeeze()


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
        return x.squeeze()


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_one = nn.Conv2d(in_channels=5, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.leaky_relu_one = nn.LeakyReLU()
        self.conv_two = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.leaky_relu_two = nn.LeakyReLU()
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
        conv_one_output = self.leaky_relu_one(conv_one_output)
        conv_two_output = self.conv_two(conv_one_output)
        conv_two_output = self.leaky_relu_two(conv_two_output)
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
