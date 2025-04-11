import torch
from torch import nn
import antialiased_cnns


##########################
# Disney Implementations
##########################

def disney_uplayer(in_channels, out_channels, upsample_mode="nearest", 
                    init_zero_weights=False, activ="leaky"):
    
    # Using this terminology because that's how it was in the disney paper... want to be consistent
    C = in_channels
    two_divided_by_C = in_channels // 2

    assert two_divided_by_C == out_channels, "ERROR: In the Generator uplayer, the nuuber of output channels is not half of the number of iput channels (output_channels != input_channels/2)"


    layers = []

    # BlurUpSample
    blurupsample_layer_1 = None
    if upsample_mode == "nearest":
        blurupsample_layer_1 = [nn.Upsample(scale_factor=2, mode='nearest'), antialiased_cnns.BlurPool(channels=C, stride=1)] # Setting the stride to 1 so there isn't any down sampling

    elif upsample_mode == "bilinear": 
        # bilinear upsampling is said to be better for image generation tasks
        blurupsample_layer_1 = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), antialiased_cnns.BlurPool(channels=C, stride=1)] # Setting the stride to 1 so there isn't any down sampling
    
    conv_layer_2 = [nn.Conv2d(in_channels=C, out_channels=two_divided_by_C, kernel_size=3, padding=1, stride=1)]
    conv_layer_3 = [nn.Conv2d(in_channels=two_divided_by_C, out_channels=two_divided_by_C, kernel_size=3, padding=1, stride=1)]

    if init_zero_weights:
        # for the torch.randn initialization: torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        conv_layer_2[0].weight.data = 0.001 * torch.randn(two_divided_by_C, C, 3, 3)
        conv_layer_3[0].weight.data = 0.001 * torch.randn(two_divided_by_C, two_divided_by_C, 3, 3)
    
    if activ == "leaky":
        conv_layer_2.append(nn.LeakyReLU())
        conv_layer_3.append(nn.LeakyReLU())
    
    # in case we wanted to experiment
    elif activ == 'relu':
        conv_layer_2.append(nn.ReLU())
        conv_layer_3.append(nn.ReLU())

    layers += blurupsample_layer_1 + conv_layer_2 + conv_layer_3

    return nn.Sequential(*layers)

def disney_downlayer(in_channels, out_channels, kernel_size=4, stride=2, 
                     padding=1, init_zero_weights=False, activ="leaky"):
    
    # Using this terminology because that's how it was in the disney paper... want to be consistent
    C = in_channels
    two_times_C = 2 * in_channels

    assert two_times_C == out_channels, "ERROR: In the Generator downlayer, the number of output channels is not double the number of input channels (output_channels != input_channels * 2)"


    layers = []
    # MaxBlurPool as described in the following github: https://github.com/adobe/antialiased-cnns/blob/master/README.md
    # The true downsampling is said to be done in the blur pool layer and NOT the maxpool, so setting the stride in the maxpool layer to 1
    # to avoid too much downsampling
    maxblurpool_layer_1 = [nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=padding), antialiased_cnns.BlurPool(channels=C, stride=stride)]
    conv_layer_2 = [nn.Conv2d(in_channels=C, out_channels=two_times_C, kernel_size=3, padding=1, stride=1)]
    conv_layer_3 = [nn.Conv2d(in_channels=two_times_C, out_channels=two_times_C, kernel_size=3, padding=1, stride=1)]

    if init_zero_weights:
        # for the torch.randn initialization: torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        conv_layer_2[0].weight.data = 0.001 * torch.randn(two_times_C, C, 3, 3)
        conv_layer_3[0].weight.data = 0.001 * torch.randn(two_times_C, two_times_C, 3, 3)
    
    if activ == "leaky":
        conv_layer_2.append(nn.LeakyReLU())
        conv_layer_3.append(nn.LeakyReLU())
    
    # in case we wanted to experiment
    elif activ == 'relu':
        conv_layer_2.append(nn.ReLU())
        conv_layer_3.append(nn.ReLU())

    layers += maxblurpool_layer_1 + conv_layer_2 + conv_layer_3 

    return nn.Sequential(*layers)


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1,
         norm='batch', init_zero_weights=False, activ=None):
    """Create a normal convolutional layer, with optional normalization."""
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

def patch_gan_conv(in_channels, out_channels, kernel_size, stride=2, padding=1,
         norm='batch', init_zero_weights=False, activ=None):
    """Create a PatchGAN convolutional layer, with optional normalization."""
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
        # Typical PatchGANs use negative_slope of 0.2
        layers.append(nn.LeakyReLU(0.2))

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


# Trying out a new generator
class DisneyGenerator(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = conv(in_channels=5, out_channels=64, kernel_size=3, padding=1, stride=1, norm=None, activ="leaky")
        # ------------
        self.conv2 = conv(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1, norm=None, activ="leaky")
        # ------------
        self.downlayer3 = disney_downlayer(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, activ="leaky")
        self.downlayer4 = disney_downlayer(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, activ="leaky")
        self.downlayer5 = disney_downlayer(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, activ="leaky")
        self.downlayer6 = disney_downlayer(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1, activ="leaky")
        # ------------
        self.uplayer7 = disney_uplayer(in_channels=1024, out_channels=512, activ="leaky")
        self.uplayer8 = disney_uplayer(in_channels=512, out_channels=256, activ="leaky")
        self.uplayer9 = disney_uplayer(in_channels=256, out_channels=128, activ="leaky")
        self.uplayer10 = disney_uplayer(in_channels=128, out_channels=64, activ="leaky")
        # ------------
        self.conv11 = conv(in_channels=64, out_channels=3, kernel_size=3, padding=1, stride=1, norm=None, activ=None)
        

    def forward(self, z):
        x = self.conv1(z)
        x = self.conv2(x)
        x = self.downlayer3(x)
        x = self.downlayer4(x)
        x = self.downlayer5(x)
        x = self.downlayer6(x)
        x = self.uplayer7(x)
        x = self.uplayer8(x)
        x = self.uplayer9(x)
        x = self.uplayer10(x)
        x = self.conv11(x)

        return x
    
# PatchGAN adapted from https://github.com/znxlwm/pytorch-pix2pix/blob/3059f2af53324e77089bbcfc31279f01a38c40b8/network.py#L104
class DiscriminatorPatchGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = patch_gan_conv(in_channels=8, out_channels=64, kernel_size=4, padding=1, stride=2, norm="batch", activ="leaky")
        self.conv2 = patch_gan_conv(in_channels=64, out_channels=128, kernel_size=4, padding=1, stride=2, norm="batch", activ="leaky")
        self.conv3 = patch_gan_conv(in_channels=128, out_channels=256, kernel_size=4, padding=1, stride=2, norm="batch", activ=None) # using BCEWithLogitsLoss... so do not need sigmoid as the final activation

    # forward method
    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


##########################
# Other Implementations
##########################
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
