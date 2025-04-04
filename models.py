import torch
from torch import nn
import antialiased_cnns

def disney_uplayer(in_channels, out_channels, stride=1, 
                    init_zero_weights=False, activ="leaky"):
    
    # Using this terminology because that's how it was in the disney paper... want to be consistent
    C = in_channels
    two_divided_by_C = out_channels // 2


    layers = []
    # MaxBlurPool as described in the following github: https://github.com/adobe/antialiased-cnns/blob/master/README.md
    # CHECK THIS BLURUPSAMPLE IMPLEMENTATION
    blurupsample_layer_1 = [nn.Upsample(scale_factor=2), antialiased_cnns.BlurPool(channels=C, stride=stride)]
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
    two_times_C = 2 * out_channels


    layers = []
    # MaxBlurPool as described in the following github: https://github.com/adobe/antialiased-cnns/blob/master/README.md
    # The true downsampling is said to be done in the blur pool layer and NOT the maxpool, so setting the stride in the maxpool layer to 1
    # to avoid too much downsampling
    maxblurpool_layer_1 = [nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=padding), antialiased_cnns.BlurPool(channels=C, stride=stride)]
    conv_layer_2 = [nn.Conv2d(in_channels=C, out_channels=two_times_C, kernel_size=4, padding=1, stride=2)]
    conv_layer_3 = [nn.Conv2d(in_channels=two_times_C, out_channels=two_times_C, kernel_size=3, padding=1, stride=1)]
    conv_layer_4 = [nn.Conv2d(in_channels=two_times_C, out_channels=two_times_C, kernel_size=3, padding=1, stride=1)]

    if init_zero_weights:
        # for the torch.randn initialization: torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        conv_layer_2[0].weight.data = 0.001 * torch.randn(two_times_C, C, 4, 4)
        conv_layer_3[0].weight.data = 0.001 * torch.randn(two_times_C, two_times_C, 3, 3)
        conv_layer_4[0].weight.data = 0.001 * torch.randn(two_times_C, two_times_C, 3, 3)
    
    if activ == "leaky":
        conv_layer_2.append(nn.LeakyReLU())
        conv_layer_3.append(nn.LeakyReLU())
        conv_layer_4.append(nn.LeakyReLU())
    
    # in case we wanted to experiment
    elif activ == 'relu':
        conv_layer_2.append(nn.ReLU())
        conv_layer_3.append(nn.ReLU())
        conv_layer_4.append(nn.ReLU())

    layers += maxblurpool_layer_1 + conv_layer_2 + conv_layer_3 + conv_layer_4

    return nn.Sequential(*layers)


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


# Pulled from https://github.com/t-vi/pytorch-tvmisc/tree/master to implement
# Conditional Batch Normalization
class ConditionalBatchNorm(torch.nn.Module):
    # as in the chainer SN-GAN implementation, we keep per-cat weight and bias
    def __init__(self, num_features, num_cats, eps=2e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.num_cats = num_cats
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = torch.nn.Parameter(torch.Tensor(num_cats, num_features))
            self.bias = torch.nn.Parameter(torch.Tensor(num_cats, num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.fill_(1.0)
            self.bias.data.zero_()

    def forward(self, input, cats):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        out = torch.nn.functional.batch_norm(
            input, self.running_mean, self.running_var, None, None,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        if self.affine:
            shape = [input.size(0), self.num_features] + (input.dim() - 2) * [1]
            weight = self.weight.index_select(0, cats).view(shape)
            bias = self.bias.index_select(0, cats).view(shape)
            out = out * weight + bias
        return out

    def extra_repr(self):
        return '{num_features}, num_cats={num_cats}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


# Trying out a new generator
class DisneyGenerator(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = conv(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1, norm=None, activ="leaky")
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
        self.conv2 = conv(in_channels=64, out_channels=3, )# TODO: find the proper dimensions for the kernel size, padding, and stride..... kernel_size=3, padding=1, stride=1, norm=None, activ="leaky")
        

    def forward(self, z):
        # TODO: Implement the forward pass
        pass

        return z


class Generator(nn.Module):
    class Encoder(nn.Module):
        def __init__(self, conv_dim=64):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=(conv_dim // 2),
                                   kernel_size=4, stride=2, padding=1, bias=False)
            self.norm1 = nn.InstanceNorm2d(conv_dim // 2)
            self.activation1 = nn.ReLU()
            self.conv2 = nn.Conv2d(in_channels=(conv_dim // 2), out_channels=conv_dim,
                                   kernel_size=4, stride=2, padding=1, bias=False)
            self.norm2 = nn.InstanceNorm2d(conv_dim)
            self.activation2 = nn.ReLU()
            self.conv3 = nn.Conv2d(in_channels=conv_dim, out_channels=conv_dim * 2,
                                   kernel_size=4, stride=2, padding=1, bias=False)
            self.norm3 = nn.InstanceNorm2d(conv_dim * 2)
            self.activation3 = nn.ReLU()
            self.conv4 = nn.Conv2d(in_channels=conv_dim * 2, out_channels=conv_dim * 4,
                                   kernel_size=4, stride=2, padding=1, bias=False)
            self.norm4 = nn.InstanceNorm2d(conv_dim * 4)
            self.activation4 = nn.ReLU()

        def forward(self, x):
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.activation1(x)
            x = self.conv2(x)
            x = self.norm2(x)
            x = self.activation2(x)
            x = self.conv3(x)
            x = self.norm3(x)
            x = self.activation3(x)
            x = self.conv4(x)
            x = self.norm4(x)
            x = self.activation4(x)
            return x

    class AgeModulator(nn.Module):

        def __init__(self, conv_dim=64):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels=conv_dim * 4 + 1, out_channels=conv_dim * 4, kernel_size=3, padding=1)
            self.cond_batch_norm1 = ConditionalBatchNorm(conv_dim * 4, 2)
            self.activation1 = nn.LeakyReLU()
            self.fc1 = nn.Linear(2, 256)

        def forward(self, x, target_age):
            age_embedding_vector = self.fc1(target_age)
            age_embedding_vector = age_embedding_vector.resize(len(target_age), 16, 16).unsqueeze(1)
            modulated_features = torch.cat([x, age_embedding_vector], dim=1)
            output = self.conv1(modulated_features)
            _, ages = torch.where(target_age == 1)
            output = self.cond_batch_norm1(output, ages)
            output = self.activation1(output)
            return output

    class Decoder(nn.Module):
        def __init__(self, conv_dim=64):
            super().__init__()
            self.up_conv1 = nn.ConvTranspose2d(in_channels=conv_dim * 4, out_channels=conv_dim * 2,
                                               kernel_size=4, stride=2, padding=1, bias=False)
            self.norm1 = nn.InstanceNorm2d(conv_dim * 2)
            self.activation1 = nn.ReLU()
            self.up_conv2 = nn.ConvTranspose2d(in_channels=conv_dim * 2, out_channels=conv_dim,
                                               kernel_size=4, stride=2, padding=1, bias=False)
            self.norm2 = nn.InstanceNorm2d(conv_dim)
            self.activation2 = nn.ReLU()
            self.up_conv3 = nn.ConvTranspose2d(in_channels=conv_dim, out_channels=conv_dim // 2,
                                               kernel_size=4, stride=2, padding=1, bias=False)
            self.norm3 = nn.InstanceNorm2d(conv_dim // 2)
            self.activation3 = nn.ReLU()
            self.up_conv4 = nn.ConvTranspose2d(in_channels=conv_dim // 2, out_channels=3,
                                               kernel_size=4, stride=2, padding=1, bias=False)
            self.norm4 = nn.InstanceNorm2d(3)
            self.activation4 = nn.Tanh()

        def forward(self, x):
            x = self.up_conv1(x)
            x = self.norm1(x)
            x = self.activation1(x)
            x = self.up_conv2(x)
            x = self.norm2(x)
            x = self.activation2(x)
            x = self.up_conv3(x)
            x = self.norm3(x)
            x = self.activation3(x)
            x = self.up_conv4(x)
            x = self.norm4(x)
            x = self.activation4(x)
            return x

    def __init__(self, conv_dim=64):
        super().__init__()
        self.encoder = self.Encoder(conv_dim=conv_dim)
        self.decoder = self.Decoder(conv_dim=conv_dim)
        self.age_modulator = self.AgeModulator(conv_dim=conv_dim)

    def forward(self, x, target_age):
        identity_features = self.encoder(x)
        age_features = self.age_modulator(identity_features, target_age)
        return self.decoder(identity_features + age_features)


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
