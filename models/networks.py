import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from torch.nn import LayerNorm

###############################################################################
# Helper Functions
###############################################################################

class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], isDepth=False):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | unet_256
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, isDepth=isDepth)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, isDepth=isDepth)
    elif netG == 'resnet_renderer':
        net = ResnetRenderer(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_predictor':
        net = ResnetPredictor(input_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=15)
    elif netG == 'resnet_predictor_twoshot':
        net = ResnetPredictorTwoshot(input_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=15)
    elif netG == 'sfpw':
        net = TransUnet(input_nc, n_classes=output_nc)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)



def define_E(input_nc, output_dim, ngf, netE, isActivation=True,  norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """
    todo:
        - make encoder flexible for any size of input feature maps, currently hard coding
    """

    """Create a encoder

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_dim (int) -- the number of dimensions in output vector
        ngf (int) -- the number of filters in the last conv layer
        netE (str) -- the architecture's name: basic_encoder | 
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a encoder

    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netE == 'basic_encoder':
        net = BasicEncoder(input_nc, output_dim, ngf, norm_layer=norm_layer, isActivation=isActivation)
    elif netE == 'sss_encoder':
        net = ResnetPredictorSSS(input_nc, ngf, output_dim=output_dim, norm_layer=norm_layer, isActivation=isActivation)
    else:
        raise NotImplementedError('Encoder model name [%s] is not recognized' % netE)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_De(input_dim, netDe, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):

    """Create a decoder

    Parameters:
        input_dim (int) -- the number of dimensions in input vector
        netDe (str) -- the architecture's name: basic_decoder |
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a encoder

    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netDe == 'basic_decoder':
        net = BasicDecoder(input_dim, norm_layer=norm_layer)
    elif netDe == 'mid_decoder':
        net = MidDecoder(input_dim, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netDe)
    return init_net(net, init_type, init_gain, gpu_ids)

##############################################################################
# Classes
##############################################################################

class ResnetRenderer(nn.Module):
    """Resnet-based renderer that consists of Resnet blocks between a few downsampling/upsampling operations.

    """
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetRenderer, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        downsampler = [nn.ReflectionPad2d(3),
                       nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                       norm_layer(ngf),
                       nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            downsampler += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
        self.downsampler = nn.Sequential(*downsampler)

        resnet_blocks = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            resnet_blocks += [ResnetBlock(ngf * mult + 32, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.resnet_blocks = nn.Sequential(*resnet_blocks)

        upsampler = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            upsampler += [nn.ConvTranspose2d(ngf * mult + 32, int(ngf * mult / 2) + 32,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2) + 32),
                      nn.ReLU(True)]
        upsampler += [nn.ReflectionPad2d(3)]
        upsampler += [nn.Conv2d(ngf + 32, output_nc, kernel_size=7, padding=0)]
        upsampler += [nn.Tanh()]

        self.upsampler = nn.Sequential(*upsampler)

    def forward(self, input):
        """Standard forward"""
        BRDF_images, light_feature = input
        BRDF_feature = self.downsampler(BRDF_images)

        return self.upsampler(self.resnet_blocks(torch.cat([BRDF_feature, light_feature], dim=1)))


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', isDepth=False):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if isDepth:
            model += [nn.ReLU(True)]
        else:
            model += [nn.Tanh()]


        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class ResnetPredictor(nn.Module):
    def __init__(self, input_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetPredictor, self).__init__()

        self.feature_encoder = ResnetEncoder(input_nc, ngf, norm_layer, use_dropout, n_blocks, padding_type)
        self.normal_decoder = ImageDecoder(3, ngf, norm_layer)
        self.depth_decoder = ImageDecoder(1, ngf, norm_layer, isDepth=True)
        self.scatter_encoder = FeatureEncoder(8, ngf, norm_layer, isActivation=True)
        self.coeff_encoder = FeatureEncoder(27, ngf, norm_layer, isActivation=False)

    def forward(self, input):
        feature = self.feature_encoder(input)
        normal = self.normal_decoder(feature)
        depth = self.depth_decoder(feature)
        scatter = self.scatter_encoder(feature)
        coeff = self.coeff_encoder(feature)
        return (normal, depth, scatter, coeff)

class ResnetPredictorTwoshot(nn.Module):
    def __init__(self, input_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetPredictorTwoshot, self).__init__()

        self.feature_encoder = ResnetEncoder(input_nc, ngf, norm_layer, use_dropout, n_blocks, padding_type)
        self.normal_decoder = ImageDecoder(3, ngf, norm_layer)
        self.rough_decoder = ImageDecoder(1, ngf, norm_layer)
        self.depth_decoder = ImageDecoder(1, ngf, norm_layer, isDepth=True)
        self.scatter_encoder = FeatureEncoder(8, ngf, norm_layer, isActivation=True)
        self.coeff_encoder = FeatureEncoder(27, ngf, norm_layer, isActivation=False)

    def forward(self, input):
        feature = self.feature_encoder(input)
        normal = self.normal_decoder(feature)
        rough = self.rough_decoder(feature)
        depth = self.depth_decoder(feature)
        scatter = self.scatter_encoder(feature)
        coeff = self.coeff_encoder(feature)
        return (normal, depth, rough, scatter, coeff)

class ResnetPredictorSSS(nn.Module):
    def __init__(self, input_nc, ngf=64, output_dim=8, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=15, padding_type='reflect', isActivation=True):
        assert(n_blocks >= 0)
        super(ResnetPredictorSSS, self).__init__()

        self.feature_encoder = ResnetEncoder(input_nc, ngf, norm_layer, use_dropout, n_blocks, padding_type)
        self.scatter_encoder = FeatureEncoder(output_dim, ngf, norm_layer, isActivation=isActivation)

    def forward(self, input):
        feature = self.feature_encoder(input)
        scatter = self.scatter_encoder(feature)
        return scatter


class ResnetEncoder(nn.Module):
    def __init__(self, input_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetEncoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, isDepth=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, isDepth=isDepth)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, isDepth=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            if isDepth:
                up = [uprelu, upconv, nn.ReLU(True)]
            else:
                up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class BasicEncoder(nn.Module):

    def __init__(self, input_nc, output_dim, ngf=64, norm_layer=nn.BatchNorm2d, isActivation=True):

        super(BasicEncoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 3
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]


        model += [nn.Flatten(),
                  nn.Linear(32*32*512, output_dim),
                  nn.BatchNorm1d(output_dim),
                  ]
        if isActivation:
            model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class BasicDecoder(nn.Module):
    def __init__(self, input_dim, norm_layer=nn.BatchNorm2d):
        super(BasicDecoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model = [nn.ConvTranspose2d(input_dim, 1024, kernel_size=4, stride=2, padding=1, bias=use_bias),  # 2x2
                 norm_layer(1024),
                 nn.ReLU(True),
                 nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=use_bias),         # 4x4
                 norm_layer(512),
                 nn.ReLU(True),
                 nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=use_bias),          # 8x8
                 norm_layer(256),
                 nn.ReLU(True),
                 nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),          # 16x16
                 norm_layer(128),
                 nn.ReLU(True),
                 nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=use_bias),            # 32x32
                 norm_layer(64),
                 nn.ReLU(True),
                 nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=use_bias),             # 64x64
                 norm_layer(32),
                 nn.ReLU(True)
                 ]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        input = torch.unsqueeze(torch.unsqueeze(input, dim=-1), dim=-1)
        return self.model(input)

class MidDecoder(nn.Module):
    def __init__(self, input_dim, norm_layer=nn.BatchNorm2d):
        super(MidDecoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model = [nn.ConvTranspose2d(input_dim, 1024, kernel_size=4, stride=2, padding=1, bias=use_bias),  # 2x2
                 norm_layer(1024),
                 nn.ReLU(True),
                 nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=use_bias),         # 4x4
                 norm_layer(512),
                 nn.ReLU(True),
                 nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=use_bias),          # 8x8
                 norm_layer(256),
                 nn.ReLU(True),
                 nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),          # 16x16
                 norm_layer(128),
                 nn.ReLU(True),
                 nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=use_bias),            # 32x32
                 norm_layer(64),
                 nn.ReLU(True),
                 nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=use_bias),             # 64x64
                 norm_layer(32),
                 nn.ReLU(True),
                 nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=use_bias),             # 128x128
                 norm_layer(16),
                 nn.ReLU(True),
                 nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1, bias=use_bias),              # 256x256
                 norm_layer(1),
                 nn.Tanh()
                 ]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        input = torch.unsqueeze(torch.unsqueeze(input, dim=-1), dim=-1)
        return self.model(input)


class ImageDecoder(nn.Module):

    def __init__(self, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, isDepth=False):

        super(ImageDecoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = []
        n_downsampling = 2
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if isDepth:
            model += [nn.ReLU(True)]
        else:
            model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class FeatureEncoder(nn.Module):

    def __init__(self, output_dim, ngf=64, norm_layer=nn.BatchNorm2d, isActivation=False):

        super(FeatureEncoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = []
        model += [nn.Conv2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(int(ngf * 2)),
                  nn.ReLU(True),
                  nn.Conv2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(int(ngf)),
                  nn.ReLU(True)
                  ]
        model += [nn.Flatten(),
                  nn.Linear(16 * 16 * ngf, output_dim),
                  nn.BatchNorm1d(output_dim),
                  ]
        if isActivation:
            model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class LayerNormConv2d(nn.Module):
    """
    Layer norm the just works on the channel axis for a Conv2d
    Ref:
    - code modified from https://github.com/Scitator/Run-Skeleton-Run/blob/master/common/modules/LayerNorm.py
    - paper: https://arxiv.org/abs/1607.06450
    Usage:
        ln = LayerNormConv(3)
        x = Variable(torch.rand((1,3,4,2)))
        ln(x).size()
    """

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features).cuda()).unsqueeze(-1).unsqueeze(-1)
        self.beta = nn.Parameter(torch.zeros(features).cuda()).unsqueeze(-1).unsqueeze(-1)
        self.eps = eps
        self.features = features

    def _check_input_dim(self, input):
        if input.size(1) != self.gamma.nelement():
            raise ValueError('got {}-feature tensor, expected {}'
                             .format(input.size(1), self.features))

    def forward(self, x):
        self._check_input_dim(x)
        x_flat = x.transpose(1, -1).contiguous().view((-1, x.size(1)))
        mean = x_flat.mean(0).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        std = x_flat.std(0).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        return self.gamma.expand_as(x) * (x - mean) / (std + self.eps) + self.beta.expand_as(x)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None,  kernel_size= 3, norm='bn'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if norm == "bn":
            norm_fn = nn.BatchNorm2d
        elif norm == "in":
            norm_fn = nn.InstanceNorm2d
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            norm_fn(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            norm_fn(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, norm='bn'):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, kernel_size=kernel_size, norm=norm)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, norm='bn'):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, norm=norm)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, norm=norm)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Mlp(nn.Module):
    def __init__(self, dim, mult=4, dropout=0):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(dim, dim * mult)
        self.fc2 = nn.Linear(dim * mult, dim)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


import math
class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout = 0.):
        super(Attention, self).__init__()
        self.heads = heads
        context_dim = context_dim or query_dim
        hidden_dim = max(query_dim, context_dim)
        # self.dim_head = int(hidden_dim / self.heads)
        self.dim_head = dim_head
        self.all_head_dim = self.heads * self.dim_head

        ## All linear layers (including query, key, and value layers and dense block layers)
        ## preserve the dimensionality of their inputs and are tiled over input index dimensions #
        # (i.e. applied as a 1 × 1 convolution).
        self.query = nn.Linear(query_dim, self.all_head_dim) # (b n d_q) -> (b n hd)
        self.key = nn.Linear(context_dim, self.all_head_dim) # (b m d_c) -> (b m hd)
        self.value = nn.Linear(context_dim, self.all_head_dim) # (b m d_c) -> (b m hd)
        self.out = nn.Linear(self.all_head_dim, query_dim) # (b n d) -> (b n d)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.heads, self.dim_head)
        x = x.view(*new_x_shape) # (b n hd) -> (b n h d)
        return x.permute(0, 2, 1, 3) # (b n h d) -> (b h n d)

    def forward(self, query, context=None):
        if context is None:
            context = query
        mixed_query_layer = self.query(query) # (b n d_q) -> (b n hd)
        mixed_key_layer = self.key(context) # (b m d_c) -> (b m hd)
        mixed_value_layer = self.value(context) # (b m d_c) -> (b m hd)

        query_layer = self.transpose_for_scores(mixed_query_layer) # (b h n d)
        key_layer = self.transpose_for_scores(mixed_key_layer) # (b h m d)
        value_layer = self.transpose_for_scores(mixed_value_layer) # (b h m d)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (b h n m)
        attention_scores = attention_scores / math.sqrt(self.dim_head) # (b h n m)
        attention_probs = self.softmax(attention_scores) # (b h n m)
        attention_probs = self.attn_dropout(attention_probs) # (b h n m)

        context_layer = torch.matmul(attention_probs, value_layer) # (b h n m) , (b h m d) -> (b h n d)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # (b h n d)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_dim,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class Block(nn.Module):
    def __init__(self, hidden_size, droppath=0., dropout=0.0):
        super(Block, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn = Mlp(hidden_size)
        self.attn = Attention(hidden_size, dropout=dropout)
        # self.drop_path = DropPath(droppath) if droppath > 0. else nn.Identity()
        self.drop_path =  nn.Identity()

    def forward(self, x):
        x = self.drop_path(self.attn(self.attention_norm(x))) + x
        x = self.drop_path(self.ffn(self.ffn_norm(x))) + x
        return x


class TransUnet(nn.Module):
    def __init__(self, n_channels, n_classes=3, dim=64, residual_num=8, bilinear=True, norm='in', dropout=0.0,
                 skip_res=True):
        super(TransUnet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.skip_res = skip_res

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(dim, dim * 2, norm=norm)
        self.down2 = Down(dim * 2, dim * 4, norm=norm)
        self.down3 = Down(dim * 4, dim * 8, norm=norm)
        factor = 2 if bilinear else 1
        self.down4 = Down(dim * 8, dim * 16 // factor, norm=norm)
        self.up1 = Up(dim * 16, dim * 8 // factor, bilinear)
        self.up2 = Up(dim * 8, dim * 4 // factor, bilinear)
        self.up3 = Up(dim * 4, dim * 2 // factor, bilinear)
        self.up4 = Up(dim * 2, dim, bilinear)
        self.outc = OutConv(dim, n_classes)
        self.resblock_layers = nn.ModuleList([])

        # Missing positinal encoding
        for i in range(residual_num):
            self.resblock_layers.append(Block(dim * 8, dropout=dropout))

            # self.resblock_layers.append(BasicBlock(512, 512, norm_layer=nn.LayerNorm))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        b, c, h, w = x5.size()
        x5 = torch.reshape(x5, [b, c, h * w]).permute(0, 2, 1)
        # print(x5.size())
        for resblock in self.resblock_layers:
            residual = resblock(x5)
            if self.skip_res:
                # print("residual", residual[0,0,0])
                # import ipdb; ipdb.set_trace()
                x5 = residual
                # print("x5", x5[0,0,0])
            else:
                x5 = x5 + residual
        x5 = torch.reshape(x5.permute(0, 2, 1), [b, c, h, w])
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
