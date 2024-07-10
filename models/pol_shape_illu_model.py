"""
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
It implements a simple image-to-image translation baseline based on regression loss.

You need to implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input data and perform data pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
"""
import torch
from .base_model import BaseModel
from . import networks
import itertools

class PolShapeIlluModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(batch_size=100, dataset_mode='translucent',
                            display_freq=12800, update_html_freq=12800, print_freq=12800,
                            n_epochs=10, n_epochs_decay=10, display_ncols=2)  # You can rewrite default values for this model..
        # You can define new arguments for this model.
        parser.add_argument('--netNormal', type=str, default='sfpw', help='specify network architecture')
        parser.add_argument('--netDepth', type=str, default='sfpw', help='specify network architecture')
        parser.add_argument('--netIlluminationEncoder', type=str, default='basic_encoder', help='specify network architecture')
        parser.add_argument('--input_list', nargs='+', default='4pol', help='specify the input images')

        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        self.phase = opt.phase
        self.input_list = opt.input_list
        channel_table = {'4pol': 12, 'dop': 3, 'aop': 3, 'scene': 3,
                         'min': 3, 'max': 3, 'stokes': 9, 'encoded_aop': 6, 'pseudo_normal': 18}

        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = ['Normal', 'Depth', 'Coeffs']

        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        self.visual_names = ['normal_predict', 'depth_predict_vis',
                             'normal_image', 'depth_image_vis']
        if '4pol' in self.input_list:
            self.visual_names += ['p0_image', 'p45_image', 'p90_image', 'p135_image']
        if 'scene' in self.input_list:
            self.visual_names += ['scene_image']

        # specify the models you want to save/load to the disk.
        if self.isTrain:
            self.model_save_names = ['Normal', 'Depth', 'IlluminationEncoder']
            self.model_load_names = []
        else:
            self.model_save_names = []
            self.model_load_names = ['Normal', 'Depth', 'IlluminationEncoder']
        self.model_names = list(set(self.model_load_names + self.model_save_names))

        # compute the number of input channels
        num_channels = 0
        for channel_name in self.input_list:
            num_channels += channel_table[channel_name]
        num_channels += 3  # add background channel

        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        self.netNormal = networks.define_G(num_channels, 3, None, opt.netNormal, gpu_ids=self.gpu_ids)
        self.netDepth = networks.define_G(num_channels, 1, None, opt.netDepth, gpu_ids=self.gpu_ids)
        self.netIlluminationEncoder = networks.define_E(num_channels, 27, opt.ngf, opt.netIlluminationEncoder, isActivation=False, gpu_ids=self.gpu_ids)

        # define your loss functions. You can use losses provided by torch.nn such as torch.nn.L1Loss.
        self.L2Loss = torch.nn.MSELoss()
        if self.isTrain:  # only defined during training time
            # define and initialize optimizers. You can define one optimizer for each network, or use itertools.chain to group them.
            self.optimizer = torch.optim.Adam(itertools.chain(self.netNormal.parameters(),
                                                              self.netDepth.parameters(),
                                                              self.netIlluminationEncoder.parameters()),
                                              lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer]
        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, inp):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.input_images = []
        self.binary_mask = inp['mask'].to(self.device)
        self.num_pixel = torch.sum(self.binary_mask, dim=[-1, -2]).squeeze()

        self.p0_image = inp['p0'].to(self.device)
        self.p45_image = inp['p45'].to(self.device)
        self.p90_image = inp['p90'].to(self.device)
        self.p135_image = inp['p135'].to(self.device)
        self.background_image = self.normalize(self.inverse_normalize(self.p0_image) * ~self.binary_mask)
        self.p0_image = self.normalize(self.inverse_normalize(self.p0_image) * self.binary_mask)
        self.p45_image = self.normalize(self.inverse_normalize(self.p45_image) * self.binary_mask)
        self.p90_image = self.normalize(self.inverse_normalize(self.p90_image) * self.binary_mask)
        self.p135_image = self.normalize(self.inverse_normalize(self.p135_image) * self.binary_mask)

        if '4pol' in self.input_list:
            self.input_images.append(self.p0_image)
            self.input_images.append(self.p45_image)
            self.input_images.append(self.p90_image)
            self.input_images.append(self.p135_image)
            self.input_images.append(self.background_image)
        if 'scene' in self.input_list:
            self.scene_image = inp['scene'].to(self.device)
            self.input_images.append(self.scene_image)
            if '4pol' not in self.input_list:
                self.input_images.append(self.background_image)
        if 'min' in self.input_list:
            self.min_image = inp['min'].to(self.device)
            self.min_image = self.normalize(self.inverse_normalize(self.min_image) * self.binary_mask)
            self.input_images.append(self.min_image)
        if 'max' in self.input_list:
            self.max_image = inp['max'].to(self.device)
            self.max_image = self.normalize(self.inverse_normalize(self.max_image) * self.binary_mask)
            self.input_images.append(self.max_image)
        if 'dop' in self.input_list:
            self.DoP_image = inp['DoP'].to(self.device)
            self.input_images.append(self.DoP_image)
        if 'aop' in self.input_list:
            self.AoP_image = inp['AoP'].to(self.device)
            self.input_images.append(self.AoP_image)
        if 'encoded_aop' in self.input_list:
            self.AoP_image = inp['AoP'].to(self.device)
            self.AoP_encode = torch.cat([torch.cos(self.AoP_image * 2), torch.sin(self.AoP_image * 2)], dim=1)
            self.input_images.append(self.AoP_encode)
        if 'stokes' in self.input_list:
            self.S0_image = inp['S0'].to(self.device)
            self.S1_image = inp['S1'].to(self.device)
            self.S2_image = inp['S2'].to(self.device)
            self.input_images.append(self.S0_image)
            self.input_images.append(self.S1_image)
            self.input_images.append(self.S2_image)
            if '4pol' not in self.input_list:
                self.input_images.append(self.background_image)
        if 'pseudo_normal' in self.input_list:
            self.pseudo_normal_image_0 = inp['pseudo_normal_0'].to(self.device)
            self.pseudo_normal_image_1 = inp['pseudo_normal_1'].to(self.device)
            self.pseudo_normal_image_2 = inp['pseudo_normal_2'].to(self.device)
            self.pseudo_normal_image_3 = inp['pseudo_normal_3'].to(self.device)
            self.pseudo_normal_image_4 = inp['pseudo_normal_4'].to(self.device)
            self.pseudo_normal_image_5 = inp['pseudo_normal_5'].to(self.device)
            self.input_images.append(self.pseudo_normal_image_0)
            self.input_images.append(self.pseudo_normal_image_1)
            self.input_images.append(self.pseudo_normal_image_2)
            self.input_images.append(self.pseudo_normal_image_3)
            self.input_images.append(self.pseudo_normal_image_4)
            self.input_images.append(self.pseudo_normal_image_5)
        if 'normal' in inp:
            self.normal_image = inp['normal'].to(self.device)
            self.depth_image = inp['depth'].to(self.device)
            self.coeffs_para = inp['coeffs'].to(self.device)

        if self.phase == 'test':
            self.image_paths = inp['image_paths']

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        inp = torch.cat(self.input_images, dim=1)
        self.normal_predict= self.netNormal(inp)
        self.depth_predict = self.netDepth(inp)
        self.coeffs_predict = self.netIlluminationEncoder(inp)
        self.coeffs_predict = self.coeffs_predict.view(self.coeffs_predict.size(0), 3, 9)

    def compute_loss(self):
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results

        self.loss_Normal = self.imageLoss(self.normal_predict, self.normal_image)
        self.loss_Depth = self.imageLoss(self.depth_predict, self.depth_image)
        # self.loss_Scatter = self.L2Loss(self.scatter_predict, self.scatter_para)
        self.loss_Coeffs = self.L2Loss(self.coeffs_predict, self.coeffs_para)

        self.loss = 100 * self.loss_Normal + \
                    self.loss_Depth + \
                    0.1 * self.loss_Coeffs

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""

        self.compute_loss()
        self.loss.backward()       # calculate gradients of network w.r.t. loss

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()               # first call forward to calculate intermediate results
        for optimizer in self.optimizers:
            optimizer.zero_grad()    # clear networks' existing gradients
        self.backward()              # calculate gradients for network s
        for optimizer in self.optimizers:
            optimizer.step()        # update gradients for networks

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        if hasattr(self, 'depth_image'):
            self.depth_image_vis = self.normalize_depth(self.depth_image)

        self.depth_predict_vis = self.normalize_depth(self.depth_predict)
        self.normal_predict = self.normalize(self.inverse_normalize(self.normal_predict) * self.binary_mask)

    def normalize_depth(self, depth_raw):
        """Normalize depth image to [-1, 1] for visualization"""
        depth_raw -= self.binary_mask * 100
        depth = (depth_raw - depth_raw.min()) * self.binary_mask
        depth = (depth/depth.max()) * 2 - 1
        return depth

    @staticmethod
    def normalize_radiance(radiance):
        """raw radiance values are between [0.35, 0.75], normalize them to [-1, 1]"""
        radiance -= 0.35
        radiance /= 0.4
        return radiance * 2 - 1

    @staticmethod
    def inverse_normalize_radiance(radiance):
        """map radiance values back to [0.35, 0.75]"""
        radiance = (radiance + 1) / 2
        radiance *= 0.4
        radiance += 0.35
        return radiance

    @staticmethod
    def normalize_albedo(albedo):
        """raw albedo values are between [0.3, 0.95], normalize them to [-1, 1]"""
        albedo -= 0.3
        albedo /= 0.65
        return albedo * 2 - 1

    @staticmethod
    def inverse_normalize_albedo(albedo):
        """map albedo values back to [0.3, 0.95]"""
        albedo = (albedo + 1) / 2
        albedo *= 0.65
        albedo += 0.3
        return albedo

    @staticmethod
    def normalize_g(g):
        """raw g values are between [0.0, 0.9], normalize them to [-1, 1]"""
        g /= 0.9
        return g * 2 - 1

    @staticmethod
    def inverse_normalize_g(g):
        """map g values back to [0.0, 0.9]"""
        g = (g + 1) / 2
        g *= 0.9
        return g

    @staticmethod
    def normalize(inp):
        return inp * 2 - 1

    @staticmethod
    def inverse_normalize(image):
        return (image + 1) / 2

    def imageLoss(self, im1, im2):
        num_channel = im1.shape[1]
        return torch.mean(torch.sum(torch.abs(im1 - im2) * self.binary_mask, dim=[-1, -2, -3]) / (self.num_pixel * num_channel))