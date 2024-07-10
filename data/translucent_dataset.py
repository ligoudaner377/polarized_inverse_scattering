import glob
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import pandas as pd
import numpy as np
import os
import torch
import scipy.ndimage as ndimage


class TranslucentDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """

        if is_train:
            parser.add_argument('--use_reconstruction_loss', action='store_true', help='use reconstruction loss or not')
        else:
            parser.set_defaults(num_test=18000)
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """

        BaseDataset.__init__(self, opt)
        self.phase = opt.phase
        self.input_list = opt.input_list
        if self.phase == 'train':
            self.use_reconstruction_loss = opt.use_reconstruction_loss
        # get scene properties
        csv_path = os.path.join(opt.dataroot, opt.phase + '.csv')
        self.csv = pd.read_csv(csv_path,
                               sep=';',
                               header=None,
                               index_col=0,
                               names=['obj', 'envmap', 'normal', 'radiance',
                                      'sigma_t', 'albedo', 'g',
                                      'obj_scale', 'obj_rotate_x', 'obj_rotate_y', 'obj_rotate_z',
                                      'obj_translate_x', 'obj_translate_y', 'obj_translate_z',
                                      'env_rotate_x', 'env_rotate_y', 'env_rotate_z',
                                      'uv_scale'])

        # get the image paths of your dataset;
        self.image_paths = sorted(glob.glob(os.path.join(opt.dataroot, opt.phase, '[0-9]*')))

        self.transform_rgb = get_transform(grayscale=False)
        self.transform_gray = get_transform(grayscale=True)

    def __getitem__(self, index):

        base_path = self.image_paths[index]
        num = int(base_path.split(os.sep)[-1])

        sigma_t = self.csv['sigma_t'][num]
        sigma_t = torch.tensor([float(i) for i in sigma_t.split(',')], dtype=torch.float32)

        albedo = self.csv['albedo'][num]
        albedo = torch.tensor([float(i) for i in albedo.split(',')], dtype=torch.float32)

        g = self.csv['g'][num]
        g = torch.tensor(g, dtype=torch.float32)

        radiance = self.csv['radiance'][num]
        radiance = torch.tensor(radiance, dtype=torch.float32) / 100

        normal_path = os.path.join(base_path, 'normal.png')
        normal_image = self.load_image(normal_path, mode='RGB')

        depth_path = os.path.join(base_path, 'depth.npy')
        depth_image = self.load_npy(depth_path)

        mask_path = os.path.join(base_path, 'mask.png')
        mask_image = self.load_mask(mask_path)

        coeffs_path = os.path.join(base_path, 'coeffs.npy')
        coeffs = np.load(coeffs_path).transpose([1, 0])[:, 0:9].astype(np.float32)

        res = {'normal': normal_image, 'depth': depth_image, 'mask': mask_image,
               'g': g, 'sigma_t': sigma_t, 'albedo': albedo, 'radiance': radiance,
               'coeffs': coeffs, 'num': num}

        p0_path = os.path.join(base_path, '0.png')
        p0_image = self.load_image(p0_path, mode="RGB")
        p0_raw = p0_image / 2 + 0.5
        res['p0'] = p0_image

        p45_path = os.path.join(base_path, '45.png')
        p45_image = self.load_image(p45_path, mode="RGB")
        p45_raw = p45_image / 2 + 0.5
        res['p45'] = p45_image

        p90_path = os.path.join(base_path, '90.png')
        p90_image = self.load_image(p90_path, mode="RGB")
        p90_raw = p90_image / 2 + 0.5
        res['p90'] = p90_image

        p135_path = os.path.join(base_path, '135.png')
        p135_image = self.load_image(p135_path, mode="RGB")
        p135_raw = p135_image / 2 + 0.5
        res['p135'] = p135_image

        if 'scene' in self.input_list:
            scene_path = os.path.join(base_path, 'scene.png')
            scene_image = self.load_image(scene_path, mode="RGB")
            res['scene'] = scene_image

        if 'stokes' in self.input_list:
            S0_path = os.path.join(base_path, 'S0.npy')
            S1_path = os.path.join(base_path, 'S1.npy')
            S2_path = os.path.join(base_path, 'S2.npy')
            S0_image = self.load_npy(S0_path).permute(2, 0, 1)
            S1_image = self.load_npy(S1_path).permute(2, 0, 1)
            S2_image = self.load_npy(S2_path).permute(2, 0, 1)
            res['S0'] = S0_image
            res['S1'] = S1_image
            res['S2'] = S2_image

        if 'dop' in self.input_list:
            DoP_path = os.path.join(base_path, 'DoP.npy')
            DoP_image = self.load_npy(DoP_path).permute(2, 0, 1)
            res['DoP'] = DoP_image

        if ('aop' in self.input_list) or ('encoded_aop' in self.input_list):
            AoP_path = os.path.join(base_path, 'AoP.npy')
            AoP_image = self.load_npy(AoP_path).permute(2, 0, 1)
            res['AoP'] = AoP_image

        if 'min' in self.input_list:
            min_path = os.path.join(base_path, 'min.png')
            min_image = self.load_image(min_path, mode="RGB")
            res['min'] = min_image

        if 'max' in self.input_list:
            max_path = os.path.join(base_path, 'max.png')
            max_image = self.load_image(max_path, mode="RGB")
            res['max'] = max_image

        if 'pseudo_normal' in self.input_list:
            for i in range(6):
                pseudo_normal_path = os.path.join(base_path, f'pseudo_normal_{i}.png')
                pseudo_normal_image = self.load_image(pseudo_normal_path, mode="RGB")
                res[f'pseudo_normal_{i}'] = pseudo_normal_image

        if self.phase == 'train':
            if self.use_reconstruction_loss:
                b0_path = os.path.join(base_path, '0_BSDF.png')
                b0_image = self.load_image(b0_path, mode="RGB")
                res['b0'] = b0_image

                b45_path = os.path.join(base_path, '45_BSDF.png')
                b45_image = self.load_image(b45_path, mode="RGB")
                res['b45'] = b45_image

                b90_path = os.path.join(base_path, '90_BSDF.png')
                b90_image = self.load_image(b90_path, mode="RGB")
                res['b90'] = b90_image

                b135_path = os.path.join(base_path, '135_BSDF.png')
                b135_image = self.load_image(b135_path, mode="RGB")
                res['b135'] = b135_image

        if self.phase == 'test':
            res['image_paths'] = base_path

        return res

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)

    def load_image(self, path, mode='RGB'):
        """Load and transform image based on the given path and mode."""
        image = Image.open(path).convert(mode)
        assert image.size[0] == 256
        if mode == "RGB":
            data = self.transform_rgb(image)
        if mode == "L":
            data = self.transform_gray(image)
        return data

    def load_mask(self, path):
        mask = Image.open(path).convert('L')
        assert mask.size[0] == 256
        mask = np.array(mask, dtype=np.float32) / 255.0
        mask = (mask > 0.999999).astype(dtype=np.int)
        mask = ndimage.binary_erosion(mask, structure=np.ones((2, 2)))
        mask = mask[np.newaxis, :, :]
        return mask

    def load_npy(self, path):
        data = torch.tensor(np.load(path), dtype=torch.float32)
        return data