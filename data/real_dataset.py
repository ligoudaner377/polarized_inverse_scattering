import glob
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import numpy as np
import os
import torch
import scipy.ndimage as ndimage

class RealDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # parser.set_defaults(new_dataset_option=2.0)  # specify dataset-specific default values

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
        # get the image paths of your dataset;
        self.image_paths = sorted(glob.glob(os.path.join(opt.dataroot, '[0-9]*')))
        self.transform_rgb = get_transform(grayscale=False)
        self.transform_gray = get_transform(grayscale=True)
        self.input_list = opt.input_list

    def __getitem__(self, index):
        base_path = self.image_paths[index]
        num = int(base_path.split(os.sep)[-1])

        mask_path = os.path.join(base_path, 'mask.png')
        mask_image = self.load_mask(mask_path)

        res = {'mask': mask_image, 'num': num, 'image_paths': base_path}

        p0_path = os.path.join(base_path, 'p0.png')
        p0_image = self.load_image(p0_path, mode="RGB")
        p0_raw = p0_image / 2 + 0.5
        res['p0'] = p0_image

        p45_path = os.path.join(base_path, 'p45.png')
        p45_image = self.load_image(p45_path, mode="RGB")
        p45_raw = p45_image / 2 + 0.5
        res['p45'] = p45_image

        p90_path = os.path.join(base_path, 'p90.png')
        p90_image = self.load_image(p90_path, mode="RGB")
        p90_raw = p90_image / 2 + 0.5
        res['p90'] = p90_image

        p135_path = os.path.join(base_path, 'p135.png')
        p135_image = self.load_image(p135_path, mode="RGB")
        p135_raw = p135_image / 2 + 0.5
        res['p135'] = p135_image

        min_path = os.path.join(base_path, 'min.png')
        min_image = self.load_image(min_path, mode="RGB")
        res['min'] = min_image

        max_path = os.path.join(base_path, 'max.png')
        max_image = self.load_image(max_path, mode="RGB")
        res['max'] = max_image

        scene_path = os.path.join(base_path, 'scene.png')
        scene_image = self.load_image(scene_path, mode="RGB")
        res['scene'] = scene_image

        S0_image = (p0_raw + p45_raw + p90_raw + p135_raw) / 2
        S1_image = p90_raw - p0_raw
        S2_image = 2 * p45_raw - S0_image
        res['S0'] = S0_image
        res['S1'] = S1_image
        res['S2'] = S2_image

        DoP_image = (S1_image**2 + S2_image**2)**0.5 / S0_image
        res['DoP'] = DoP_image

        AoP_image = 0.5 * torch.atan(S2_image / S1_image)
        res['AoP'] = AoP_image
        return res

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)

    def load_image(self, path, mode='RGB'):
        """Load and transform image based on the given path and mode."""
        image = Image.open(path).convert(mode)
        if image.size[0] != 256:
            image = image.resize((256, 256))
        if mode == "RGB":
            data = self.transform_rgb(image)
        if mode == "L":
            data = self.transform_gray(image)
        return data

    def load_mask(self, path):
        mask = Image.open(path).convert('L')
        if mask.size[0] != 256:
            mask = mask.resize((256, 256))
        mask = np.array(mask, dtype=np.float32) / 255.0
        if len(mask.shape) == 3:
            mask = (mask[:, :, 0] > 0.999999).astype(dtype=np.int)
        else:
            mask = (mask > 0.999999).astype(dtype=np.int)
        mask = ndimage.binary_erosion(mask, structure=np.ones((2, 2)))
        mask = mask[np.newaxis, :, :]
        return mask

    def load_npy(self, path):
        data = torch.tensor(np.load(path), dtype=torch.float32)
        return data
