import numpy as np
import random
from PIL import Image, ImageFilter
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask=None):
        for t in self.transforms:
            image, mask = t(image, mask)
        return {'image': image, 'mask': mask}


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask=None):
        image = F.resize(image, self.size)
        if mask is not None:
            mask = F.resize(mask, self.size, interpolation=F.InterpolationMode.NEAREST)
        return image, mask


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, mask=None):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            if mask is not None:
                mask = F.hflip(mask)
        return image, mask


class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, mask=None):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            if mask is not None:
                mask = F.vflip(mask)
        return image, mask


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask):
        crop_params = T.RandomCrop.get_params(image, self.size)
        image = F.crop(image, *crop_params)
        if mask is not None:
            mask = F.crop(mask, *crop_params)
        return image, mask


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask):
        image = F.center_crop(image, self.size)
        if mask is not None:
            mask = F.center_crop(mask, self.size)
        return image, mask


class Pad(object):
    def __init__(self, padding_n, padding_fill_value=0, padding_fill_target_value=0):
        self.padding_n = padding_n
        self.padding_fill_value = padding_fill_value
        self.padding_fill_target_value = padding_fill_target_value

    def __call__(self, image, mask):
        image = F.pad(image, self.padding_n, self.padding_fill_value)
        if mask is not None:
            mask = F.pad(mask, self.padding_n, self.padding_fill_target_value)
        return image, mask


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, image, mask):
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = image.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return img, mask


class RandomGaussianBlur(object):
    def __call__(self, image, mask):
        if random.random() < 0.5:
            image = image.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return image, mask


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img, mask):
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return img, mask


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, img, mask):
        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return img, mask


class ToTensor(object):
    def __call__(self, image, mask):
        img = np.array(image).astype(np.float32)
        mask = np.array(mask).astype(np.float32)

        image = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()
        return image, mask

