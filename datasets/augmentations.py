# Adapted from https://github.com/ZijunDeng/pytorch-semantic-segmentation/blob/master/utils/joint_transforms.py

import math
import random
import numbers
import numpy as np

from PIL import Image, ImageOps


def resize(img_w, img_h, size, max_size=1000):
    """
    Resize the input PIL image to the given size.
    The bounding boxes change the same.
    :param img_h:
    :param img_w:
    :param size:     (tuple or int)
                     - if is tuple, resize image to the size.
                     - if is int, resize the shorter side to the size while maintaining the aspect ratio.
    :param max_size: (int) when size is int, limit the image longer size to max_size.
                     - this is essential to limit the usage of GPU memory.
    :return:

    """
    if isinstance(size, int):
        size_min = min(img_w, img_h)
        size_max = max(img_w, img_h)

        sw = sh = float(size) / size_min
        if sw * size_max > max_size:
            sw = sh = float(max_size) / size_max

        ow = int(img_w * sw + 0.5)
        oh = int(img_h * sh + 0.5)
    else:
        ow, oh = size
        sw = float(ow) / img_w
        sh = float(oh) / img_h
    return ow, oh, sw, sh


def random_crop(img_w, img_h):

    ox, oy, ow, oh = 0, 0, img_w, img_h
    success = False
    for attempt in range(10):
        area = img_w * img_h
        target_area = random.uniform(0.56, 1.0) * area
        aspect_ratio = random.uniform(3. / 4, 4. / 3)

        ow = int(round(math.sqrt(target_area * aspect_ratio)))
        oh = int(round(math.sqrt(target_area / aspect_ratio)))

        if random.random() < 0.5:
            ow, oh = oh, ow

        if ow <= img_w and oh <= img_h:
            ox = random.randint(0, img_w - ow)
            oy = random.randint(0, img_h - oh)
            success = True
            break

    # Fallback
    if not success:
        ow = oh = min(img_w, img_h)
        ox = (img_w - ow) // 2
        oy = (img_h - oh) // 2

    return ox, oy, ow, oh


def center_crop(img_w, img_h, size):

    ow, oh = size
    ox = int(round((img_w - ow) / 2.))
    oy = int(round((img_h - oh) / 2.))

    return ox, oy, ow, oh


class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, img, mask):
        img, mask = Image.fromarray(img, mode='RGB'), Image.fromarray(mask, mode='L')
        assert img.size == mask.size

        for a in self.augmentations:
            img, mask = a(img, mask)
        return np.array(img, dtype=np.uint8), np.array(mask, dtype=np.uint8)


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask


class FreeScale(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask):
        assert img.size == mask.size
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST)


class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        if (w >= h and w == self.size[1]) or (h >= w and h == self.size[0]):
            return img, mask

        oh, ow = self.size
        return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                return img.resize((self.size, self.size), Image.BILINEAR), mask.resize((self.size, self.size),
                                                                                       Image.NEAREST)

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, mask))


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST)


class RandomSized(object):
    def __init__(self, limit):
        self.limit = limit

    def __call__(self, img, mask):

        scale = random.uniform(self.limit[0], self.limit[1])
        w = int(scale * img.size[0])
        h = int(scale * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)

        return img, mask
