"""
    Data Loader for COCO Stuff Task

"""

import os
import random
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from datasets.coco_api.PythonAPI.pycocotools.coco import COCO
from datasets.coco_api.PythonAPI.pycocotools import mask as mask_utils
from datasets.augmentations import resize, random_crop, center_crop


class COCOStuffLoader(data.Dataset):
    def __init__(self, root, split="train", year="2017", img_size=(512, 512), transform=None, is_augment=False):
        self.root = root
        self.split = split
        self.year = year
        self.img_size = img_size
        self.transform = transform

        self.class2cat_id = {'cloth': 104, 'furniture-other': 123, 'straw': 163, 'vegetable': 170,
                             'structural-other': 164, 'textile-other': 167, 'rug': 152, 'wall-tile': 176,
                             'wall-stone': 175, 'salad': 153, 'other': 183, 'skyscraper': 158, 'floor-wood': 118,
                             'sand': 154, 'water-other': 178, 'wall-wood': 177, 'leaves': 129, 'moss': 134,
                             'floor-marble': 114, 'clothes': 105, 'wall-panel': 174, 'stairs': 161, 'wall-brick': 171,
                             'door-stuff': 112, 'flower': 119, 'counter': 107, 'cupboard': 108, 'road': 149,
                             'carpet': 101, 'clouds': 106, 'river': 148, 'platform': 144, 'fog': 120, 'rock': 150,
                             'mat': 131, 'railroad': 147, 'shelf': 156, 'roof': 151, 'stone': 162, 'pavement': 140,
                             'mirror-stuff': 133, 'table': 165, 'desk-stuff': 110, 'blanket': 93, 'hill': 127,
                             'tree': 169, 'plastic': 143, 'branch': 94, 'floor-stone': 116, 'fence': 113, 'cage': 99,
                             'pillow': 141, 'house': 128, 'dirt': 111, 'window-blind': 180, 'floor-tile': 117,
                             'window-other': 181, 'napkin': 137, 'ceiling-other': 102, 'wall-concrete': 172,
                             'waterdrops': 179, 'railing': 146, 'wood': 182, 'mud': 136, 'cabinet': 98, 'towel': 168,
                             'cardboard': 100, 'bush': 97, 'sea': 155, 'ceiling-tile': 103, 'fruit': 122,
                             'food-other': 121, 'playingfield': 145, 'light': 130, 'floor-other': 115, 'gravel': 125,
                             'building-other': 96, 'bridge': 95, 'grass': 124, 'tent': 166, 'net': 138, 'banner': 92,
                             'curtain': 109, 'plant-other': 142, 'ground-other': 126, 'wall-other': 173, 'metal': 132,
                             'sky-other': 157, 'solid-other': 160, 'paper': 139, 'mountain': 135, 'snow': 159}

        self.cat_id2cls_id = {92: 1, 93: 2, 94: 3, 95: 4, 96: 5, 97: 6, 98: 7, 99: 8, 100: 9, 101: 10,
                              102: 11, 103: 12, 104: 13, 105: 14, 106: 15, 107: 16, 108: 17, 109: 18, 110: 19, 111: 20,
                              112: 21, 113: 22, 114: 23, 115: 24, 116: 25, 117: 26, 118: 27, 119: 28, 120: 29, 121: 30,
                              122: 31, 123: 32, 124: 33, 125: 34, 126: 35, 127: 36, 128: 37, 129: 38, 130: 39, 131: 40,
                              132: 41, 133: 42, 134: 43, 135: 44, 136: 45, 137: 46, 138: 47, 139: 48, 140: 49, 141: 50,
                              142: 51, 143: 52, 144: 53, 145: 54, 146: 55, 147: 56, 148: 57, 149: 58, 150: 59, 151: 60,
                              152: 61, 153: 62, 154: 63, 155: 64, 156: 65, 157: 66, 158: 67, 159: 68, 160: 69, 161: 70,
                              162: 71, 163: 72, 164: 73, 165: 74, 166: 75, 167: 76, 168: 77, 169: 78, 170: 79, 171: 80,
                              172: 81, 173: 82, 174: 83, 175: 84, 176: 85, 177: 86, 178: 87, 179: 88, 180: 89, 181: 90,
                              182: 91, 183: 92}

        self.num_classes = 93

        self.images_base = os.path.join(self.root, "{}{}".format(self.split, self.year))
        self.mask_base = os.path.join(self.root, "annotations/stuff/masks/{}{}".format(self.split, self.year))

        self.mask_list = os.listdir(self.mask_base)

        self.transform = transform
        self.is_augment = is_augment

        self.min_box_size = 10

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, index):
        """
        Load image and process annotations to prepare bounding boxes and masks for current image
        :param   index: (int) image index.
        :return: image: (tensor) image tensor.
                 loc_targets: (tensor) bounding box targets.
                 cls_targets: (tensor) class label targets.
                 msk_targets: (tensor) mask  targets.
        """
        ann_name = self.mask_list[index]
        image_name = ann_name
        image_id = int(ann_name.split('.')[0])

        mask_path = os.path.join(self.mask_base, image_name)
        image_path = os.path.join(self.images_base, image_name.replace('png', 'jpg'))

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 1. Loading image (RGB)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        img_width, img_height = image.size

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 2. Preparing bounding boxes and masks
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        masks_image = Image.open(mask_path)

        if self.is_augment:
            if self.split == "train":
                # ---------------------------------------- #
                # Randomly Flip the image and masks
                # ---------------------------------------- #
                if random.random() < 0.5:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    masks_image = masks_image.transpose(Image.FLIP_LEFT_RIGHT)
                image = image.resize((self.img_size[0], self.img_size[1]), Image.BILINEAR)
                masks_image = masks_image.resize((self.img_size[0], self.img_size[1]), Image.NEAREST)

            elif self.split == 'val':
                image = image.resize((self.img_size[0], self.img_size[1]), Image.BILINEAR)
                masks_image = masks_image.resize((self.img_size[0], self.img_size[1]), Image.NEAREST)

        masks_image = np.array(masks_image).copy()
        masks_image = np.expand_dims(masks_image, axis=0)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 3. Transform of the image bounding box and masks
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        image = self.transform(image)

        masks = torch.LongTensor(masks_image)

        return image, masks, image_id

    def collate_fn(self, batch):
        """Pad images and encode targets.

        As for images are of different sizes, we need to pad them to the same size.

        Args:
          batch: (list) of images, cls_targets, loc_targets.

        Returns:
          padded images, stacked cls_targets, stacked loc_targets.
        """
        images = [x[0] for x in batch]
        masks = [x[1] for x in batch]
        images_id = [x[2] for x in batch]

        batch_size = len(images)
        inputs = torch.zeros(batch_size, 3, self.img_size[1], self.img_size[0])

        sem_masks = []

        for idx in range(batch_size):
            inputs[idx] = images[idx]
            sem_masks.append(masks[idx])

        gt_masks = torch.cat(sem_masks, dim=0)

        return inputs, gt_masks, images_id


if __name__ == '__main__':

    import math
    import time
    import multiprocessing
    from tqdm import tqdm
    from torch.autograd import Variable

    coco_root = os.path.join("/home/pingguo/PycharmProject", "COCO")

    print("> # +++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")
    print("> Data DataLoader...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_loader = COCOStuffLoader(coco_root, split="train", year="2017",
                              img_size=(512, 512),
                              transform=transform, is_augment=True)

    valid_loader = COCOStuffLoader(coco_root, split="val", year="2017",
                              img_size=(512, 512),
                              transform=transform, is_augment=True)

    tra_loader = data.DataLoader(train_loader, batch_size=2,
                                 num_workers=int(multiprocessing.cpu_count() / 2),
                                 shuffle=True, collate_fn=train_loader.collate_fn)
    val_loader = data.DataLoader(valid_loader, batch_size=2,
                                 num_workers=int(multiprocessing.cpu_count() / 2),
                                 shuffle=True, collate_fn=train_loader.collate_fn)

    start_epoch = 0
    num_batches = int(math.ceil(len(tra_loader.dataset.mask_list) / float(tra_loader.batch_size)))
    for epoch in np.arange(start_epoch, 120):
        pbar = tqdm(np.arange(num_batches))

        for train_i, (images, masks, images_id) in enumerate(
                tra_loader):  # One mini-Batch datasets, One iteration
            full_iter = (epoch * num_batches) + train_i + 1

            pbar.update(1)
            pbar.set_description("> Epoch [%d/%d]" % (epoch + 1, 120))

            images = Variable(images.cuda(), requires_grad=True)  # Image feed into the deep neural network
            masks = Variable(masks.cuda(), requires_grad=False)
