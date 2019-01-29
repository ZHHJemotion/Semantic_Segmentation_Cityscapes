import os
import cv2
import time
import numpy as np

from PIL import Image, ImageDraw
from lxml import etree, objectify
from matplotlib import pyplot as plt
from datasets.coco_api.PythonAPI.pycocotools.coco import COCO
from datasets.coco_api.PythonAPI.pycocotools import mask as mask_utils


# noinspection PyTypeChecker
class COCO2Pascal(object):
    def __init__(self, root, split="train", year="2017"):
        """
        Convert COCO annotation to Pascal VOC annotation
        :param root:  (string): file path to COCO folder.
        :param split: (string): image set to use (eg. 'train', 'val', 'test')
        """
        self.root = root
        self.year = year
        self.split = split  # "train" "val" "test"

        self.ann_file = os.path.join(root, "annotations/stuff_{}{}.json".format(self.split, self.year))
        self.coco = COCO(self.ann_file)
        self.ids = list(self.coco.imgs.keys())

        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple(['__background__'] + [c['name'] for c in cats])
        self.num_classes = len(self._classes)
        self._class_to_ind = dict(zip(self._classes, range(self.num_classes)))
        self._class_to_coco_cat_id = dict(zip([c['name'] for c in cats], self.coco.getCatIds()))

        # Lookup table to map from COCO category ids to our internal class indices
        self.coco_cat_id_to_class_ind = dict([(self._class_to_coco_cat_id[cls], self._class_to_ind[cls])
                                              for cls in self._classes[1:]])
        self.name = ""

    @staticmethod
    def __decode_rle(annotation, img_height, img_width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segmentation = annotation['segmentation']

        if type(segmentation) is list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = mask_utils.frPyObjects(segmentation, img_height, img_width)
            rle = mask_utils.merge(rles)
        else:
            # rle
            if type(segmentation['counts']) == list:
                rle = mask_utils.frPyObjects([annotation['segmentation']], img_height, img_width)
            else:
                rle = [annotation['segmentation']]

        return mask_utils.decode(rle)

    def __root(self, folder, filename, img_width, img_height):
        element = objectify.ElementMaker(annotate=False)
        return element.annotation(element.folder(folder),
                                  element.filename(filename),
                                  element.source(element.database('MS COCO {}'.format(self.year)),
                                                 element.annotation('MS COCO {}'.format(self.year)),
                                                 element.image('Flickr'),),
                                  element.size(element.width(img_width),
                                               element.height(img_height),
                                               element.depth(3),),
                                  element.segmented(1))

    @staticmethod
    def __instance_to_xml(x_min, y_min, x_max, y_max, cat_id):
        element = objectify.ElementMaker(annotate=False)
        return element.object(element.name(cat_id),
                              element.bndbox(element.xmin(x_min),
                                             element.ymin(y_min),
                                             element.xmax(x_max),
                                             element.ymax(y_max),),)

    def _parse_annotation(self, index):
        """
        Loads COCO segmentation semantic annotations.
        Crowd instances are handled by marking their overlaps (with all categories) to -1. 
        This overlap value means that crowd "instances" are excluded from training.
        :param index: (int) index of the coco image
        :return: 
        """
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        img_name, img_width, img_height = img_info['file_name'], img_info['width'], img_info['height']

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        annotations = self.coco.loadAnns(ann_ids)
        annotations = sorted(annotations, key=lambda x: x['area'])  # let smaller items first

        mask_image = np.zeros((img_height, img_width), dtype=np.uint8)  # image to store the mask of the objects

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 1. Processing objects in the current image
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        obj_index = 0
        for annotation in annotations:
            if not annotation['iscrowd']:  # here we ignore crowd annotations
                # COCO bounding box annotation: (x_min, y_min, width, height)
                #                      ======>> (x_min, y_min, x_max, y_max)
                x_min = np.max((0, annotation['bbox'][0]))
                y_min = np.max((0, annotation['bbox'][1]))
                x_max = np.min((img_width - 1, x_min + np.max((0, annotation['bbox'][2] - 1))))
                y_max = np.min((img_height - 1, y_min + np.max((0, annotation['bbox'][3] - 1))))

                if annotation['area'] > 100 and x_max > x_min and y_max > y_min:  # why remove area < 100 ???
                    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
                    # 1.1 Masks
                    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
                    if 'segmentation' in annotation:
                        cat_id = annotation['category_id']
                        cls_id = self.coco_cat_id_to_class_ind[cat_id]
                        mask = self.__decode_rle(annotation, img_height, img_width)  # decode masks from annotation
                        mask = mask.squeeze(2)

                        obj_index += 1
                        mask_image = np.where(mask_image > 0, mask_image, mask*cls_id)

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 2. Save the instance masks and bounding boxes
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        if obj_index is not 0:
            print(">       {}, objects count: {}".format(img_name, obj_index))
        else:
            print("> !!! +++++++ Image {} NOT contain a object +++++++ !!! ".format(img_name))

        return mask_image  # box_cls, mask_image

    def convert(self):
        for index in np.arange(len(self.ids)):
            img_id = self.ids[index]
            img_name = self.coco.loadImgs(img_id)[0]['file_name']
            img_path = os.path.join(self.root, "{}{}/{}".format(self.split, self.year, img_name))

            if not os.path.isfile(img_path) or not os.path.exists(img_path):
                raise Exception("> !!! +++++++ {} is not a file, can not open with "
                                "imageio.imread(...). !!! +++++++ ".format(img_path))

            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
            # 1. Read image
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
            print("> {}. Converting image: {}".format(str(index + 1).zfill(6), img_name))
            # box_cls, mask = self._parse_annotation(index)
            mask = self._parse_annotation(index)
            mask = Image.fromarray(mask, mode="L")

            mask_path = os.path.join(self.root, "annotations/stuff/masks/{}{}/{}".format(self.split,
                                                                                         self.year,
                                                                                         img_name.replace('.jpg', '.png')))
            mask.save(mask_path)

            # mask_test = np.array(Image.open(mask_path))
            print(">       {}, semantic mask saved !!!".format(img_name))

            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
            # 2. Show result
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
            is_show = False
            if is_show:
                image = Image.open(img_path)
                image = np.array(image).copy()
                mask = np.array(mask).copy()

                img_msk = [image, mask]
                fig, axes = plt.subplots(nrows=1, ncols=2)

                for ax, feat in zip(axes.flat, img_msk):
                    feat = feat.astype(np.uint8)
                    ax.imshow(feat, cmap='viridis')

                plt.show()


if __name__ == '__main__':
    year = "2017"
    traval = "val"  # train, val
    coco_root = os.path.join("/home/zhanghangjian/", "COCO")

    # pascal_ann_root = os.path.join(coco_root, "annotations/stuff/pascal/{}{}".format(traval, year))
    instance_mask_root = os.path.join(coco_root, "annotations/stuff/masks/{}{}".format(traval, year))

    # if not os.path.exists(pascal_ann_root):
    #     os.mkdir(pascal_ann_root)
    if not os.path.exists(instance_mask_root):
        os.mkdir(instance_mask_root)

    converter = COCO2Pascal(root=coco_root, split=traval, year=year)

    print("> ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ <")
    print("> Starting conversion ...")
    print("> ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ <")
    start_time = time.time()
    converter.convert()
    print("> ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ <")
    print("> Conversion ended | Time cost: {}s !!!".format(time.time()-start_time))
    print("> ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ <")
