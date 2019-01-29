import cv2
import numpy as np
from PIL import Image


test_image_path = '/home/zhanghangjian/COCO/annotations/stuff/masks/train2017/000000000009.png'

img = np.array(Image.open(test_image_path))

img_cv2 = cv2.imread(test_image_path)

print('ok!')

