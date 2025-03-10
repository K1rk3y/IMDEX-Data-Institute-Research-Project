'''
Code taken from https://github.com/hfslyc/AdvSemiSeg
Slightly modified
'''

import os.path as osp
import numpy as np
import random
import torch
import cv2
from torch.utils import data

class VOCDataSet(data.Dataset):
    def __init__(self, root, split="train", max_iters=None, crop_size=(321, 321), scale=True, mirror=True, ignore_label=255, pretraining='COCO'):
        self.root = root
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.pretraining = pretraining
        self.ignore_label = ignore_label
        self.is_mirror = mirror
        self.split=split
        if split == "train":
            list_path = './data/voc_list/train_aug.txt'
        elif split == "val":
            list_path = './data/voc_list/val.txt'

        self.img_ids = [i_id.strip() for i_id in open(list_path)]

        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))

        self.files = []
        
        for name in self.img_ids:
            img_file = osp.join(self.root, "JPEGImages/%s.jpg" % name)
            label_file = osp.join(self.root, "SegmentationClassAug/%s.png" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
        self.class_names = ['background',  # always index 0
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor']

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        if self.pretraining == 'COCO': # if pratraining is not COCO, change to RGB
            image = image
        else:
            image = image[:, :, ::-1]

        size = image.shape
        name = datafiles["name"]
        image = np.asarray(image, np.float32)
        img_h, img_w = label.shape
        if "val" not in self.split:  # output size with pad or crop
            pad_h = max(self.crop_h - img_h, 0)
            pad_w = max(self.crop_w - img_w, 0)
            if pad_h > 0 or pad_w > 0:
                img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                    pad_w, cv2.BORDER_CONSTANT,
                    value=(0.0, 0.0, 0.0))
                label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                    pad_w, cv2.BORDER_CONSTANT,
                    value=(self.ignore_label,))
            else:
                img_pad, label_pad = image, label

            img_h, img_w = label_pad.shape
            h_off = random.randint(0, img_h - self.crop_h)
            w_off = random.randint(0, img_w - self.crop_w)
            image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
            label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.int64)

        image = image.transpose((2, 0, 1))
        label = label.astype(int)

        return torch.from_numpy(image).float(), torch.from_numpy(label).long(), np.array(size), name, index
