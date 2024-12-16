import os
import os.path as osp
import numpy as np
import random
import torch
import cv2
from torch.utils import data

class MCFSDataSet(data.Dataset):
    def __init__(self, root, split="train", max_iters=None, crop_size=(321, 321), 
                 scale=True, mirror=True, ignore_label=255, pretraining='COCO'):
        """
        Args:
            root (str): Path to content/All_data directory
            split (str): 'train' or 'val'
            max_iters (int): Used for training, defines max number of iterations
            crop_size (tuple): (height, width) for crop size
            scale (bool): Enable scaling augmentation
            mirror (bool): Enable mirror augmentation
            ignore_label (int): Label value to ignore in loss computation
        """
        self.root = root
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.is_mirror = mirror
        self.split = split
        self.pretraining = pretraining

        # Setup paths based on split
        split_dir = osp.join(self.root, split)
        self.image_dir = osp.join(split_dir, "image")
        self.label_dir = osp.join(split_dir, "seg")

        # Get all image files
        self.files = []
        
        # Check if directories exist
        if not osp.exists(self.image_dir) or not osp.exists(self.label_dir):
            raise ValueError(f"Image or label directory not found at {split_dir}")
            
        image_files = sorted(os.listdir(self.image_dir))  # Sort for consistency
        
        for img_name in image_files:
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            name = osp.splitext(img_name)[0]
            label_name = f"{name}.png"
            label_path = osp.join(self.label_dir, label_name)
            
            # Verify label exists
            if not osp.exists(label_path):
                print(f"Warning: No label found for {img_name}")
                continue
                
            self.files.append({
                "img": osp.join(self.image_dir, img_name),
                "label": label_path,
                "name": name
            })

        print(f"Found {len(self.files)} valid image-label pairs")

        if len(self.files) == 0:
            raise RuntimeError(f"Found 0 images in {self.image_dir}")

        # Handle max_iters
        if max_iters is not None:
            self.files = self.files * int(np.ceil(float(max_iters) / len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        
        # Load image and label
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load image: {datafiles['img']}")
            
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise ValueError(f"Failed to load label: {datafiles['label']}")
        
        # Convert BGR to RGB
        if self.pretraining == 'COCO': # if pratraining is not COCO, change to RGB
            image = image
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        size = image.shape
        name = datafiles["name"]
        
        # Convert to float32
        image = np.asarray(image, np.float32)
        
        img_h, img_w = label.shape
        
        if self.split == "train":
            # Scale augmentation
            if self.scale:
                scale = random.uniform(0.5, 2.0)
                image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                label = cv2.resize(label, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                img_h, img_w = label.shape
            
            # Padding if necessary
            pad_h = max(self.crop_h - img_h, 0)
            pad_w = max(self.crop_w - img_w, 0)
            
            if pad_h > 0 or pad_w > 0:
                img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, 
                                           cv2.BORDER_CONSTANT, 
                                           value=(0.0, 0.0, 0.0))
                label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, pad_w,
                                             cv2.BORDER_CONSTANT,
                                             value=(self.ignore_label,))
            else:
                img_pad, label_pad = image, label
                
            img_h, img_w = label_pad.shape
            
            # Random crop
            h_off = random.randint(0, max(0, img_h - self.crop_h))
            w_off = random.randint(0, max(0, img_w - self.crop_w))
            
            image = img_pad[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w].copy()
            label = label_pad[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w].copy()
            
            # Random mirroring
            if self.is_mirror and random.random() < 0.5:
                image = np.fliplr(image).copy()  # Use copy() to ensure contiguous array
                label = np.fliplr(label).copy()

        image = image.transpose((2, 0, 1))  # HWC -> CHW
        
        # Ensure arrays are contiguous
        image = np.ascontiguousarray(image)
        label = np.ascontiguousarray(label)
        
        return (torch.from_numpy(image).float(),
                torch.from_numpy(label).long(),
                np.array(size),
                name,
                index)
