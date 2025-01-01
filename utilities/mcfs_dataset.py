import os
import os.path as osp
import numpy as np
import random
import torch
import cv2
from torch.utils import data



import torch
import numpy as np
import cv2
import os
from pathlib import Path

def save_debug_images(image_tensor, label_tensor, save_dir, filename_prefix, max_images=None):
    """
    Save processed images and their corresponding labels for debugging visualization.
    
    Args:
        image_tensor (torch.Tensor): Processed image tensor in CxHxW format
        label_tensor (torch.Tensor): Label tensor in HxW format
        save_dir (str): Directory to save debug images
        filename_prefix (str): Prefix for saved files
        max_images (int, optional): Maximum number of images to save if processing a batch
    
    Returns:
        tuple: Paths to saved image and label files
    """
    # Create save directory if it doesn't exist
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to numpy and handle batch dimension if present
    if image_tensor.dim() == 4:  # Batch of images
        images = image_tensor.cpu().numpy()
        labels = label_tensor.cpu().numpy()
        if max_images is not None:
            images = images[:max_images]
            labels = labels[:max_images]
    else:  # Single image
        images = image_tensor.cpu().numpy()[None]
        labels = label_tensor.cpu().numpy()[None]
    
    saved_paths = []
    
    for idx, (img, label) in enumerate(zip(images, labels)):
        # Convert image back to HWC format and handle normalization
        img = img.transpose(1, 2, 0)  # CHW -> HWC
        
        # If image was normalized, scale back to 0-255 range
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
            
        # Convert to BGR for OpenCV saving
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Create unique filenames
        img_filename = f"{filename_prefix}_img_{idx}.png"
        label_filename = f"{filename_prefix}_label_{idx}.png"
        
        img_path = save_dir / img_filename
        label_path = save_dir / label_filename
        
        # Save image and label
        cv2.imwrite(str(img_path), img_bgr)
        cv2.imwrite(str(label_path), label.astype(np.uint8))
        
        saved_paths.append((img_path, label_path))
        
        print(f"Saved debug images to:\nImage: {img_path}\nLabel: {label_path}\n")
    
    return saved_paths



class MCFSDataSet(data.Dataset):
    def __init__(self, root, split="train", max_iters=None, crop_size=(224, 224), 
                 scale=True, mirror=True, ignore_label=250, pretraining='COCO'):
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

    def _resize_image_and_label(self, image, label, target_size):
        """Helper method to resize both image and label while preserving aspect ratio"""
        h, w = image.shape[:2]
        target_h, target_w = target_size
        
        # Calculate scaling factor to preserve aspect ratio
        scale_h = target_h / h
        scale_w = target_w / w
        scale = min(scale_h, scale_w)
        
        # Calculate new dimensions
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # Resize image with antialiasing for better quality
        image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Important: Ensure label values are preserved during resizing
        # Convert label to one-hot encoding before resizing to prevent interpolation artifacts
        unique_labels = np.unique(label)
        label_resized = np.zeros((new_h, new_w), dtype=label.dtype)
        
        for lbl in unique_labels:
            if lbl == self.ignore_label:
                continue
            mask = (label == lbl)
            mask_resized = cv2.resize(mask.astype(np.uint8), (new_w, new_h), 
                                    interpolation=cv2.INTER_NEAREST)
            label_resized[mask_resized > 0] = lbl
        
        # Create empty arrays with target size
        final_image = np.zeros((target_h, target_w, 3), dtype=np.float32)
        final_label = np.full((target_h, target_w), self.ignore_label, dtype=label.dtype)
        
        # Calculate padding to center the image
        pad_h = (target_h - new_h) // 2
        pad_w = (target_w - new_w) // 2
        
        # Place resized image and label in center
        final_image[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = image_resized
        final_label[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = label_resized
        
        return final_image, final_label

    def __getitem__(self, index):
        datafiles = self.files[index]
        
        # Load image and label
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load image: {datafiles['img']}")
            
        # Important: Load label as-is without any color conversion
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise ValueError(f"Failed to load label: {datafiles['label']}")
        
        # Convert BGR to RGB if needed
        if self.pretraining != 'COCO':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        original_size = image.shape
        name = datafiles["name"]
        
        # Convert image to float32, but keep label as integer
        image = np.asarray(image, np.float32)
        
        if self.split == "train":
            if self.scale:
                scale = random.uniform(0.5, 2.0)
                scaled_h = int(image.shape[0] * scale)
                scaled_w = int(image.shape[1] * scale)
                image = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
                # Use NEAREST neighbor for label scaling to preserve label values
                label = cv2.resize(label, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
            
            # Resize to crop_size with proper interpolation and padding
            image, label = self._resize_image_and_label(image, label, (self.crop_h, self.crop_w))
            
            if self.is_mirror and random.random() < 0.5:
                image = np.fliplr(image).copy()
                label = np.fliplr(label).copy()
        else:
            image, label = self._resize_image_and_label(image, label, (self.crop_h, self.crop_w))
        
        # Add debug assertions
        assert label.min() >= 0, f"Invalid negative label values: {label.min()}"
        assert label.max() <= 255, f"Invalid label values > 255: {label.max()}"
        
        image = image.transpose((2, 0, 1))
        
        image = np.ascontiguousarray(image)
        label = np.ascontiguousarray(label)

        """image_tensor = torch.from_numpy(image).float()
        label_tensor = torch.from_numpy(label).long()
        
        # Save debug images
        save_debug_images(
            image_tensor,
            label_tensor,
            save_dir="debug_visualizations",
            filename_prefix=f"sample_{index}"
        )"""
        
        return (torch.from_numpy(image).float(),
                torch.from_numpy(label).long(),
                np.array(original_size),
                name,
                index)
