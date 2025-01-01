import numbers
import cv2
import numpy as np
import PIL
import torch


def _is_tensor_clip(clip):
    return torch.is_tensor(clip) and clip.ndimension() == 4


def crop_clip(clip, min_h, min_w, h, w):
    if isinstance(clip[0], np.ndarray):
        cropped = [img[min_h:min_h + h, min_w:min_w + w, :] for img in clip]

    elif isinstance(clip[0], PIL.Image.Image):
        cropped = [
            img.crop((min_w, min_h, min_w + w, min_h + h)) for img in clip
        ]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return cropped


def resize_clip(clip, target_size, interpolation='bilinear', padding_mode='constant', padding_value=0):
    """
    Resize a clip of images to a fixed square dimension (n x n) using interpolation and padding.
    Maintains aspect ratio and adds padding to create a square output.
    
    Args:
        clip: List of images (either numpy arrays or PIL Images)
        target_size: Integer specifying the desired width and height
        interpolation: String, either 'bilinear' or 'nearest'
        padding_mode: String, padding mode ('constant', 'edge', 'reflect', 'symmetric')
        padding_value: Value to use for constant padding
    
    Returns:
        List of resized images with dimensions target_size x target_size
    """
    def resize_with_padding(img, is_numpy=True):
        if is_numpy:
            orig_h, orig_w = img.shape[:2]
            aspect = orig_w / orig_h
            
            if aspect > 1:  # wider than tall
                new_w = target_size
                new_h = int(target_size / aspect)
                pad_top = (target_size - new_h) // 2
                pad_bottom = target_size - new_h - pad_top
                pad_left, pad_right = 0, 0
            else:  # taller than wide
                new_h = target_size
                new_w = int(target_size * aspect)
                pad_left = (target_size - new_w) // 2
                pad_right = target_size - new_w - pad_left
                pad_top, pad_bottom = 0, 0
                
            # First resize with proper interpolation
            if interpolation == 'bilinear':
                resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                
            # Then pad to square
            if len(img.shape) == 3:  # Color image
                padding = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
            else:  # Grayscale
                padding = ((pad_top, pad_bottom), (pad_left, pad_right))
                
            return np.pad(resized, padding, mode=padding_mode, constant_values=padding_value)
            
        else:  # PIL Image
            orig_w, orig_h = img.size
            aspect = orig_w / orig_h
            
            if aspect > 1:
                new_w = target_size
                new_h = int(target_size / aspect)
            else:
                new_h = target_size
                new_w = int(target_size * aspect)
                
            # First resize with proper interpolation
            if interpolation == 'bilinear':
                resized = img.resize((new_w, new_h), PIL.Image.BILINEAR)
            else:
                resized = img.resize((new_w, new_h), PIL.Image.NEAREST)
                
            # Create new square image with padding
            new_img = PIL.Image.new(img.mode, (target_size, target_size), padding_value)
            paste_x = (target_size - new_w) // 2
            paste_y = (target_size - new_h) // 2
            new_img.paste(resized, (paste_x, paste_y))
            
            return new_img
    
    if isinstance(clip[0], np.ndarray):
        scaled = [resize_with_padding(img, is_numpy=True) for img in clip]
    elif isinstance(clip[0], PIL.Image.Image):
        scaled = [resize_with_padding(img, is_numpy=False) for img in clip]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                       'but got list of {0}'.format(type(clip[0])))
    
    return scaled


def get_resize_sizes(im_h, im_w, size):
    if im_w < im_h:
        ow = size
        oh = int(size * im_h / im_w)
    else:
        oh = size
        ow = int(size * im_w / im_h)
    return oh, ow


def normalize(clip, mean, std, inplace=False):
    if not _is_tensor_clip(clip):
        raise TypeError('tensor is not a torch clip.')

    if not inplace:
        clip = clip.clone()

    dtype = clip.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])

    return clip
