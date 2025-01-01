import numpy as np
import random
import torch
import cv2


def save_debug_frames(buffer, save_dir, video_idx=0, color_space='RGB'):
    """
    Save frames from the video buffer as RGB images with proper color space handling.
    
    Args:
        buffer (torch.Tensor or numpy.ndarray): Video frames tensor of shape (C, T, H, W) or (T, C, H, W)
        save_dir (str): Directory where frames will be saved
        video_idx (int): Video index for naming the frames
        color_space (str): Color space of input buffer ('RGB', 'BGR', 'GBR', etc.)
    """
    import os
    import torch
    from PIL import Image
    import numpy as np
    import cv2
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy if tensor
    if isinstance(buffer, torch.Tensor):
        buffer = buffer.cpu().numpy()
    
    # Convert to (T, C, H, W) format if needed
    if buffer.shape[0] == 3:  # If in (C, T, H, W) format
        buffer = np.transpose(buffer, (1, 0, 2, 3))
    
    # Define channel order mappings
    channel_orders = {
        'RGB': [0, 1, 2],
        'BGR': [2, 1, 0],
        'GBR': [1, 2, 0],
        'GRB': [1, 0, 2],
        'RBG': [0, 2, 1],
        'BRG': [2, 0, 1]
    }
    
    # Save each frame
    for frame_idx, frame in enumerate(buffer):
        # Convert from (C, H, W) to (H, W, C)
        frame = np.transpose(frame, (1, 2, 0))
        
        # Check value range and normalize if needed
        if frame.max() <= 1.0:
            frame = frame * 255.0
        
        # Clip values to valid range
        frame = np.clip(frame, 0, 255)
        
        # Convert to uint8
        frame = frame.astype(np.uint8)
        
        # Save frames in different channel orders
        for space, order in channel_orders.items():
            reordered_frame = frame[:, :, order]
            save_path = os.path.join(save_dir, 
                                   f'video_{video_idx:04d}_frame_{frame_idx:04d}_{space}.jpg')
            Image.fromarray(reordered_frame).save(save_path, quality=95)
        
        # Save grayscale version
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray_save_path = os.path.join(save_dir, 
                                    f'video_{video_idx:04d}_frame_{frame_idx:04d}_gray.jpg')
        Image.fromarray(gray_frame).save(gray_save_path)
        

class SeqToImagesProcessor:
    """
    Processes video buffer outputs from CelebDFDataSet into individual frames
    using improved image processing pipeline with proper resizing and interpolation.
    Handles both single and multiple sample cases.
    """
    def __init__(self, crop_size=(224, 224), scale=False, mirror=False, pretraining='COCO'):
        """
        Args:
            crop_size (tuple): (height, width) for crop size
            scale (bool): Enable scaling augmentation
            mirror (bool): Enable mirror augmentation
            pretraining (str): Pretraining dataset type
        """
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.is_mirror = mirror
        self.pretraining = pretraining

    def _process_single_buffer(self, buffer):
        """
        Process a single video buffer into processed frames.
        
        Args:
            buffer: Video frames tensor (C x T x H x W)
            
        Returns:
            list: List of processed frame tensors
        """
        processed_frames = []
        
        # Handle different tensor dimensions
        if len(buffer.shape) == 4:  # C x T x H x W
            num_frames = buffer.shape[1]
            for t in range(num_frames):
                frame = buffer[:, t, :, :]  # Get C x H x W frame
                processed_frame = self._process_single_frame(frame)
                processed_frames.append(processed_frame)
        else:  # Handle single frame case
            processed_frames.append(self._process_single_frame(buffer))
            
        return processed_frames

    def _process_single_frame(self, frame):
        """
        Process a single frame using improved resizing pipeline.
        
        Args:
            frame: Input frame tensor (C x H x W)
            
        Returns:
            torch.Tensor: Processed frame tensor
        """
        # Convert to numpy if tensor
        if isinstance(frame, torch.Tensor):
            frame = frame.numpy()
        
        # Ensure frame is in HWC format
        if frame.shape[0] == 3:  # If in CHW format
            frame = np.transpose(frame, (1, 2, 0))

        # Convert BGR to RGB if needed
        if self.pretraining != 'COCO':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to float32
        image = np.asarray(frame, np.float32)
        
        # Apply random scaling if enabled
        if self.scale:
            scale = random.uniform(0.5, 2.0)
            h, w = image.shape[:2]
            scaled_h = int(h * scale)
            scaled_w = int(w * scale)
            image = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        
        # Random mirroring
        if self.is_mirror and random.random() < 0.5:
            image = np.fliplr(image).copy()
        
        # Convert back to CHW format
        image = image.transpose((2, 0, 1))
        image = np.ascontiguousarray(image)
        
        return torch.from_numpy(image).float()

    def process_celebdf_output(self, celebdf_output):
        """
        Process output from CelebDFDataSet, handling both single and multiple sample cases.
        
        Args:
            celebdf_output: Tuple from CelebDFDataset
                For single sample (num_sample=1): (buffer, label, index, {})
                For multiple samples (num_sample>1): (frame_list, label_list, index_list, {})
                
        Returns:
            tuple: For single sample: (processed_frames, label, index, {})
                  For multiple samples: (processed_frames_list, label_list, index_list, {})
        """
        # Check if this is a multiple sample case
        if isinstance(celebdf_output[0], list):
            # Multiple samples case
            frame_list = celebdf_output
            
            # Process each sample's frames
            processed_samples = []
            for frames in frame_list:
                processed_frames = self._process_single_buffer(frames)
                processed_samples.append(processed_frames)
            
            return processed_samples
        else:
            # Single sample case
            buffer = celebdf_output
            processed_frames = self._process_single_buffer(buffer)
            
            return processed_frames
    

class ImagesToSeqProcessor:
    def __init__(self):
        pass

    def process_frames(self, processed_frames):
        """
        Convert processed frames back into the original (buffer, label, index, metadata) format.
        
        Args:
            processed_frames: List of tuples containing (processed_frame, label, index, metadata)
            
        Returns:
            tuple: (buffer, label, index, metadata) where buffer contains all frames in sequence
        """
        if not processed_frames:
            return None, None, None, None
        
        # Extract all frames and create a buffer tensor
        frames = [frame[0] for frame in frames]
        buffer = torch.stack(frames, dim=0)  # Stack along new dimension to create sequence
        
        # Since all frames in a sequence should have the same label, index, and metadata,
        # we can take these values from the first frame
        _, label, index, metadata = processed_frames[0]
        
        return buffer, label, index, metadata
