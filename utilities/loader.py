import numpy as np
import random
import torch
import cv2


class SeqToImagesProcessor:
    """
    Processes video buffer outputs from CelebDFDataSet into individual frames
    using MCFSDataSet's image processing pipeline.
    Handles both single and multiple sample cases.
    """
    def __init__(self, crop_size=(321, 321), scale=True, mirror=True, pretraining='COCO'):
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
        Process a single frame using MCFSDataSet's pipeline.
        
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

        # Convert BGR to RGB
        if self.pretraining == 'COCO': # if pratraining is not COCO, change to RGB
            frame = frame
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to float32
        image = np.asarray(frame, np.float32)
        img_h, img_w = image.shape[:2]
        
        # Apply MCFSDataSet's image processing pipeline
        if self.scale:
            scale = random.uniform(0.5, 2.0)
            image = cv2.resize(image, None, fx=scale, fy=scale, 
                             interpolation=cv2.INTER_LINEAR)
            img_h, img_w = image.shape[:2]
        
        # Padding
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, 
                                       cv2.BORDER_CONSTANT, 
                                       value=(0.0, 0.0, 0.0))
        else:
            img_pad = image
            
        img_h, img_w = img_pad.shape[:2]
        
        # Random crop
        h_off = random.randint(0, max(0, img_h - self.crop_h))
        w_off = random.randint(0, max(0, img_w - self.crop_w))
        image = img_pad[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w].copy()
        
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
            frame_list, label_list, index_list, metadata = celebdf_output
            
            # Process each sample's frames
            processed_samples = []
            for frames in frame_list:
                processed_frames = self._process_single_buffer(frames)
                processed_samples.append(processed_frames)
            
            return processed_samples, label_list, index_list, metadata
        else:
            # Single sample case
            buffer, label, index, metadata = celebdf_output
            processed_frames = self._process_single_buffer(buffer)
            
            return processed_frames, label, index, metadata
    

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
