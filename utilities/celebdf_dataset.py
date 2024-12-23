import os
import os
import io
import numpy as np
import torch
from torchvision import transforms
import warnings
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from .random_erasing import RandomErasing
from .volume_transforms import ClipToTensor
from .loader import SeqToImagesProcessor, ImagesToSeqProcessor
from torch.autograd import Variable
from torch.utils import data, model_zoo

from .video_transforms import (
    Compose, Resize, CenterCrop, Normalize,
    create_random_augment, random_short_side_scale_jitter, 
    random_crop, random_resized_crop_with_shift, random_resized_crop,
    horizontal_flip, random_short_side_scale_jitter, uniform_crop, 
)

class CelebDFDataSet(Dataset):
    """Load the Celeb-DF dataset for deepfake detection."""

    def __init__(self, root_path, test_list_path=None, mode='train', clip_len=8,
                 frame_sample_rate=2, crop_size=224, short_side_size=256,
                 new_height=256, new_width=340, keep_aspect_ratio=True,
                 num_segment=1, num_crop=1, test_num_segment=10, test_num_crop=3,
                 semantic_loading=False, model=None, normalize=None, args=None):
        
        self.root_path = root_path
        self.test_list_path = test_list_path
        self.mode = mode
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.args = args
        self.aug = False
        self.rand_erase = False
        self.semantic_loading = semantic_loading

        if self.semantic_loading:
            self.model = model
            self.normalize = normalize
        
        assert num_segment == 1
        if self.mode in ['train']:
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True

        # Load dataset samples
        self.dataset_samples = []
        self.label_array = []
        
        # First, load test files to exclude them from training/validation
        test_videos = set()
        if test_list_path and os.path.exists(test_list_path):
            with open(test_list_path, 'r') as f:
                test_files = f.readlines()
            test_videos = {(file.strip())[2:] for file in test_files}
        
        if mode == 'test' and test_list_path:
            # Load test split from provided list
            for file_path in test_videos:
                self.dataset_samples.append(file_path)
                # Label 1 for real, 0 for fake
                self.label_array.append(1 if 'real' in file_path.lower() else 0)
        else:
            # Temporary lists to store samples before shuffling
            temp_samples = []
            temp_labels = []
            
            # Load train/validation split
            # Real videos from Celeb-real
            celeb_real_path = os.path.join(root_path, 'Celeb-real')
            if os.path.exists(celeb_real_path):
                for video in os.listdir(celeb_real_path):
                    if video.endswith('.mp4'):
                        video_path = os.path.join('Celeb-real', video)
                        if video_path not in test_videos:
                            temp_samples.append(video_path)
                            temp_labels.append(1)  # Real label

            # Fake videos from Celeb-synthesis
            celeb_fake_path = os.path.join(root_path, 'Celeb-synthesis')
            if os.path.exists(celeb_fake_path):
                for video in os.listdir(celeb_fake_path):
                    if video.endswith('.mp4'):
                        video_path = os.path.join('Celeb-synthesis', video)
                        if video_path not in test_videos:
                            temp_samples.append(video_path)
                            temp_labels.append(0)  # Fake label

            # Real videos from YouTube-real
            youtube_real_path = os.path.join(root_path, 'YouTube-real')
            if os.path.exists(youtube_real_path):
                for video in os.listdir(youtube_real_path):
                    if video.endswith('.mp4'):
                        video_path = os.path.join('YouTube-real', video)
                        if video_path not in test_videos:
                            temp_samples.append(video_path)
                            temp_labels.append(1)  # Real label
            
            # Shuffle the dataset
            if len(temp_samples) > 0:
                shuffle_idx = np.random.permutation(len(temp_samples))
                self.dataset_samples = [temp_samples[i] for i in shuffle_idx]
                self.label_array = [temp_labels[i] for i in shuffle_idx]

        # Set up data transformations
        if (mode == 'train'):
            pass  # Transformations will be handled in _aug_frame
        elif (mode == 'validation'):
            self.data_transform = Compose([
                Resize(self.short_side_size, interpolation='bilinear'),
                CenterCrop(size=(self.crop_size, self.crop_size)),
                ClipToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
            ])
        elif mode == 'test':
            self.data_resize = Compose([
                Resize(size=(short_side_size), interpolation='bilinear')
            ])
            self.data_transform = Compose([
                ClipToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
            ])
            # Prepare test segments
            self.test_seg = []
            self.test_dataset = []
            self.test_label_array = []
            for ck in range(self.test_num_segment):
                for cp in range(self.test_num_crop):
                    for idx in range(len(self.label_array)):
                        sample_label = self.label_array[idx]
                        self.test_label_array.append(sample_label)
                        self.test_dataset.append(self.dataset_samples[idx])
                        self.test_seg.append((ck, cp))

    def __getitem__(self, index):
        if self.mode == 'train':
            args = self.args 
            scale_t = 1

            sample = self.dataset_samples[index]
            buffer = self.loadvideo_decord(sample, sample_rate_scale=scale_t)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during training".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.loadvideo_decord(sample, sample_rate_scale=scale_t)

            if args.num_sample > 1:
                frame_list = []
                label_list = []
                index_list = []
                for _ in range(args.num_sample):
                    new_frames = self._aug_frame(buffer, args)
                    label = self.label_array[index]
                    frame_list.append(new_frames)
                    label_list.append(label)
                    index_list.append(index)
                return frame_list, label_list, index_list, {}
            else:
                buffer = self._aug_frame(buffer, args)
         
            if self.semantic_loading:
                sip = SeqToImagesProcessor(crop_size=(args.crop_size, args.crop_size), scale=True, mirror=True, pretraining='COCO')

                processed_samples = sip.process_celebdf_output(buffer)
                    
                testloader = data.DataLoader(processed_samples, batch_size=1, shuffle=False, pin_memory=True)

                mappings = []
                    
                for index, batch in enumerate(testloader):
                    image = batch
                        
                    with torch.no_grad():
                        # Get model prediction
                        interp = torch.nn.Upsample(size=(image.shape[2], image.shape[3]), mode='bilinear', align_corners=True)
                        #print("SHAPE:", image.shape[2], image.shape[3])
                        output = self.model(self.normalize(Variable(image).cuda(), None))
                        output = interp(output)
                        #print("IMAGE SHAPE:", image.size())
                        #print(output.size())
                            
                        # Convert output to numpy array and get class predictions
                        output = output.cpu().data[0].numpy()
                        prediction = np.argmax(output, axis=0)
                            
                        # Create coordinate mapping for each class
                        class_coordinates = {}
                        height, width = prediction.shape
                            
                        # Iterate through the prediction mask to collect coordinates
                        for y in range(height):
                            for x in range(width):
                                class_id = int(prediction[y, x])
                                if class_id not in class_coordinates:
                                    class_coordinates[class_id] = []
                                class_coordinates[class_id].append((x, y))
                            
                        # Store results for this image
                        mappings.append(class_coordinates)
                        
                    if (index + 1) % args.clip_len == 0:
                        print(f'Processed {index + 1} images')

                return buffer, self.label_array[index], index, {}, mappings
            
            return buffer, self.label_array[index], index, {}

        elif self.mode == 'validation':
            sample = self.dataset_samples[index]
            buffer = self.loadvideo_decord(sample)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during validation".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.loadvideo_decord(sample)
            buffer = self.data_transform(buffer)

            if self.semantic_loading:
                sip = SeqToImagesProcessor(crop_size=(self.crop_size, self.crop_size), scale=True, mirror=True, pretraining='COCO')     # SELF CROP INSTEAD OF ARGS CROP

                processed_samples = sip.process_celebdf_output(buffer)
                    
                testloader = data.DataLoader(processed_samples, batch_size=1, shuffle=False, pin_memory=True)

                mappings = []
                    
                for index, batch in enumerate(testloader):
                    image = batch
                        
                    with torch.no_grad():
                        # Get model prediction
                        interp = torch.nn.Upsample(size=(image.shape[2], image.shape[3]), mode='bilinear', align_corners=True)
                        #print("SHAPE:", image.shape[2], image.shape[3])
                        output = self.model(self.normalize(Variable(image).cuda(), None))
                        output = interp(output)
                        #print("IMAGE SHAPE:", image.size())
                        #print(output.size())
                            
                        # Convert output to numpy array and get class predictions
                        output = output.cpu().data[0].numpy()
                        prediction = np.argmax(output, axis=0)
                            
                        # Create coordinate mapping for each class
                        class_coordinates = {}
                        height, width = prediction.shape
                            
                        # Iterate through the prediction mask to collect coordinates
                        for y in range(height):
                            for x in range(width):
                                class_id = int(prediction[y, x])
                                if class_id not in class_coordinates:
                                    class_coordinates[class_id] = []
                                class_coordinates[class_id].append((x, y))
                            
                        # Store results for this image
                        mappings.append(class_coordinates)                     

                return buffer, self.label_array[index], sample.split("/")[-1].split(".")[0], mappings

            return buffer, self.label_array[index], sample.split("/")[-1].split(".")[0]

        elif self.mode == 'test':
            sample = self.test_dataset[index]
            chunk_nb, split_nb = self.test_seg[index]
            buffer = self.loadvideo_decord(sample, chunk_nb=chunk_nb)

            while len(buffer) == 0:
                warnings.warn("video {}, temporal {}, spatial {} not found during testing".format(
                    str(self.test_dataset[index]), chunk_nb, split_nb))
                index = np.random.randint(self.__len__())
                sample = self.test_dataset[index]
                chunk_nb, split_nb = self.test_seg[index]
                buffer = self.loadvideo_decord(sample, chunk_nb=chunk_nb)

            buffer = self.data_resize(buffer)
            if isinstance(buffer, list):
                buffer = np.stack(buffer, 0)

            if self.test_num_crop == 1:
                spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) - self.short_side_size) / 2
                spatial_start = int(spatial_step)
            else:
                spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) - self.short_side_size) \
                                    / (self.test_num_crop - 1)
                spatial_start = int(split_nb * spatial_step)
            if buffer.shape[1] >= buffer.shape[2]:
                buffer = buffer[:, spatial_start:spatial_start + self.short_side_size, :, :]
            else:
                buffer = buffer[:, :, spatial_start:spatial_start + self.short_side_size, :]

            buffer = self.data_transform(buffer)

            if self.semantic_loading:
                sip = SeqToImagesProcessor(crop_size=(self.crop_size, self.crop_size), scale=True, mirror=True, pretraining='COCO')     # SELF CROP INSTEAD OF ARGS CROP

                processed_samples = sip.process_celebdf_output(buffer)
                    
                testloader = data.DataLoader(processed_samples, batch_size=1, shuffle=False, pin_memory=True)

                mappings = []
                    
                for index, batch in enumerate(testloader):
                    image = batch
                        
                    with torch.no_grad():
                        # Get model prediction
                        interp = torch.nn.Upsample(size=(image.shape[2], image.shape[3]), mode='bilinear', align_corners=True)
                        #print("SHAPE:", image.shape[2], image.shape[3])
                        output = self.model(self.normalize(Variable(image).cuda(), None))
                        output = interp(output)
                        #print("IMAGE SHAPE:", image.size())
                        #print(output.size())
                            
                        # Convert output to numpy array and get class predictions
                        output = output.cpu().data[0].numpy()
                        prediction = np.argmax(output, axis=0)
                            
                        # Create coordinate mapping for each class
                        class_coordinates = {}
                        height, width = prediction.shape
                            
                        # Iterate through the prediction mask to collect coordinates
                        for y in range(height):
                            for x in range(width):
                                class_id = int(prediction[y, x])
                                if class_id not in class_coordinates:
                                    class_coordinates[class_id] = []
                                class_coordinates[class_id].append((x, y))
                            
                        # Store results for this image
                        mappings.append(class_coordinates)                     

                return buffer, self.test_label_array[index], sample.split("/")[-1].split(".")[0], \
                        chunk_nb, split_nb, mappings

            return buffer, self.test_label_array[index], sample.split("/")[-1].split(".")[0], \
                   chunk_nb, split_nb

        else:
            raise NameError('mode {} unknown'.format(self.mode))

    def _aug_frame(self, buffer, args):
        """Perform data augmentation on video frames."""
        aug_transform = create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
        )

        buffer = [transforms.ToPILImage()(frame) for frame in buffer]
        buffer = aug_transform(buffer)
        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer)  # T C H W
        buffer = buffer.permute(0, 2, 3, 1)  # T H W C 
        
        buffer = tensor_normalize(
            buffer, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        buffer = buffer.permute(3, 0, 1, 2)  # C T H W

        scl, asp = ([0.08, 1.0], [0.75, 1.3333])
        buffer = spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=self.crop_size,
            random_horizontal_flip=True,
            inverse_uniform_sampling=False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False
        )

        if self.rand_erase:
            erase_transform = RandomErasing(
                args.reprob,
                mode=args.remode,
                max_count=args.recount,
                num_splits=args.recount,
                device="cpu",
            )
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return buffer

    def loadvideo_decord(self, sample, sample_rate_scale=1, chunk_nb=0):
        """Load video content using Decord."""
        fname = os.path.join(self.root_path, sample)

        try:
            if self.keep_aspect_ratio:
                vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
            else:
                vr = VideoReader(fname, width=self.new_width, height=self.new_height,
                               num_threads=1, ctx=cpu(0))

            # Handle temporal segments
            converted_len = int(self.clip_len * self.frame_sample_rate)
            seg_len = len(vr) // self.num_segment

            if self.mode == 'test':
                temporal_step = max(1.0 * (len(vr) - converted_len) / (self.test_num_segment - 1), 0)
                temporal_start = int(chunk_nb * temporal_step)

                bound = min(temporal_start + converted_len, len(vr))
                all_index = [x for x in range(temporal_start, bound, self.frame_sample_rate)]
                while len(all_index) < self.clip_len:
                    all_index.append(all_index[-1])
                vr.seek(0)
                buffer = vr.get_batch(all_index).asnumpy()
                return buffer

            all_index = []
            for i in range(self.num_segment):
                if seg_len <= converted_len:
                    index = np.linspace(0, seg_len, num=seg_len // self.frame_sample_rate)
                    index = np.concatenate((index, np.ones(self.clip_len - seg_len // self.frame_sample_rate) * seg_len))
                    index = np.clip(index, 0, seg_len - 1).astype(np.int64)
                else:
                    if self.mode == 'validation':
                        end_idx = (seg_len - converted_len) // 2
                    else:
                        end_idx = np.random.randint(converted_len, seg_len)
                    str_idx = end_idx - converted_len
                    index = np.linspace(str_idx, end_idx, num=self.clip_len)
                    index = np.clip(index, str_idx, end_idx - 1).astype(np.int64)
                index = index + i*seg_len
                all_index.extend(list(index))

            all_index = all_index[::int(sample_rate_scale)]
            vr.seek(0)
            buffer = vr.get_batch(all_index).asnumpy()
            return buffer
        except:
            print("video cannot be loaded by decord: ", fname)
            return []

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)
        

def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, _ = random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = random_crop(frames, crop_size)
        else:
            transform_func = (
                random_resized_crop_with_shift
                if motion_shift
                else random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames, _ = horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = uniform_crop(frames, crop_size, spatial_idx)
    return frames


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


"""class Parameters:
    def __init__(self):
        self.reprob = 0.0
        self.num_sample = 1
        self.aa = 'rand'
        self.train_interpolation = 'bilinear'
        self.remode = 'rand'

args = Parameters()

dataset = CelebDFDataSet(
        root_path="celebdf_dataset",
        mode="train",
        clip_len=16,
        frame_sample_rate=2,
        crop_size=224,
        args=args
    )
    
print(dataset[1])"""