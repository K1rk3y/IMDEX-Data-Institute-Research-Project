import os
from torch.utils import model_zoo
import torch
from utilities.celebdf_dataset import CelebDFDataSet


class DLArguments:
    def __init__(self):
        self.num_classes = 18

        # Augmentation parameters
        self.color_jitter = 0.4    # float, default=0.4
        self.num_sample = 1        # int, default=2, Repeated_aug
        self.aa = 'rand-m7-n4-mstd0.5-inc1'  # str, default='rand-m7-n4-mstd0.5-inc1', AutoAugment policy
        self.smoothing = 0.1       # float, default=0.1, Label smoothing
        self.train_interpolation = 'bicubic'  # str, default='bicubic'

        # Random Erase parameters
        self.reprob = 0.25         # float, default=0.25, Random erase prob
        self.remode = 'pixel'      # str, default='pixel', Random erase mode
        self.recount = 1           # int, default=1, Random erase count
        self.resplit = False       # bool, default=False

        self.data_path = 'celebdf_dataset_slim'  # str, default='you_data_path'
        self.test_list_path = self.data_path + '/List_of_testing_videos.txt'
        self.mode = 'train'
        self.clip_len = 16
        self.frame_sample_rate = 2 
        self.crop_size = 512
        self.semantic_loading = True
        self.model = "DeepLab"
        self.version = "2"
        self.deeplabv2 = True
        self.num_workers = 3
        self.pretraining = "COCO"
        self.checkpoint_dir = "checkpoints/Deeplab"
    

args = DLArguments()


def class_pixel_loader(args, dataset, pretraining='COCO'):
    if args.deeplabv2:
        if args.pretraining == 'COCO': # coco and imagenet resnet architectures differ a little, just on how to do the stride
            from utilities.deeplabv2 import Res_Deeplab
        else: # imagenet pretrained (more modern modification)
            from utilities.deeplabv2_imagenet import Res_Deeplab

        # load pretrained parameters
        if args.pretraining == 'COCO':
            saved_state_dict = model_zoo.load_url('http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet101COCO-41f33a49.pth') # COCO pretraining
        else:
            saved_state_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth') # imagenet pretrainning

    else:
        from utilities.deeplabv3 import Res_Deeplab50 as Res_Deeplab
        saved_state_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth') # imagenet pretrainning

    # create network
    model = Res_Deeplab(num_classes=args.num_classes)

    # Copy loaded parameters to model
    new_params = model.state_dict().copy()
    for name, param in new_params.items():
        if name in saved_state_dict and param.size() == saved_state_dict[name].size():
            new_params[name].copy_(saved_state_dict[name])
    model.load_state_dict(new_params)

    checkpoint = torch.load(os.path.join(args.checkpoint_dir, f'best_model.pth'))
    model.load_state_dict(checkpoint['model'])
    model = model.cuda()

    model.eval()
    
    # Set up normalization based on pretraining
    if pretraining == 'COCO':
        from utilities.transformsgpu import normalize_bgr as normalize
    else:
        from utilities.transformsgpu import normalize_rgb as normalize

    # Set up dataset
    if dataset == 'celebdf':
        test_dataset = CelebDFDataSet(root_path=args.data_path, test_list_path=args.test_list_path, mode=args.mode, clip_len=args.clip_len, frame_sample_rate=args.frame_sample_rate, crop_size=args.crop_size, semantic_loading=args.semantic_loading, model=model, normalize=normalize, args=args)

        print(test_dataset[0][0].size())

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
        
    return test_dataset


test_dataset_new = class_pixel_loader(args, "celebdf")
print(len(test_dataset_new))
print(test_dataset_new[0][4][0].keys())
print(test_dataset_new[0][4][1].keys())
print(test_dataset_new[0][4][2].keys())
