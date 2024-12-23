import os
import timeit
import datetime
import pickle
import torch.backends.cudnn as cudnn
from torch.utils import data, model_zoo
from torch.autograd import Variable
import torch.nn as nn
import torch
import torch.nn.functional as F
from utilities.loss import CrossEntropy2d
from utilities.loss import CrossEntropyLoss2dPixelWiseWeighted
import math

import multiprocessing as mp
from torch.multiprocessing import Process, Queue
from torch.utils.data import Dataset, DataLoader
import numpy as np
from functools import partial
from tqdm import tqdm

from utilities import transformmasks
from utilities import transformsgpu

from utilities.voc_dataset import VOCDataSet
from utilities.mcfs_dataset import MCFSDataSet
from utilities.celebdf_dataset import CelebDFDataSet

from evaluateSSL import evaluate
from utilities.class_balancing import ClassBalancing
from utilities.feature_memory import *

from utilities.loader import SeqToImagesProcessor, ImagesToSeqProcessor

start = timeit.default_timer()
start_writeable = datetime.datetime.now().strftime('%m-%d_%H-%M')

class Learning_Rate_Object(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate


class Parameters:
    def __init__(self):
        self.model = "DeepLab"
        self.version = "2"
        self.dataset = "mcfs"
        self.deeplabv2 = True

        # Training parameters
        self.batch_size = 5
        self.num_workers = 3
        self.optimizer = "SGD"
        self.momentum = 0.9
        self.num_iterations = 600
        self.learning_rate = 2e-4
        self.lr_schedule = "Poly"
        self.pretraining = "COCO"
        self.lr_power = 0.9
        self.weight_decay = 5e-4
        self.use_teacher = True

        # Data parameters
        self.split_id = None
        self.labeled_samples = 50
        self.input_size = (512,512)
        self.path = "mcfs_dataset"
        self.num_classes = 18

        # Miscellaneous
        self.random_seed = 5555
        self.ignore_label = 250

        # Utility parameters
        self.save_checkpoint_every = 200
        self.checkpoint_dir = "checkpoints/Deeplab"
        self.val_per_iter = 1000
        self.save_best_model = True
        self.save_teacher = True


para = Parameters()


def contrastive_class_to_class_learned_memory(model, features, class_labels, num_classes, memory):
    """

    Args:
        model: segmentation model that contains the self-attention MLPs for selecting the features
        to take part in the contrastive learning optimization
        features: Nx256  feature vectors for the contrastive learning (after applying the projection and prediction head)
        class_labels: N corresponding class labels for every feature vector
        num_classes: number of classesin the dataet
        memory: memory bank [List]

    Returns:
        returns the contrastive loss between features vectors from [features] and from [memory] in a class-wise fashion.
    """

    loss = 0

    for c in range(num_classes):
        # get features of an specific class
        mask_c = class_labels == c
        features_c = features[mask_c,:]
        memory_c = memory[c] # N, 256

        # get the self-attention MLPs both for memory features vectors (projected vectors) and network feature vectors (predicted vectors)
        selector = model.__getattr__('contrastive_class_selector_' + str(c))
        selector_memory = model.__getattr__('contrastive_class_selector_memory' + str(c))

        if memory_c is not None and features_c.shape[0] > 1 and memory_c.shape[0] > 1:

            memory_c = torch.from_numpy(memory_c).cuda()

            # L2 normalize vectors
            memory_c = F.normalize(memory_c, dim=1) # N, 256
            features_c_norm = F.normalize(features_c, dim=1) # M, 256

            # compute similarity. All elements with all elements
            similarities = torch.mm(features_c_norm, memory_c.transpose(1, 0))  # MxN
            distances = 1 - similarities # values between [0, 2] where 0 means same vectors
            # M (elements), N (memory)


            # now weight every sample

            learned_weights_features = selector(features_c.detach()) # detach for trainability
            learned_weights_features_memory = selector_memory(memory_c)

            # self-atention in the memory featuers-axis and on the learning contrsative featuers-axis
            learned_weights_features = torch.sigmoid(learned_weights_features)
            rescaled_weights = (learned_weights_features.shape[0] / learned_weights_features.sum(dim=0)) * learned_weights_features
            rescaled_weights = rescaled_weights.repeat(1, distances.shape[1])
            distances = distances * rescaled_weights


            learned_weights_features_memory = torch.sigmoid(learned_weights_features_memory)
            learned_weights_features_memory = learned_weights_features_memory.permute(1, 0)
            rescaled_weights_memory = (learned_weights_features_memory.shape[0] / learned_weights_features_memory.sum(dim=0)) * learned_weights_features_memory
            rescaled_weights_memory = rescaled_weights_memory.repeat(distances.shape[0], 1)
            distances = distances * rescaled_weights_memory


            loss = loss + distances.mean()

    return loss / num_classes


def entropy_loss(v, mask):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    n, c, h, w = v.size()

    loss_image = torch.mul(v, torch.log2(v + 1e-30))
    loss_image = torch.sum(loss_image, dim=1)
    loss_image = mask.float().squeeze(1) * loss_image


    percentage_valid_points = torch.mean(mask.float())

    return -torch.sum(loss_image) / (n * h * w * np.log2(c) * percentage_valid_points)

def lr_poly(base_lr, iter, max_iter, power):
    """

    Args:
        base_lr: initial learning rate
        iter: current iteration
        max_iter: maximum number of iterations
        power: power value for polynomial decay

    Returns: the updated learning rate with polynomial decay

    """

    return base_lr * ((1 - float(iter) / float(max_iter)) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    """

    Args:
        optimizer: pytorch optimizer
        i_iter: current iteration

    Returns: sets learning rate with poliynomial decay

    """
    lr = lr_poly(para.learning_rate, i_iter, para.num_iterations, para.lr_power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr


def sigmoid_ramp_up(iter, max_iter):
    """

    Args:
        iter: current iteration
        max_iter: maximum number of iterations to perform the rampup

    Returns:
        returns 1 if iter >= max_iter
        returns [0,1] incrementally from 0 to max_iters if iter < max_iter

    """
    if iter >= max_iter:
        return 1
    else:
        return np.exp(- 5 * (1 - float(iter) / float(max_iter)) ** 2)

def update_BN_weak_unlabeled_data(model, norm_func, batch_size, loader, iters=1000):
    iterator = iter(loader)
    model.train()
    for _ in range(iters):
        ''' UNLABELED SAMPLES '''
        try:
            batch = next(iterator)
            if batch[0].shape[0] != batch_size:
                batch = next(iterator)
        except:
            iterator = iter(loader)
            batch = next(iterator)

        # Unlabeled
        unlabeled_images, _, _, _, _ = batch
        unlabeled_images = unlabeled_images.cuda()

        # Create pseudolabels
        _, _ = model(norm_func(unlabeled_images, para.dataset), return_features=True)

    return model

def augmentationTransform(parameters, data=None, target=None, probs=None, jitter_vale=0.4, min_sigma=0.2, max_sigma=2., ignore_label=255):
    """

    Args:
        parameters: dictionary with the augmentation configuration
        data: BxCxWxH input data to augment
        target: BxWxH labels to augment
        probs: BxWxH probability map to augment
        jitter_vale:  jitter augmentation value
        min_sigma: min sigma value for blur
        max_sigma: max sigma value for blur
        ignore_label: value for ignore class

    Returns:
            augmented data, target, probs
    """
    assert ((data is not None) or (target is not None))
    if "Mix" in parameters:
        data, target, probs = transformsgpu.mix(mask=parameters["Mix"], data=data, target=target, probs=probs)

    if "RandomScaleCrop" in parameters:
        data, target, probs = transformsgpu.random_scale_crop(scale=parameters["RandomScaleCrop"], data=data,
                                                              target=target, probs=probs, ignore_label=ignore_label)
    if "flip" in parameters:
        data, target, probs = transformsgpu.flip(flip=parameters["flip"], data=data, target=target, probs=probs)

    if "ColorJitter" in parameters:
        data, target, probs = transformsgpu.colorJitter(colorJitter=parameters["ColorJitter"], data=data, target=target,
                                                        probs=probs, s=jitter_vale)
    if "GaussianBlur" in parameters:
        data, target, probs = transformsgpu.gaussian_blur(blur=parameters["GaussianBlur"], data=data, target=target,
                                                          probs=probs, min_sigma=min_sigma, max_sigma=max_sigma)

    if "Grayscale" in parameters:
        data, target, probs = transformsgpu.grayscale(grayscale=parameters["Grayscale"], data=data, target=target,
                                                      probs=probs)
    if "Solarize" in parameters:
        data, target, probs = transformsgpu.solarize(solarize=parameters["Solarize"], data=data, target=target,
                                                     probs=probs)

    return data, target, probs


def _save_checkpoint(iteration, model, optimizer, save_best=False, overwrite=True):
    """
    Saves the current checkpoint

    Args:
        iteration: current iteration [int]
        model: segmentation model
        optimizer: pytorch optimizer
        config: configuration
        save_best: Boolean: whether to sae only if best metric
        overwrite: whether to overwrite if ther is an existing checkpoint

    Returns:

    """
    checkpoint = {
        'iteration': iteration,
        'optimizer': optimizer.state_dict(),
    }
    checkpoint['model'] = model.state_dict()

    if save_best:
        filename = os.path.join(para.checkpoint_dir, f'best_model.pth')
        torch.save(checkpoint, filename)
        print(f'\nSaving a checkpoint: {filename} ...')
        print("Saving current best model: best_model.pth")
    else:
        filename = os.path.join(para.checkpoint_dir, f'checkpoint-iter{iteration}.pth')
        print(f'\nSaving a checkpoint: {filename} ...')
        torch.save(checkpoint, filename)
        if overwrite:
            try:
                os.remove(os.path.join(para.checkpoint_dir, f'checkpoint-iter{iteration - para.save_checkpoint_every}.pth'))
            except:
                pass


def create_ema_model(model, net_class):
    """

    Args:
        model: segmentation model to copy parameters from
        net_class: segmentation model class

    Returns: Segmentation model from [net_class] with same parameters than [model]

    """
    ema_model = net_class(num_classes=para.num_classes)

    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()

    return ema_model

def update_ema_variables(ema_model, model, alpha_teacher, iteration):
    """

    Args:
        ema_model: model to update
        model: model from which to update parameters
        alpha_teacher: value for weighting the ema_model
        iteration: current iteration

    Returns: ema_model, with parameters updated follwoing the exponential moving average of [model]

    """
    # Use the "true" average until the exponential average is more correct
    alpha_teacher = min(1 - 1 / (iteration*10 + 1), alpha_teacher)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]

    return ema_model

def augment_samples(images, labels, probs, do_classmix, batch_size, ignore_label, weak = False):
    """
    Perform data augmentation

    Args:
        images: BxCxWxH images to augment
        labels:  BxWxH labels to augment
        probs:  BxWxH probability maps to augment
        do_classmix: whether to apply classmix augmentation
        batch_size: batch size
        ignore_label: ignore class value
        weak: whether to perform weak or strong augmentation

    Returns:
        augmented data, augmented labels, augmented probs

    """

    if do_classmix:
        # ClassMix: Get mask for image A
        for image_i in range(batch_size):  # for each image
            classes = torch.unique(labels[image_i])  # get unique classes in pseudolabel A
            nclasses = classes.shape[0]

            # remove ignore class
            if ignore_label in classes and len(classes) > 1 and nclasses > 1:
                classes = classes[classes != ignore_label]
                nclasses = nclasses - 1

            if para.dataset == 'pascal_voc':  # if voc dataaset, remove class 0, background
                if 0 in classes and len(classes) > 1 and nclasses > 1:
                    classes = classes[classes != 0]
                    nclasses = nclasses - 1

            # pick half of the classes randomly
            classes = (classes[torch.Tensor(
                np.random.choice(nclasses, int(((nclasses - nclasses % 2) / 2) + 1), replace=False)).long()]).cuda()

            # acumulate masks
            if image_i == 0:
                MixMask = transformmasks.generate_class_mask(labels[image_i], classes).unsqueeze(0).cuda()
            else:
                MixMask = torch.cat(
                    (MixMask, transformmasks.generate_class_mask(labels[image_i], classes).unsqueeze(0).cuda()))


        params = {"Mix": MixMask}
    else:
        params = {}

    if weak:
        params["flip"] = random.random() < 0.5
        params["ColorJitter"] = random.random() < 0.2
        params["GaussianBlur"] = random.random() < 0.
        params["Grayscale"] = random.random() < 0.0
        params["Solarize"] = random.random() < 0.0
        if random.random() < 0.5:
            scale = random.uniform(0.75, 1.75)
        else:
            scale = 1
        params["RandomScaleCrop"] = scale

        # Apply strong augmentations to unlabeled images
        image_aug, labels_aug, probs_aug = augmentationTransform(params,
                                                                 data=images, target=labels,
                                                                 probs=probs, jitter_vale=0.125,
                                                                 min_sigma=0.1, max_sigma=1.5,
                                                                 ignore_label=ignore_label)
    else:
        params["flip"] = random.random() < 0.5
        params["ColorJitter"] = random.random() < 0.8
        params["GaussianBlur"] = random.random() < 0.2
        params["Grayscale"] = random.random() < 0.0
        params["Solarize"] = random.random() < 0.0
        if random.random() < 0.80:
            scale = random.uniform(0.75, 1.75)
        else:
            scale = 1
        params["RandomScaleCrop"] = scale

        # Apply strong augmentations to unlabeled images
        image_aug, labels_aug, probs_aug = augmentationTransform(params,
                                                                 data=images, target=labels,
                                                                 probs=probs, jitter_vale=0.25,
                                                                 min_sigma=0.1, max_sigma=1.5,
                                                                 ignore_label=ignore_label)

    return image_aug, labels_aug, probs_aug, params


def main():
    cudnn.enabled = True
    torch.manual_seed(para.random_seed)
    torch.cuda.manual_seed(para.random_seed)
    np.random.seed(para.random_seed)
    random.seed(para.random_seed)
    torch.backends.cudnn.deterministic = True

    if para.pretraining == 'COCO': # depending the pretraining, normalize with bgr or rgb
        from utilities.transformsgpu import normalize_bgr as normalize
    else:
        from utilities.transformsgpu import normalize_rgb as normalize

    batch_size_unlabeled = int(para.batch_size / 2) # because of augmentation anchoring, 2 augmentations per sample
    batch_size_labeled = int(para.batch_size * 1)
    assert batch_size_unlabeled >= 2, "batch size should be higher than 2"
    assert batch_size_labeled >= 2, "batch size should be higher than 2"
    RAMP_UP_ITERS = 2000 # iterations until contrastive and self-training are taken into account

    # DATASETS / LOADERS
    if para.dataset == 'pascal_voc':
        train_dataset = VOCDataSet(para.path, crop_size=para.input_size, scale=False, mirror=False, pretraining=para.pretraining)

    elif para.dataset == 'mcfs':
        train_dataset = MCFSDataSet(para.path, crop_size=para.input_size, scale=False, mirror=False)

    train_dataset_size = len(train_dataset)
    print('dataset size: ', train_dataset_size)

    partial_size = para.labeled_samples
    print('Training on number of samples:', partial_size)

    # class weighting  taken unlabeled data into acount in an incremental fashion.
    class_weights_curr = ClassBalancing(labeled_iters=int(para.labeled_samples / batch_size_labeled),
                                                  unlabeled_iters=int(
                                                      (train_dataset_size - para.labeled_samples) / batch_size_unlabeled),
                                                  n_classes=para.num_classes)
    # Memory Bank
    feature_memory = FeatureMemory(num_samples=para.labeled_samples, dataset=para.dataset, memory_per_class=256, feature_size=256, n_classes=para.num_classes)

    # select the partition
    if para.split_id is not None:
        train_ids = pickle.load(open(para.split_id, 'rb'))
        print('loading train ids from {}'.format(para.split_id))
    else:
        train_ids = np.arange(train_dataset_size)
        np.random.shuffle(train_ids)

    # Samplers for labeled data
    train_sampler = data.sampler.SubsetRandomSampler(train_ids[:partial_size])
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=batch_size_labeled, sampler=train_sampler, num_workers=para.num_workers,
                                  pin_memory=True)
    trainloader_iter = iter(trainloader)

    # Samplers for unlabeled data
    train_remain_sampler = data.sampler.SubsetRandomSampler(train_ids[partial_size:])
    trainloader_remain = data.DataLoader(train_dataset,
                                         batch_size=batch_size_unlabeled, sampler=train_remain_sampler,
                                         num_workers=para.num_workers, pin_memory=True)
    trainloader_remain_iter = iter(trainloader_remain)

    # supervised loss
    supervised_loss = CrossEntropy2d(ignore_label=para.ignore_label).cuda()

    ''' Deeplab model '''
    # Define network
    if para.deeplabv2:
        if para.pretraining == 'COCO': # coco and imagenet resnet architectures differ a little, just on how to do the stride
            from utilities.deeplabv2 import Res_Deeplab
        else: # imagenet pretrained (more modern modification)
            from utilities.deeplabv2_imagenet import Res_Deeplab

        # load pretrained parameters
        if para.pretraining == 'COCO':
            saved_state_dict = model_zoo.load_url('http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet101COCO-41f33a49.pth') # COCO pretraining
        else:
            saved_state_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth') # imagenet pretrainning

    else:
        from utilities.deeplabv3 import Res_Deeplab50 as Res_Deeplab
        saved_state_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth') # imagenet pretrainning

    # create network
    model = Res_Deeplab(num_classes=para.num_classes)

    # Copy loaded parameters to model
    new_params = model.state_dict().copy()
    for name, param in new_params.items():
        if name in saved_state_dict and param.size() == saved_state_dict[name].size():
            new_params[name].copy_(saved_state_dict[name])
    model.load_state_dict(new_params)


    # Optimizer for segmentation network
    learning_rate_object = Learning_Rate_Object(para.learning_rate)

    optimizer = torch.optim.SGD(model.optim_parameters(learning_rate_object),
                          lr=para.learning_rate, momentum=para.momentum, weight_decay=para.weight_decay)

    ema_model = create_ema_model(model, Res_Deeplab)
    ema_model.train()
    ema_model = ema_model.cuda()
    model.train()
    model = model.cuda()
    cudnn.benchmark = True

    # pickle.dump(train_ids, open(os.path.join(para.checkpoint_dir, 'train_split.pkl'), 'wb'))

    interp = nn.Upsample(size=(para.input_size[0], para.input_size[1]), mode='bilinear', align_corners=True)

    epochs_since_start = 0
    start_iteration = 0
    best_mIoU = 0  # best metric while training
    iters_without_improve = 0

    # TRAINING
    for i_iter in range(start_iteration, para.num_iterations):
        model.train()  # set mode to training
        optimizer.zero_grad()

        loss_l_value = 0.
        adjust_learning_rate(optimizer, i_iter)

        ''' LABELED SAMPLES '''
        # Get batch
        try:
            batch = next(trainloader_iter)
            if batch[0].shape[0] != batch_size_labeled:
                batch = next(trainloader_iter)
        except:  # finish epoch, rebuild the iterator
            epochs_since_start = epochs_since_start + 1
            # print('Epochs since start: ',epochs_since_start)
            trainloader_iter = iter(trainloader)
            batch = next(trainloader_iter)

        images, labels, _, _, _ = batch
        images = images.cuda()
        labels = labels.cuda()

        ''' UNLABELED SAMPLES '''
        try:
            batch_remain = next(trainloader_remain_iter)
            if batch_remain[0].shape[0] != batch_size_unlabeled:
                batch_remain = next(trainloader_remain_iter)
        except:
            trainloader_remain_iter = iter(trainloader_remain)
            batch_remain = next(trainloader_remain_iter)

        # Unlabeled
        unlabeled_images, _, _, _, _ = batch_remain
        unlabeled_images = unlabeled_images.cuda()

        # Create pseudolabels
        with torch.no_grad():
            if para.use_teacher:
                logits_u_w, features_weak_unlabeled = ema_model(normalize(unlabeled_images, para.dataset), return_features=True)
            else:
                model.eval()
                logits_u_w, features_weak_unlabeled = model(normalize(unlabeled_images, para.dataset), return_features=True)
                model.train()

            logits_u_w = interp(logits_u_w).detach()  # prediction unlabeled
            softmax_u_w = torch.softmax(logits_u_w, dim=1)
            max_probs, pseudo_label = torch.max(softmax_u_w, dim=1)  # Get pseudolabels

        model.train()

        images_aug, labels_aug, _, _ = augment_samples(images, labels, None, random.random()  < 0.2, batch_size_labeled, para.ignore_label, weak=True)

        '''
        UNLABELED DATA
        '''
        unlabeled_images_aug1, pseudo_label1, max_probs1, unlabeled_aug1_params = augment_samples(unlabeled_images,
                                                                                                  pseudo_label,
                                                                                                  max_probs,
                                                                                                  i_iter > RAMP_UP_ITERS and random.random() < 0.75,
                                                                                                  batch_size_unlabeled,
                                                                                                  para.ignore_label)


        unlabeled_images_aug2, pseudo_label2, max_probs2, unlabeled_aug2_params = augment_samples(unlabeled_images,
                                                                                                  pseudo_label,
                                                                                                  max_probs,
                                                                                                  i_iter > RAMP_UP_ITERS and random.random() < 0.75,
                                                                                                  batch_size_unlabeled,
                                                                                                  para.ignore_label)
        
        target_tensor = pseudo_label1  # replace with your actual target tensor name
        print("pseudo_label1 tensor shape:", target_tensor.shape)
        print("Unique values in pseudo_label1:", torch.unique(target_tensor).tolist())

        # concatenate two augmentations of unlabeled data
        joined_unlabeled = torch.cat((unlabeled_images_aug1, unlabeled_images_aug2), dim=0)
        joined_pseudolabels = torch.cat((pseudo_label1, pseudo_label2), dim=0)
        joined_maxprobs = torch.cat((max_probs1, max_probs2), dim=0)

        pred_joined_unlabeled, features_joined_unlabeled = model(normalize(joined_unlabeled, para.dataset), return_features=True)
        pred_joined_unlabeled = interp(pred_joined_unlabeled)

        # labeled data
        labeled_pred, labeled_features = model(normalize(images_aug, para.dataset), return_features=True)
        labeled_pred = interp(labeled_pred)

        # apply class balance for cityspcaes dataset
        class_weights = torch.from_numpy(np.ones((para.num_classes))).cuda()

        loss = 0

        # SUPERVISED SEGMENTATION
        labeled_loss = supervised_loss(labeled_pred, labels_aug, weight=class_weights.float())
        loss = loss + labeled_loss

        # SELF-SUPERVISED SEGMENTATION
        print("class_weights shape:", class_weights.shape)
        unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted(ignore_index=para.ignore_label, weight=class_weights.float()).cuda() #

        # Pseudo-label weighting
        print("joined_maxprobs shape:", joined_maxprobs.shape)
        print("joined_maxprobs stats:", 
            "min:", joined_maxprobs.min().item(),
            "max:", joined_maxprobs.max().item(),
            "has_nan:", torch.isnan(joined_maxprobs).any().item())
        pixelWiseWeight = sigmoid_ramp_up(i_iter, RAMP_UP_ITERS) * torch.ones(joined_maxprobs.shape).cuda()
        print("pixelWiseWeight shape:", pixelWiseWeight.shape)
        pixelWiseWeight = pixelWiseWeight * torch.pow(joined_maxprobs.detach(), 6)

        # Pseudo-label loss
        target_tensor = joined_pseudolabels  # replace with your actual target tensor name
        print("Target tensor shape:", target_tensor.shape)
        print("Unique values in target:", torch.unique(target_tensor).tolist())
        print("Min target value:", target_tensor.min().item())
        print("Max target value:", target_tensor.max().item())
        loss_ce_unlabeled = unlabeled_loss(pred_joined_unlabeled, joined_pseudolabels, pixelWiseWeight)

        loss = loss + loss_ce_unlabeled

        # entropy loss
        valid_mask = (joined_pseudolabels != para.ignore_label).unsqueeze(1)
        loss = loss + entropy_loss(torch.nn.functional.softmax(pred_joined_unlabeled, dim=1), valid_mask) * 0.01

        # CONTRASTIVE LEARNING
        if i_iter > RAMP_UP_ITERS - 1000:
            # Build Memory Bank 1000 iters before starting to do contrastive

            with torch.no_grad():
                # Get feature vectors from labeled images with EMA model
                if para.use_teacher:
                    labeled_pred_ema, labeled_features_ema = ema_model(normalize(images_aug, para.dataset), return_features=True)
                else:
                    model.eval()
                    labeled_pred_ema, labeled_features_ema = model(normalize(images_aug, para.dataset), return_features=True)
                    model.train()

                labeled_pred_ema = interp(labeled_pred_ema)
                probability_prediction_ema, label_prediction_ema = torch.max(torch.softmax(labeled_pred_ema, dim=1),dim=1)  # Get pseudolabels

            # Resize labels, predictions and probabilities,  to feature map resolution
            labels_down = nn.functional.interpolate(labels_aug.float().unsqueeze(1), size=(labeled_features_ema.shape[2], labeled_features_ema.shape[3]),
                                                    mode='nearest').squeeze(1)
            label_prediction_down = nn.functional.interpolate(label_prediction_ema.float().unsqueeze(1), size=(labeled_features_ema.shape[2], labeled_features_ema.shape[3]),
                                                    mode='nearest').squeeze(1)
            probability_prediction_down = nn.functional.interpolate(probability_prediction_ema.float().unsqueeze(1), size=(labeled_features_ema.shape[2], labeled_features_ema.shape[3]),
                                                    mode='nearest').squeeze(1)


            # get mask where the labeled predictions are correct and have a confidence higher than 0.95
            mask_prediction_correctly = ((label_prediction_down == labels_down).float() * (probability_prediction_down > 0.95).float()).bool()

            # Apply the filter mask to the features and its labels
            labeled_features_correct = labeled_features_ema.permute(0, 2, 3, 1)
            labels_down_correct = labels_down[mask_prediction_correctly]
            labeled_features_correct = labeled_features_correct[mask_prediction_correctly, ...]

            # get projected features
            with torch.no_grad():
                if para.use_teacher:
                    proj_labeled_features_correct = ema_model.projection_head(labeled_features_correct)
                else:
                    model.eval()
                    proj_labeled_features_correct = model.projection_head(labeled_features_correct)
                    model.train()
            # updated memory bank
            feature_memory.add_features_from_sample_learned(ema_model, proj_labeled_features_correct, labels_down_correct, batch_size_labeled)



        if i_iter > RAMP_UP_ITERS:
            '''
            CONTRASTIVE LEARNING ON LABELED DATA. Force features from labeled samples, to be similar to other features from the same class (which also leads to good predictions
            '''
            # mask features that do not have ignore label in the labels (zero-padding because of data augmentation like resize/crop)
            mask_prediction_correctly = (labels_down != para.ignore_label)

            labeled_features_all = labeled_features.permute(0, 2, 3, 1)
            labels_down_all = labels_down[mask_prediction_correctly]
            labeled_features_all = labeled_features_all[mask_prediction_correctly, ...]

            # get predicted features
            proj_labeled_features_all = model.projection_head(labeled_features_all)
            pred_labeled_features_all = model.prediction_head(proj_labeled_features_all)

            # Apply contrastive learning loss
            loss_contr_labeled = contrastive_class_to_class_learned_memory(model, pred_labeled_features_all, labels_down_all,
                                para.num_classes, feature_memory.memory)

            loss = loss + loss_contr_labeled * 0.1


            '''
            CONTRASTIVE LEARNING ON UNLABELED DATA. align unlabeled features to labeled features
            '''
            joined_pseudolabels_down = nn.functional.interpolate(joined_pseudolabels.float().unsqueeze(1),
                                                    size=(features_joined_unlabeled.shape[2], features_joined_unlabeled.shape[3]),
                                                    mode='nearest').squeeze(1)

            # mask features that do not have ignore label in the labels (zero-padding because of data augmentation like resize/crop)
            mask = (joined_pseudolabels_down != para.ignore_label)

            features_joined_unlabeled = features_joined_unlabeled.permute(0, 2, 3, 1)
            features_joined_unlabeled = features_joined_unlabeled[mask, ...]
            joined_pseudolabels_down = joined_pseudolabels_down[mask]

            # get predicted features
            proj_feat_unlabeled = model.projection_head(features_joined_unlabeled)
            pred_feat_unlabeled = model.prediction_head(proj_feat_unlabeled)

            # Apply contrastive learning loss
            loss_contr_unlabeled = contrastive_class_to_class_learned_memory(model, pred_feat_unlabeled, joined_pseudolabels_down,
                                para.num_classes, feature_memory.memory)

            loss = loss + loss_contr_unlabeled * 0.1

        print("STEP: ", i_iter, " LOSS: ", loss)
        loss_l_value += loss.item()

        # optimize
        loss.backward()
        optimizer.step()

        m = 1 - (1 - 0.995) * (math.cos(math.pi * i_iter / para.num_iterations) + 1) / 2
        ema_model = update_ema_variables(ema_model=ema_model, model=model, alpha_teacher=m, iteration=i_iter)


        if i_iter % para.save_checkpoint_every == 0 and i_iter != 0:
            if para.save_teacher:
                _save_checkpoint(i_iter, ema_model, optimizer)
            else:
                _save_checkpoint(i_iter, model, optimizer)

        if i_iter % para.val_per_iter == 0 and i_iter != 0:
            print('iter = {0:6d}/{1:6d}'.format(i_iter, para.num_iterations))

            model.eval()
            mIoU, eval_loss = evaluate(model, para.dataset, para.path, deeplabv2=para.deeplabv2, ignore_label=para.ignore_label, save_dir=para.checkpoint_dir, pretraining=para.pretraining, show_visualizations=False)
            model.train()

            if mIoU > best_mIoU:
                best_mIoU = mIoU
                if para.save_teacher:
                    _save_checkpoint(i_iter, ema_model, optimizer, save_best=True)
                else:
                    _save_checkpoint(i_iter, model, optimizer, save_best=True)
                iters_without_improve = 0
            else:
                iters_without_improve += para.val_per_iter

            '''
            if the performance has not improve in N iterations, try to reload best model to optimize again with a lower LR
            Simulating an iterative training'''
            if iters_without_improve > para.num_iterations/5.:
                print('Re-loading a previous best model')
                checkpoint = torch.load(os.path.join(para.checkpoint_dir, f'best_model.pth'))
                model.load_state_dict(checkpoint['model'])
                ema_model = create_ema_model(model, Res_Deeplab)
                ema_model.train()
                ema_model = ema_model.cuda()
                model.train()
                model = model.cuda()
                iters_without_improve = 0 # reset timer

    _save_checkpoint(para.num_iterations, model, optimizer)

    # FINISH TRAINING, evaluate again
    model.eval()
    mIoU, eval_loss = evaluate(model, para.dataset, para.path, deeplabv2=para.deeplabv2, ignore_label=para.ignore_label, save_dir=para.checkpoint_dir, pretraining=para.pretraining, show_visualizations=False)
    model.train()

    if mIoU > best_mIoU and para.save_best_model:
        best_mIoU = mIoU
        _save_checkpoint(i_iter, model, optimizer, save_best=True)

    # TRY IMPROVING BEST MODEL WITH EMA MODEL OR UPDATING BN STATS

    # Load best model
    checkpoint = torch.load(os.path.join(para.checkpoint_dir, f'best_model.pth'))
    model.load_state_dict(checkpoint['model'])
    model = model.cuda()

    model = update_BN_weak_unlabeled_data(model, normalize, batch_size_unlabeled, trainloader_remain)
    model.eval()
    mIoU, eval_loss = evaluate(model, para.dataset, para.path, deeplabv2=para.deeplabv2, ignore_label=para.ignore_label, save_dir=para.checkpoint_dir, pretraining=para.pretraining, show_visualizations=True)
    if mIoU > best_mIoU and para.save_best_model:
        best_mIoU = mIoU
        _save_checkpoint(i_iter, model, optimizer, save_best=True)

    print('BEST MIOU')
    print(best_mIoU)

    end = timeit.default_timer()
    print('Total time: ' + str(end - start) + ' seconds')


main()
