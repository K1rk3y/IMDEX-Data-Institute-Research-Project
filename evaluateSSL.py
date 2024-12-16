from utilities.metric import ConfusionMatrix
from multiprocessing import Pool
from torch.autograd import Variable
from torch.utils import data
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
from utilities.voc_dataset import VOCDataSet
from utilities.mcfs_dataset import MCFSDataSet
from utilities.loss import CrossEntropy2d


def create_color_map(num_classes):
    """
    Create a color map for visualizing different classes.
    Returns a map where each class gets a unique color.
    """
    color_map = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    color_map[0] = [0, 0, 0]  # Background is black
    return color_map


def get_colored_segmentation_mask(prediction, color_map):
    """
    Convert prediction mask to RGB image using color map.
    """
    prediction = prediction.cpu().numpy() if isinstance(prediction, torch.Tensor) else prediction
    colored_mask = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    
    for class_idx in range(len(color_map)):
        colored_mask[prediction == class_idx] = color_map[class_idx]
    
    return colored_mask


def visualize_segmentation_results(image, prediction, ground_truth=None, num_classes=21, 
                                 class_names=None):
    """
    Visualize segmentation results with class outlines.
    """
    color_map = create_color_map(num_classes)
    pred_colored = get_colored_segmentation_mask(prediction, color_map)
    boundaries = find_boundaries(prediction, mode='outer')
    
    if ground_truth is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    if len(image.shape) == 2:
        ax1.imshow(image, cmap='gray')
    else:
        ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(pred_colored)
    ax2.imshow(boundaries, cmap='gray', alpha=0.3)
    ax2.set_title('Prediction with Boundaries')
    ax2.axis('off')
    
    if ground_truth is not None:
        gt_colored = get_colored_segmentation_mask(ground_truth, color_map)
        gt_boundaries = find_boundaries(ground_truth, mode='outer')
        ax3.imshow(gt_colored)
        ax3.imshow(gt_boundaries, cmap='gray', alpha=0.3)
        ax3.set_title('Ground Truth')
        ax3.axis('off')
    
    if class_names is not None:
        legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color_map[i]/255) 
                         for i in range(num_classes)]
        fig.legend(legend_elements, class_names, 
                  loc='center right', bbox_to_anchor=(0.98, 0.5))
    
    plt.tight_layout()
    plt.show()


def get_iou(confM, dataset):
    """Calculate IoU and return metrics."""
    aveJ, j_list, M = confM.jaccard()

    if dataset == 'pascal_voc':
        classes = np.array(('background',  # always index 0
                            'aeroplane', 'bicycle', 'bird', 'boat',
                            'bottle', 'bus', 'car', 'cat', 'chair',
                            'cow', 'diningtable', 'dog', 'horse',
                            'motorbike', 'person', 'pottedplant',
                            'sheep', 'sofa', 'train', 'tvmonitor'))
    elif dataset == 'mcfs':
        classes = np.array(('background', 'face', 'nose', 'upper_lip', 'under_lip', 'hair', 'left_eyebrow', 'right_eyebrow', 'right_eye', 'left_eye', 'tongue', 'right_ear', 'left_ear', 'glasses', 'headdress', 'head', 'left_eyelashes', 'right_eyelashes'))

    for i, iou in enumerate(j_list):
        print('class {:2d} {:12} IU {:.4f}'.format(i, classes[i], j_list[i]))

    print('meanIOU: ' + str(aveJ) + '\n')
    return aveJ


def evaluate(model, dataset, data_path, deeplabv2=True, ignore_label=250, save_dir=None, pretraining='COCO', show_visualizations=True):
    """
    Evaluate model performance with optional visualization.
    """
    model.eval()
    if pretraining == 'COCO':
        from utilities.transformsgpu import normalize_bgr as normalize
    else:
        from utilities.transformsgpu import normalize_rgb as normalize

    if dataset == 'pascal_voc':
        num_classes = 21
        test_dataset = VOCDataSet(data_path, split="val", scale=False, mirror=False, pretraining=pretraining)
    elif dataset == 'mcfs':
        num_classes = 18
        test_dataset = MCFSDataSet(data_path, split="val", scale=False, mirror=False, pretraining=pretraining)

    testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
    print('Evaluating, found ' + str(len(testloader)) + ' images.')
    
    confM = ConfusionMatrix(num_classes)
    data_list = []
    total_loss = []

    for index, batch in enumerate(testloader):
        image, label, size, name, _ = batch
        
        with torch.no_grad():
            interp = torch.nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
            output = model(normalize(Variable(image).cuda(), dataset))
            output = interp(output)

            label_cuda = Variable(label.long()).cuda()
            criterion = CrossEntropy2d(ignore_label=ignore_label).cuda()
            loss = criterion(output, label_cuda)
            total_loss.append(loss.item())

            output = output.cpu().data[0].numpy()
            gt = np.asarray(label[0].numpy(), dtype=int)
            prediction = np.asarray(np.argmax(output, axis=0), dtype=int)
            
            # Show visualization if enabled
            if show_visualizations:
                img_np = image[0].numpy().transpose(1, 2, 0)
                img_np = (img_np * 255).astype(np.uint8)
                visualize_segmentation_results(
                    image=img_np,
                    prediction=prediction,
                    ground_truth=gt,
                    num_classes=num_classes
                )

            data_list.append((np.reshape(gt, (-1)), np.reshape(prediction, (-1))))

        if (index + 1) % 100 == 0:
            process_list_evaluation(confM, data_list)
            data_list = []

    process_list_evaluation(confM, data_list)
    mIoU = get_iou(confM, dataset)
    loss = np.mean(total_loss)
    return mIoU, loss


def process_list_evaluation(confM, data_list):
    """Process evaluation data list."""
    if len(data_list) > 0:
        f = confM.generateM
        pool = Pool(4)
        m_list = pool.map(f, data_list)
        pool.close()
        pool.join()
        pool.terminate()

        for m in m_list:
            confM.addM(m)
            