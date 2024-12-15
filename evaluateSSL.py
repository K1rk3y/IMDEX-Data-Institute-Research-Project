'''
Code taken from https://github.com/WilhelmT/ClassMix
Slightly modified
'''

from utilities.metric import ConfusionMatrix
from multiprocessing import Pool

from torch.autograd import Variable
from torch.utils import data
import torch
import numpy as np
from utilities.voc_dataset import VOCDataSet
from utilities.mcfs_dataset import MCFSDataSet
from utilities.loss import CrossEntropy2d


def get_iou(confM, dataset):
    aveJ, j_list, M = confM.jaccard()

    if dataset == 'pascal_voc':
        classes = np.array(('background',  # always index 0
                            'aeroplane', 'bicycle', 'bird', 'boat',
                            'bottle', 'bus', 'car', 'cat', 'chair',
                            'cow', 'diningtable', 'dog', 'horse',
                            'motorbike', 'person', 'pottedplant',
                            'sheep', 'sofa', 'train', 'tvmonitor'))
    elif dataset == 'cityscapes':
        classes = np.array(("road", "sidewalk",
                            "building", "wall", "fence", "pole",
                            "traffic_light", "traffic_sign", "vegetation",
                            "terrain", "sky", "person", "rider",
                            "car", "truck", "bus",
                            "train", "motorcycle", "bicycle"))

    for i, iou in enumerate(j_list):
        print('class {:2d} {:12} IU {:.4f}'.format(i, classes[i], j_list[i]))

    print('meanIOU: ' + str(aveJ) + '\n')

    return aveJ


def evaluate(model, dataset, data_path, deeplabv2=True, ignore_label=250, save_dir=None, pretraining='COCO'):
    model.eval()
    if pretraining == 'COCO':
        from utilities.transformsgpu import normalize_bgr as normalize
    else:
        from utilities.transformsgpu import normalize_rgb as normalize

    if dataset == 'pascal_voc':
        num_classes = 21
        test_dataset = VOCDataSet(data_path, split="val", scale=False, mirror=False, pretraining=pretraining)
        testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

    elif dataset == 'mcfs':
        num_classes = 17
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
            gt = np.asarray(label[0].numpy(), dtype=np.int)

            output = np.asarray(np.argmax(output, axis=0), dtype=np.int)
            data_list.append((np.reshape(gt, (-1)), np.reshape(output, (-1))))

            # filename = 'output_images/' + name[0].split('/')[-1]
            # cv2.imwrite(filename, output)

        if (index + 1) % 100 == 0:
            # print('%d processed' % (index + 1))
            process_list_evaluation(confM, data_list)
            data_list = []

    process_list_evaluation(confM, data_list)

    mIoU = get_iou(confM, dataset)
    loss = np.mean(total_loss)
    return mIoU, loss


def process_list_evaluation(confM, data_list):
    if len(data_list) > 0:
        f = confM.generateM
        pool = Pool(4)
        m_list = pool.map(f, data_list)
        pool.close()
        pool.join()
        pool.terminate()

        for m in m_list:
            confM.addM(m)
