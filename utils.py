import torch
import logging
import torch.nn as nn
import numpy as np
from skimage import measure
from torch._utils import _accumulate
from torch import randperm
from scipy.ndimage import morphology


def random_split(dataset, lengths, inds=None, israndom=True):
    r"""
    Randomly split a data into non-overlapping new datasets of given lengths.

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input data!")

    if israndom:
        indices = randperm(sum(lengths)).tolist()
        print(indices)
    else:
        indices = inds

    return [torch.utils.data.Subset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    if not logger.handlers:
        fh = logging.FileHandler(filename, "w")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    return logger


def iou_score(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().round().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coeff(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def universal3Dlargestregion(deal):
    """找到3D丈量最大连通域,输出为值为1的mask.deal:输入的3D张量"""
    labels = measure.label(deal, connectivity=3)  # 找白色区域的8连通域，并给予每个连通域标号，connectivity为ndarry的维数，三维数组故为3
    jj = measure.regionprops(labels)  # 这里是取得labels的属性，属性有许多
    save_indexs = []
    num = labels.max()  # 找白色部分的连通域有几个
    print('白色区域数量', num)
    del_array = np.array([0] * (num + 1))
    for k in range(num):  # 这里是找最大的那个白色连通域的标号
        if k == 0:
            initial_area = jj[0].area
            save_index = 1  # 初始保留第一个连通域
            if save_index not in save_indexs:
                save_indexs.append(save_index)
        else:
            k_area = jj[k].area  # 将元组转换成array
            if initial_area < k_area:
                initial_area = k_area
                save_index = k + 1  # python从0开始，而连通域标记是从1开始
                if save_index not in save_indexs:
                    save_indexs.append(save_index)
    print('save_index: ', save_indexs)
    del_array[save_indexs[-1]] = 1
    del_mask = del_array[labels]
    return del_mask


def measureimg(o_img,t_num=1):
    p_img=np.zeros_like(o_img)
    # temp_img=morphology.binary_dilation(o_img.astype("bool"),iterations=2)
    testa1 = measure.label(o_img.astype("bool"))
    props = measure.regionprops(testa1)
    numPix = []
    for ia in range(len(props)):
        numPix += [props[ia].area]
    # print(numPix)
    # 像素最多的连通区域及其指引
    for i in range(0, t_num):
        index = numPix.index(max(numPix)) + 1
        p_img[testa1 == index]=o_img[testa1 == index]
        numPix[index-1]=0
    return p_img


def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord