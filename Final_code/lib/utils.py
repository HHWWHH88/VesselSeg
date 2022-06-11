import os
import cv2
import torch
import random
import numpy as np
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict

from Final_code.lib.Evaluate import Evaluate
from Final_code.lib.my_classes import AverageMeter


# ========================val===============================
def val(val_loader, model, criterion, device):
    model.eval()
    val_loss = AverageMeter()
    evaluater = Evaluate()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(val_loader), total=len(val_loader)):
            inputs, targets = inputs.to(device), targets.to(device)

            inputs = inputs.float()
            outputs = model(inputs)

            outputs = torch.squeeze(outputs)
            targets = torch.squeeze(targets).long()

            loss,_ = criterion(outputs, targets)
            val_loss.update(loss.item(), inputs.size(0))

            outputs = outputs.data.cpu().numpy()
            targets = targets.data.cpu().numpy()
            evaluater.add_batch(targets, outputs)
            # evaluater.add_batch(targets, np.argmax(outputs,axis=1))
    log = OrderedDict([('val_loss', val_loss.avg),
                       ('val_Dice', evaluater.Dice()),
                       ('val_BAC',evaluater.bac())])
    return log


# =======================Round off================================
def dict_round(dic, num):
    for key, value in dic.items():
        dic[key] = round(value, num)
    return dic


# =======================Seed for repeatability================================
def setpu_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)


# =======================create directories================================
def check_dir_exist(dir):
    if os.path.exists(dir):
        return
    else:
        names = os.path.split(dir)
        dir = ''
        for name in names:
            dir = os.path.join(dir, name)
            if not os.path.exists(dir):
                try:
                    os.mkdir(dir)
                except:
                    pass
        print('dir', '\'' + dir + '\'', 'is created.')


# ========================draw to check data===============================
def visual_data(image, label):
    assert (len(image.shape) == 2 and len(label.shape) == 2)
    plt.subplot(121)
    plt.imshow(image)
    plt.subplot(122)
    plt.imshow(label)
    plt.show()


# ========================histogram equalization===============================
def histo_equalized(imgs):
    assert (len(imgs.shape) == 4)  # 2D arrays
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i][0] = cv2.equalizeHist(np.round(255 * imgs[i][0]).astype(np.uint8))
    return imgs_equalized.astype(float) / 255


# ========================for IPN===============================
def cal_downsampling_size_combine(size, num):
    th = size * 11 // 10  # range of product of pooling size
    seq = cal_single_ds_c(size, num, th)
    if not seq:
        print('Cal failed! Please redefine num!')
    seq.reverse()
    return seq


def cal_single_ds_c(maximum, num, th, ds_c=[], index=1):
    # maximum:size, num:stride_num, th:the range deviate from size, ds_c:seq, index:the current product
    if index >= maximum:
        if index > th or len(ds_c) != num:
            return
        else:
            final_ds_c = copy.deepcopy(ds_c)
            return final_ds_c

    if len(ds_c) > num:
        return

    if ds_c:
        start = ds_c[-1]
    else:
        start = 2

    final_ds_c = []
    # max size 10000
    for i in range(start, 101):
        ds_c.append(i)
        index *= i
        temp = cal_single_ds_c(maximum, num, th, ds_c, index)
        if temp:
            if final_ds_c:
                if cal_product(temp) < cal_product(final_ds_c):
                    final_ds_c = temp
                if cal_product(temp) == cal_product(final_ds_c):
                    if np.sum(np.array(temp)) < np.sum(final_ds_c):
                        final_ds_c = temp
            else:
                final_ds_c = temp
        ds_c.pop(-1)
        index /= i

    return final_ds_c


def cal_product(list):
    index = 1
    for l in list:
        index *= l
    return index
