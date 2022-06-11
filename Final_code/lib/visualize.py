from PIL import Image
import numpy as np
from copy import deepcopy


# Prediction result splicing (original img, predicted probability, binary img, groundtruth)
def concat_result(ori_img, pred_res, gt):
    ori_img = np.transpose(ori_img, (1, 2, 0))
    pred_res = np.transpose(pred_res, (1, 2, 0))
    gt = np.expand_dims(gt, axis=0)
    gt = np.transpose(gt, (1, 2, 0))

    binary = deepcopy(pred_res)
    binary[binary >= 0.5] = 1
    binary[binary < 0.5] = 0

    if ori_img.shape[2] == 3:
        pred_res = np.repeat((pred_res * 255).astype(np.uint8), repeats=3, axis=2)
        binary = np.repeat((binary * 255).astype(np.uint8), repeats=3, axis=2)
        gt = np.repeat((gt * 255).astype(np.uint8), repeats=3, axis=2)
    total_img = np.concatenate((ori_img * 255, pred_res * 255, binary * 255, gt * 255), axis=1)
    return total_img


# visualize image, save as PIL image
def save_img(data, filename):
    assert (len(data.shape) == 3)  # height*width*channels
    if data.shape[2] == 1:  # in case it is black and white
        data = np.reshape(data, (data.shape[0], data.shape[1]))
    img = Image.fromarray(data.astype(np.uint8))  # the image is between 0-1
    img.save(filename)
    return img
