import torch
from tqdm import tqdm
from copy import deepcopy
import torch.nn as nn
import torch.utils.data as data
from Final_code.lib.visualize import *
from Final_code.lib.extract_patches import *
from Final_code.lib.Evaluate import Evaluate
from Final_code.Data.TestDataset import TestDataset
from Final_code.lib.utils import *


# ========================Record the loss result===============================
class AverageMeter(object):
    """Computes and stores the average and current value for calculate average loss"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ========================IoU===============================
class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets, eps=1e-8):
        # get prediction
        #input = torch.argmax(input, dim=1)

        # get batch_size
        N = targets.size()[0]

        # smooth_variant
        smooth = 1

        # reshape
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)

        # intersection
        intersection = torch.mul(input_flat, targets_flat)
        num = torch.add(torch.mul(2, torch.sum(intersection, dim=1)), smooth)
        den = torch.add(torch.add(torch.sum(input_flat, dim=1), torch.sum(targets_flat, dim=1)), 1)
        dice_eff = torch.div(num, den)

        # average loss
        loss = torch.div(torch.sum(torch.sub(1, dice_eff)), N)
        return loss, torch.sum(intersection)


# ========================test===============================
class Test():
    def __init__(self, opt):
        self.opt = opt
        assert (opt.stride_height <= opt.crop_height and opt.stride_width <= opt.crop_width)

        # save path
        self.path_experiment = opt.result

        # data process
        self.patches_imgs_test, self.test_imgs, self.test_labels, self.new_height, self.new_width = get_data_test_overlap(
            opt=self.opt)
        self.img_height = self.test_imgs.shape[2]
        self.img_width = self.test_imgs.shape[3]
        test_set = TestDataset(self.patches_imgs_test)
        self.test_loader = data.DataLoader(test_set, batch_size=opt.batch_size, shuffle=False, num_workers=3)

    # Inference prediction process
    def inference(self, model, device):
        model.eval()
        preds = []
        with torch.no_grad():
            for batch_idx, inputs in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                inputs = inputs.to(device)  # shape[batch_size, channel, height, wigth]
                outputs = model(inputs)  # shape[batch_size, class, height, wigth]
                # outputs = torch.argmax(outputs, dim=1).cpu().numpy()  # shape[batch_size, height, wigth]
                preds.append(outputs.cpu().numpy())
        predictions = np.concatenate(preds, axis=0)  # shape[num_of_crop_picture, height, wigth]
        self.pred_patches = predictions  # shape[num_of_crop_picture, 1, height, wigth]

    # Evaluate and visualize the predicted images
    def evaluate(self):
        self.pred_imgs = recompone_overlap(
            self.pred_patches, self.new_height, self.new_width, self.opt.stride_height, self.opt.stride_width)

        # restore to original dimensions
        self.pred_imgs = self.pred_imgs[:, :, 0:self.img_height, 0:self.img_width]

        # visualize the predict result
        # for i in range(self.pred_imgs.shape[0]):
        #     visual_data(self.pred_imgs[i][0],self.test_labels[i])

        # Compute some criterion of the prediction result
        y_true = self.test_labels.reshape(-1, 1)
        y_pre = deepcopy(self.pred_imgs.reshape(-1, 1))
        y_pre[y_pre >= 0.5] = 1
        y_pre[y_pre < 0.5] = 0
        eval = Evaluate(save_path=self.path_experiment)
        eval.add_batch(y_true, y_pre)
        log = eval.save_all_result(plot_curve=True, save_name="performance.txt")
        # save labels and probs for plot ROC and PR curve when k-fold Cross-validation
        np.save('{}/result.npy'.format(self.path_experiment), np.asarray([y_true, y_pre]))
        return dict_round(log, 6)

    # save segmentation imgs
    def save_segmentation_result(self):
        self.save_img_path = os.path.join(self.path_experiment, 'result_img')
        if not os.path.exists(os.path.join(self.save_img_path)):
            os.makedirs(self.save_img_path)
        for i in range(self.test_imgs.shape[0]):
            total_img = concat_result(self.test_imgs[i], self.pred_imgs[i], self.test_labels[i])
            save_img(total_img, os.path.join(self.save_img_path, "Result_" + str(i) + '.png'))
