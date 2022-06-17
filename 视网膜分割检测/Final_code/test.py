import torch.backends.cudnn as cudnn
import torch
import os
from lib.my_classes import Test
import Model
from lib.utils import setpu_seed
from options.Test_options import TestOptions

setpu_seed(2022)
if __name__ == '__main__':
    # configuration
    opt = TestOptions().parse()

    # best_model path
    save_path = os.path.join(opt.saveroot, 'best_model')

    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model
    model = Model.Unet(in_chan=opt.input_nc).to(device)
    cudnn.benchmark = True

    # Load checkpoint
    print('==> Loading checkpoint...')
    checkpoint = torch.load(os.path.join(save_path, 'best_model.pth'))
    model.load_state_dict(checkpoint['model'])

    # Evaluate our model
    eval = Test(opt)
    eval.inference(model, device)
    print(eval.evaluate())
    eval.save_segmentation_result()
