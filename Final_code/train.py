"""
Code Reference
Image Projection Network 1.2
From 3D to 2D image segmentation
Author 'Mingchao Li, Yerui Chen'
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from Final_code.Data.readData import read_dataset
from Data.MakeDataset import MakeDataset
from Data.MyDataset import MyData
from options.train_options import TrainOptions
from Model import Unet
import Final_code.lib.utils as utils
from Final_code.lib.my_classes import BinaryDiceLoss


# import matplotlib.pyplot as plt

def main(argv=None):
    opt = TrainOptions().parse()
    # path to save model
    model_save_path = os.path.join(opt.saveroot, 'checkpoints')
    best_model_save_path = os.path.join(opt.saveroot, 'best_model')
    utils.check_dir_exist(model_save_path)
    utils.check_dir_exist(best_model_save_path)

    # Get Config
    '''
     I delete the raw data because it's too big
    '''
    # train_data_num = len(os.listdir(os.path.join(opt.dataroot, 'train', 'ct')))
    # val_data_num = len(os.listdir(os.path.join(opt.dataroot, 'val', 'ct')))  # validation cube num
    train_data_num = 160
    val_data_num = 20
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids

    # Read Data
    '''
    I delete the raw data because it's too big
    '''
    # print("Start Setup dataset record")
    # train_records, val_records, _ = read_dataset(opt.dataroot)
    # print("Setting up dataset reader")
    train_records = []
    val_records = []
    get_train = MakeDataset(opt, train_records, train_data_num, 'train')
    get_val = MakeDataset(opt, val_records, val_data_num, 'val')
    train_dataset = MyData(opt, get_train)
    val_dataset = MyData(opt, get_val)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False)
    # check the generated data
    # for i, input in enumerate(val_loader):
    #     image, targets = input
    #     print(image.shape,targets.shape)
    #     plt.subplot(121)
    #     plt.imshow(image[0][0])
    #     plt.subplot(122)
    #     plt.imshow(targets[0])
    #     plt.show()
    #     for j in image:
    #         print(j.shape)

    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Unet(in_chan=opt.input_nc).to(device)

    # loss
    Loss = nn.CrossEntropyLoss()
    IoU = BinaryDiceLoss()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    # lr update
    schedule = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    # restore for the best model
    if opt.useRestore:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(os.path.join(best_model_save_path, 'best_model.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        opt.start_epoch = checkpoint['epoch'] + 1

    # train!
    best = {'epoch': 0, 'Dice': 0.}  # Initialize the best epoch and performance(CrossEntropyLoss)
    trigger = 0  # Early stop counter
    for epoch in range(opt.start_epoch, opt.max_iteration):
        model.train()
        # train
        for i, input in enumerate(train_loader):
            image, targets = input
            image, targets = image.to(device), targets.to(device)
            # image[batch_size,channel, 100, 100],targets[batch_size,100, 100]
            image = image.float()
            outputs = model(image)
            print(outputs.size())

            outputs = torch.squeeze(outputs)
            # (batch,class,crop_height,crop_width)
            targets = torch.squeeze(targets).long()
            # (batch,crop_height,crop_width)

            loss, intersec = IoU(outputs, targets)

            optimizer.zero_grad()
            loss.requires_grad_(True) #for self defined loss_func
            loss.backward()
            optimizer.step()

            # cri = IoU(outputs, targets)
            # print("step", i, "loss is", loss.item(), "IoU is", cri, "lr is",
            #       optimizer.state_dict()['param_groups'][0]['lr'])
            print("step", i, "loss is", loss.item(), "intersection is", intersec, "lr is",
                  optimizer.state_dict()['param_groups'][0]['lr'])
        schedule.step()

        # val
        if epoch % 20 == 0:
            val_log = utils.val(val_loader, model, IoU, device)
            trigger += 1
            if val_log['val_Dice'] > best['Dice']:
                print(val_log['val_Dice'],best['Dice'])
                best['epoch'] = epoch
                best['Dice'] = val_log['val_Dice']
                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                torch.save(state, os.path.join(best_model_save_path, "models{}.pth").format(epoch + 1))
                torch.save(state, os.path.join(best_model_save_path, "best_model.pth"))
                trigger = 0
                print('\033[0;33mSaving best model!\033[0m')

        # save model
        if epoch % opt.save_info_freq == 0:
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, os.path.join(model_save_path, "models{}.pth").format(epoch + 1))

        # early stopping
        if not opt.early_stop is None:
            if trigger >= opt.early_stop:
                print("=> early stopping")
                break
        print(trigger)
        print("--------------epoch:{0}----------------".format(epoch))


if __name__ == '__main__':
    main()
