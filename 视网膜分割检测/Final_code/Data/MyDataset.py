import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import albumentations as aug


# image.shape[num,channel,400,400][num,channel,height,width](float) and has been normalized
# label.shape[num,400,400][num,height,width](int) and has been set to {0,1}
# generated data
# image.shape[batchsize,channel,opt.crop_height,opt.crop_width]
# label.shape[batchsize,opt.crop_height,opt.crop_width]
class MyData(Dataset):
    def __init__(self, opt, data):
        super(MyData, self).__init__()
        self.opt = opt
        self.image = data.image  # (num,channel,400, 400)
        self.label = data.label  # (num,400, 400)
        self.channel = opt.input_nc
        DATA_SIZE = opt.data_size.split(',')
        self.datasize = [int(DATA_SIZE[0][1:]), int(DATA_SIZE[1]), int(DATA_SIZE[2][:-1])]
        self.transform = aug.Compose([
            aug.RandomCrop(opt.crop_height, opt.crop_width),
            aug.HorizontalFlip(p=0.5),
            aug.VerticalFlip(p=0.5),
            aug.RandomRotate90(),
        ])

    def __getitem__(self, index):
        image = self.image[index][0]
        annotations = self.label[index]
        aument = self.transform(image=image, mask=annotations)
        image, annotations = aument['image'], aument['mask']
        images = np.zeros((self.channel, image.shape[0], image.shape[1]))
        images[0] = image
        images = torch.from_numpy(images)
        annotations = torch.from_numpy(annotations)
        return images, annotations

    def __len__(self):
        return len(self.image)
