import numpy as np
import h5py
from PIL import Image
from Final_code.lib.utils import *
import os


# import matplotlib.pyplot as plt


# Read data and some pre-processing
# Data:
# image.shape[num,channel,400,400][num,channel,height,width](float) and has been normalized
# label.shape[num,400,400][num,height,width](int) and has been set to {0,1}
class MakeDataset:
    def __init__(self, opt, records_list, cube_num, dataclass):
        self.datasave = opt.datasave  # C:\Users\ASUS\Desktop\Final_dir\data\pre_process'
        self.filelist = records_list
        DATA_SIZE = opt.data_size.split(',')
        self.datasize = [int(DATA_SIZE[0][1:]), int(DATA_SIZE[1]), int(DATA_SIZE[2][:-1])]
        self.channels = opt.input_nc
        self.dataclass = dataclass
        self.cube_num = cube_num
        # a 3D data
        self.data = np.zeros(self.datasize)
        # processed image size[num,channel,height,width]
        self.image = np.zeros((self.cube_num, self.channels, self.datasize[1], self.datasize[2]))
        # label size[num,height,width]
        self.label = np.zeros((self.cube_num, self.datasize[1], self.datasize[2]))
        self.read_images()

    def read_images(self):
        if not os.path.exists(os.path.join(self.datasave, self.dataclass + "data.hdf5")):
            print(self.dataclass + "picking ...It will take some minutes")
            for modality in self.filelist.keys():  # ct/label
                if modality != 'label':
                    ctlist = list(self.filelist[modality])
                    ct_num = -1
                    for ct in ctlist:  # ct1/ct2/.../ctn
                        ct_num += 1
                        scanlist = list(self.filelist[modality][ct])
                        scan_num = -1
                        for scan in scanlist:  # 1.bmp/2.bmp/.../n.bmp
                            scan_num += 1
                            self.data[:, :, scan_num] = np.array(
                                self.image_transform(scan))
                        # projection and normalization then rotate to the same direction as label
                        tmp = np.sum(self.data, axis=0)
                        xmax, xmin = np.max(tmp), np.min(tmp)
                        tmp = (tmp - xmin) / (xmax - xmin)
                        self.image[ct_num, 0, :, :] = np.rot90(tmp)
                else:
                    ctlist = list(self.filelist[modality])
                    ct_num = -1
                    for ct in ctlist:  # label1/label2/.../labeln
                        ct_num += 1
                        labeladress = self.filelist[modality][ct]
                        self.label[ct_num, :, :] = np.floor(np.array(self.image_transform(labeladress)) / 255).astype(
                            int)
            # histogram equalization
            # self.image = histo_equalized(self.image)
            # for i in range(self.image.shape[0]):
            #     visual_data(self.image[i][0], self.images[i][0])
            f = h5py.File(os.path.join(self.datasave, self.dataclass + "data.hdf5"), "w")
            f.create_dataset('image', data=self.image)
            f.create_dataset('label', data=self.label)
            f.close
        else:
            print("found pickle !!!")
            f = h5py.File(os.path.join(self.datasave, self.dataclass + "data.hdf5"), "r")
            self.image = f['image']
            self.label = f['label']
            f.close

    def image_transform(self, filename):
        image = np.array(Image.open(filename))
        return image
