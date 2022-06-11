import os
import natsort

#Produce data path dictionary
def read_dataset(data_dir):
    # data_dir:r'C:\Users\ASUS\Desktop\Final_dir\data\raw_dataset\train'
    datasetlist = {'train': {}, 'val': {}, 'test': {}}
    datalist = list(datasetlist.keys())
    for dlist in datalist:  # train/val/test
        modalitylist = os.listdir(os.path.join(data_dir, dlist))
        modalitylist = natsort.natsorted(modalitylist)
        for modal in modalitylist:  # ct/labels
            datasetlist[dlist].update({modal: {}})
            if modal != 'label':
                ctlist = os.listdir(os.path.join(data_dir, dlist, modal))
                ctlist = natsort.natsorted(ctlist)
                for ct in ctlist:  # ct1/ct2/ct3/.../ctn
                    datasetlist[dlist][modal].update({ct: {}})
                    scanlist = os.listdir(os.path.join(data_dir, dlist, modal, ct))
                    scanlist = natsort.natsorted(scanlist)
                    for i in range(0, len(scanlist)):  # 1.bmp/2.bmp/.../n.bmp
                        scanlist[i] = os.path.join(data_dir, dlist, modal, ct, scanlist[i])
                    datasetlist[dlist][modal][ct] = scanlist
            else:
                ctlist = os.listdir(os.path.join(data_dir, dlist, modal))
                ctlist = natsort.natsorted(ctlist)
                for ct in ctlist:  # label1/label2/label3/.../labeln
                    datasetlist[dlist][modal].update({ct: {}})
                    labeladdress = os.path.join(data_dir, dlist, modal, ct)
                    datasetlist[dlist][modal][ct] = labeladdress
    train_records = datasetlist['train']
    val_records = datasetlist['val']
    test_records = datasetlist['test']
    return train_records, val_records, test_records
# train_records, val_records = read_dataset(r'C:\Users\ASUS\Desktop\Final_dir\data\raw_dataset\train')
