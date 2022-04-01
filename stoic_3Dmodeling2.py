# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 15:05:54 2022

@author: Abdul Qayyum1
"""

#%%%%%%%%%%%% stoic2021 dataset
# import argparse

# import pandas as pd
# from sklearn.model_selection import StratifiedKFold

# parser = argparse.ArgumentParser()
# parser.add_argument("--n_folds", default=5, type=int)
# args = parser.parse_args()

# train = pd.read_csv("C:\\Users\\Administrateur\\Desktop\\Desktopmaterial\\challeneges2022\\stoic2021\\to\\destination\\metadata\\reference.csv")

# skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=518)
# oof = []
# targets = []
# target = "probCOVID"

# for fold, (trn_idx, val_idx) in enumerate(
#     skf.split(train, train[target])
# ):
#     train.loc[val_idx, "fold"] = int(fold)


# train.to_csv("C:\\Users\\Administrateur\\Desktop\\Desktopmaterial\\challeneges2022\\stoic2021\\to\\destination\\metadata\\train.csv", index=False)


# data = pd.read_csv("../input/train.csv")
# data=train
# fold=0
# train_df = data[data.fold != fold].reset_index(drop=False)
# val_df = data[data.fold == fold].reset_index(drop=False)

# fold=1
# train_df1 = data[data.fold != fold].reset_index(drop=False)
# val_df1 = data[data.fold == fold].reset_index(drop=False)

#%%
import torch
import os
from torch.utils.data import Dataset
import pandas as pd

##########################################################new dataset loader ######################################

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,4,5,6'
import numpy as np
import SimpleITK as sitk
from typing import Iterable

def clip_and_normalize(np_image: np.ndarray,
                       clip_min: int = -1100,
                       clip_max: int = 300
                       ) -> np.ndarray:
    np_image = np.clip(np_image, clip_min, clip_max)
    np_image = (np_image - clip_min) / (clip_max - clip_min)
    return np_image

def resample(itk_image: sitk.Image,
             new_spacing: Iterable[float],
             outside_val: float = 0
             ) -> sitk.Image:

    shape = itk_image.GetSize()
    spacing = itk_image.GetSpacing()
    output_shape = tuple(int(round(s * os / ns)) for s, os, ns in zip(shape, spacing, new_spacing))
    return sitk.Resample(
        itk_image,
        output_shape,
        sitk.Transform(),
        sitk.sitkLinear,
        itk_image.GetOrigin(),
        new_spacing,
        itk_image.GetDirection(),
        outside_val,
        sitk.sitkFloat32,
    )

def center_crop(np_image: np.ndarray,
                new_shape: Iterable[int],
                outside_val: float = 0
                ) -> np.ndarray:
    output_image = np.full(new_shape, outside_val, np_image.dtype)

    slices = tuple()
    offsets = tuple()
    for it, sh in enumerate(new_shape):
        size = sh // 2
        if it == 0:
            center = np_image.shape[it] - size
        else:
            center = (np_image.shape[it] // 2)
        start = center - size
        stop = center + size + (sh % 2)

        # computing what area of the original image will be in the cropped output
        slce = slice(max(0, start), min(np_image.shape[it], stop))
        slices += (slce,)

        # computing offset to pad if the crop is partly outside of the scan
        offset = slice(-min(0, start), 2 * size - max(0, (start + 2 * size) - np_image.shape[it]))
        offsets += (offset,)

    output_image[offsets] = np_image[slices]

    return output_image


class stoic2021(Dataset) :
    def __init__(self,root,csvpath):
        self.root=root
        self.csvpath=csvpath
        #self.pathd=pd.read_csv(os.path.join(self.csvpath,'reference1.csv'))
        #csvpath
        self.pathd=self.csvpath
        self.patient_id=self.pathd['PatientID']
        
        
        
    def __getitem__(self, index):
        patient_id=self.pathd['PatientID'][index]
        #probcovid=self.pathd['probCOVID'][index]
        #probsvere=self.pathd['probSevere'][index]
        labels=np.array(self.pathd[['probCOVID','probSevere']],dtype=float)[index]
        patient_path=os.path.join(self.root,str(patient_id)+'.mha')
        input_image = sitk.ReadImage(patient_path)
        process_vol=self._preprocess(input_image)
        #process_vol=
        #process_vol=np.expand_dims(process_vol,axis=0)
        
        ### expand channels into 3 diemsnion
        torch_img=torch.from_numpy(process_vol)
        torch_img=torch.unsqueeze(torch_img,axis=0)
        expdim=torch_img.expand(3,process_vol.shape[0],process_vol.shape[1],process_vol.shape[2])
        process_vol=expdim
        return process_vol,labels
        
    
    def _preprocess(self,input_image: sitk.Image,
               new_spacing: Iterable[float] = (1.6, 1.6, 1.6),
               new_shape: Iterable[int] = (240, 240, 240),
               ) -> np.ndarray:
        input_image = resample(input_image, new_spacing=new_spacing)
        input_image = sitk.GetArrayFromImage(input_image)
        input_image = center_crop(input_image, new_shape=new_shape)
        input_image = clip_and_normalize(input_image)
        
        return input_image
    
        
    def __len__(self):
        return len(self.patient_id)

path='/home/imranr/abdulcovid22/covid2022/data/mha'
#path_csv='/content/drive/MyDrive/dataset_3d'

csvpath='/home/imranr/abdulcovid22/train.csv'
data=pd.read_csv(csvpath)
fold=2
train_df = data[data.fold != fold].reset_index(drop=False)
val_df = data[data.fold == fold].reset_index(drop=False)

# pathd=train_df
# patient_id=pathd['PatientID']
# patient_id=pathd['PatientID']

dataset_train=stoic2021(path,train_df)
dataset_valid=stoic2021(path,val_df)
img,p1=dataset_train[9]
print(img.shape)
print(p1)
# #import matplotlib.pyplot as plt
# # def show(im, title):
# #     fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
# #     ax1.imshow(im[im.shape[0] // 2, :, :])
# #     ax2.imshow(im[:, im.shape[1] // 2, :])
# #     ax3.imshow(im[:, :, im.shape[2] // 2])
# #     plt.title(title)
# #     plt.show()

# #show(img, 'processed')
# img,p2=dataset_valid[9]
# print(img.shape)
# print(p2)


from torch.utils.data.dataloader import DataLoader
train_dataloader = DataLoader(dataset_train, batch_size=6,pin_memory=True,num_workers=6,shuffle=True)

valid_dataloader = DataLoader(dataset_valid, batch_size=6,pin_memory=True,num_workers=6,shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import monai
model = monai.networks.nets.DenseNet121(spatial_dims=3, 
                                        in_channels=3, 
                                        out_channels=2)

from sklearn.metrics import roc_auc_score
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
#from sklearn.metrics import roc_auc_score
loss_func = nn.BCEWithLogitsLoss()
optimizer=optim.Adam(model.parameters(),lr=0.0001)
model=nn.DataParallel(model)
model.to(device)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.5, last_epoch=-1, verbose=True)
from collections import OrderedDict
def train(train_dataloader, model, criterion, optimizer, epoch, scheduler):
    #bar = tqdm(dataloader)
    losses_avg, gt_train = [], []
    train_loss, auc_train = [], []
    model.to(device)
    model.train()
    for i,data in enumerate(tqdm(train_dataloader)):
        #counter+=1
        # extract dataset
        imge,label=data
        imge=imge.float()
        label=label.float()
        imge=imge.to(device)
        label=label.to(device)
        # zero_out the gradient
        optimizer.zero_grad()
        output=model(imge)
        loss=criterion(output,label)
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        output=torch.sigmoid(output)
        pred=output.detach().cpu().numpy().astype(float)
        y=label.detach().cpu().numpy()
        pred1 = np.array(pred > 0.5, dtype=float)
        #auc=roc_auc_score(y.flatten(), pred1.flatten())
        #try:
         #   auc=roc_auc_score(y.flatten(), pred1.flatten())
        #except ValueError:
         #   pass
        auc_train.append(pred1)
        gt_train.append(y)
        #bar.set_description(f"loss {np.mean(train_loss):.5f} iou {np.mean(train_iou):.5f}")
    scheduler.step()  # Update learning rate schedule
    losses_avg=np.mean(train_loss)
    #auc_avg=np.mean(auc_train)
    
    log = OrderedDict([('loss', losses_avg),])
    return log,gt_train,auc_train

def validate(valid_dataloader, model, criterion):
    #bar = tqdm(dataloader)
    test_loss, gt_val = [], []
    losses_avg, auc_val = [], []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i,data in enumerate(tqdm(valid_dataloader)):
            #counter+=1
            # extract dataset
            imge,label=data
            imge=imge.float()
            label=label.float()
            imge=imge.to(device)
            label=label.to(device)
            output=model(imge)
            loss=criterion(output,label)
            test_loss.append(loss.item())
            output=torch.sigmoid(output)
            pred=output.detach().cpu().numpy().astype(float)
            y=label.detach().cpu().numpy()
            pred1 = np.array(pred > 0.5, dtype=float)
            #auc_va=roc_auc_score(y.flatten(), pred1.flatten())
            #try:
             #   auc_va=roc_auc_score(y.flatten(), pred1.flatten())
            #except ValueError:
              #  pass
            #auc_val.append(auc_va)
            auc_val.append(pred1)
            gt_val.append(y)
            #bar.set_description(f"test_loss {np.mean(test_loss):.5f} test_iou {np.mean(test_iou):.5f}")
    losses_avg=np.mean(test_loss)
    #auc_avg=np.mean(auc_val)
    log = OrderedDict([('loss', losses_avg),])
    
    return log,gt_val,auc_val

criterion = torch.nn.BCEWithLogitsLoss()
log = pd.DataFrame(index=[], columns=['epoch', 'lr', 'loss', 'auc', 'val_loss', 'val_auc'])
early_stop=20
epochs=10000
best_auc = 0
lr=0.0001
name='3DDensUnetf2'
trigger = 0
for epoch in range(epochs):
    print('Epoch [%d/%d]' %(epoch, epochs))
    # train for one epoch
    train_log,gt_train,auc_train = train(train_dataloader, model, criterion, optimizer, epoch,scheduler)
    auc_train_epoch=roc_auc_score(np.array(gt_train).flatten(), np.array(auc_train).flatten())
    
    val_log,gt_val,auc_val =validate(valid_dataloader, model, criterion)
    auc_valid_epoch=roc_auc_score(np.array(gt_val).flatten(), np.array(auc_val).flatten())
    
    print('loss %.4f - auc %.4f - val_loss %.4f - val_auc %.4f'%(train_log['loss'],auc_train_epoch , 
                                                                 val_log['loss'], auc_valid_epoch))

    tmp = pd.Series([epoch,lr,train_log['loss'],auc_train_epoch,val_log['loss'],auc_valid_epoch], 
                    index=['epoch', 'lr', 'loss', 'auc', 'val_loss', 'val_auc'])

    log = log.append(tmp, ignore_index=True)
    log.to_csv('models/%s/log_3dcovid.csv' %name, index=False)

    trigger += 1

    if auc_valid_epoch > best_auc:
        torch.save(model.state_dict(), 'models/%s/model_densnet3d.pth' %name)
        best_auc = auc_valid_epoch
        print("=> saved best model")
        trigger = 0

    # early stopping
    if not early_stop is None:
        if trigger >= early_stop:
            print("=> early stopping")
            break

    torch.cuda.empty_cache()
print("done training")
