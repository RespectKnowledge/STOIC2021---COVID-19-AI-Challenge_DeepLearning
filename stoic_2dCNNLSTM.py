# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 15:05:54 2022

@author: Abdul Qayyum
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

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,5,6'
import torch
import os
from torch.utils.data import Dataset
import pandas as pd

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
        process_vol=self._slice_extraction(process_vol)
        torch_img=torch.from_numpy(process_vol)
        torch_img=torch.unsqueeze(torch_img,axis=0)
        torch_img=torch.unsqueeze(torch_img,axis=2)
        #expdim=torch_img.expand(3,process_vol.shape[0],process_vol.shape[1],process_vol.shape[2])
        process_vol=torch_img
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
    def _slice_extraction(self,im):
      lis=[]
      z=9
      for i in range(1,self.slices):
          if i==0:
            #print(i)
            val=im[:,:,i]
          else:
            #print(z)
            val=im[:,:,z]
            z=z+10
          lis.append(val)
      vol=np.array(lis)
      return vol
    
        
    def __len__(self):
        return len(self.patient_id)

path='/home/imranr/abdulcovid22/covid2022/data/mha'
#path_csv='/content/drive/MyDrive/dataset_3d'

csvpath='/home/imranr/abdulcovid22/train.csv'
data=pd.read_csv(csvpath)
fold=3
train_df = data[data.fold != fold].reset_index(drop=False)
val_df = data[data.fold == fold].reset_index(drop=False)


# pathd=train_df
# patient_id=pathd['PatientID']
# patient_id=pathd['PatientID']

#dataset_train=stoic2021(path,train_df)
dataset_train=stoic2021(path,train_df,slices=25)
dataset_valid=stoic2021(path,val_df,slices=25)
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
#import monai
#model = monai.networks.nets.DenseNet121(spatial_dims=3, 
                                        #in_channels=3, 
                                        #out_channels=2)


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

@register_model
def convnext_tiny(pretrained=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_small(pretrained=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

import sys 
import os
import glob
import time
import random
import os
import glob
import time
import random
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 123
seed_everything(seed)

class CFG:
    img_size = 240
    n_frames = 24
    
    cnn_features = 240
    lstm_hidden = 32
    
    n_fold = 5
    n_epochs = 15
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.map = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)
        self.net =convnext_base(pretrained=True)
        #checkpoint = torch.load("../input/efficientnet-pytorch/efficientnet-b0-08094119.pth")
        #self.net.load_state_dict(checkpoint)
        self.net.head = nn.Linear(in_features=1024, out_features=240, bias=True)
    
    def forward(self, x):
        x = F.relu(self.map(x))
        out = self.net(x)
        return out
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cnn = CNN()
        self.rnn = nn.LSTM(CFG.cnn_features, CFG.lstm_hidden, 2, batch_first=True)
        self.fc = nn.Linear(CFG.lstm_hidden, 2, bias=True)

    def forward(self, x):
        # x shape: BxTxCxHxW
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        #print(c_in.shape)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        #print(r_in.shape)
        output, (hn, cn) = self.rnn(r_in)
        
        out = self.fc(hn[-1])
        return out
model = Model()
model=nn.DataParallel(model)
model=model.to(device)

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
    losses_avg, auc_avg = [], []
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
        try:
            auc=roc_auc_score(y.flatten(), pred1.flatten())
        except ValueError:
            pass
        auc_train.append(auc)
        #bar.set_description(f"loss {np.mean(train_loss):.5f} iou {np.mean(train_iou):.5f}")
    scheduler.step()  # Update learning rate schedule
    losses_avg=np.mean(train_loss)
    auc_avg=np.mean(auc_train)
    
    log = OrderedDict([('loss', losses_avg),('auc', auc_avg),])
    return log

def validate(valid_dataloader, model, criterion):
    #bar = tqdm(dataloader)
    test_loss, auc_val = [], []
    losses_avg, auc_avg = [], []
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
            try:
                auc_va=roc_auc_score(y.flatten(), pred1.flatten())
            except ValueError:
                pass
            auc_val.append(auc_va)
            #bar.set_description(f"test_loss {np.mean(test_loss):.5f} test_iou {np.mean(test_iou):.5f}")
    losses_avg=np.mean(test_loss)
    auc_avg=np.mean(auc_val)
    log = OrderedDict([('loss', losses_avg),('auc', auc_avg),])
    
    return log

criterion = torch.nn.BCEWithLogitsLoss()
log = pd.DataFrame(index=[], columns=['epoch', 'lr', 'loss', 'auc', 'val_loss', 'val_auc'])
early_stop=20
epochs=10000
best_auc = 0
lr=0.0001
name='3DDensUnetf3'
trigger = 0
for epoch in range(epochs):
    print('Epoch [%d/%d]' %(epoch, epochs))
    # train for one epoch
    train_log = train(train_dataloader, model, criterion, optimizer, epoch,scheduler)
    #train_log = train(train_loader, model, optimizer, epoch)
    # evaluate on validation set
    #val_log = validate(val_loader, model)
    val_log =validate(valid_dataloader, model, criterion)
    print('loss %.4f - auc %.4f - val_loss %.4f - val_auc %.4f'%(train_log['loss'], train_log['auc'], 
                                                                 val_log['loss'], val_log['auc']))

    tmp = pd.Series([epoch,lr,train_log['loss'],train_log['auc'],val_log['loss'],val_log['auc']], 
                    index=['epoch', 'lr', 'loss', 'auc', 'val_loss', 'val_auc'])

    log = log.append(tmp, ignore_index=True)
    log.to_csv('models/%s/log_3dcovid.csv' %name, index=False)

    trigger += 1

    if val_log['auc'] > best_auc:
        torch.save(model.state_dict(), 'models/%s/model_densnet3d.pth' %name)
        best_auc = val_log['auc']
        print("=> saved best model")
        trigger = 0

    # early stopping
    if not early_stop is None:
        if trigger >= early_stop:
            print("=> early stopping")
            break

    torch.cuda.empty_cache()
print("done training")
