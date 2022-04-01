# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 14:28:59 2022

@author: Administrateur
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 11:51:50 2022

@author: Abdul Qayyum
"""

from typing import Dict
from pathlib import Path
import SimpleITK
import torch
import torch.nn as nn

import monai
#model = monai.networks.nets.DenseNet121(spatial_dims=3, 
                                        #in_channels=3, 
                                        #out_channels=2)

#model = monai.networks.nets.resnet10(spatial_dims=3, n_input_channels=1, n_classes=2)

#import monai
# model = monai.networks.nets.DenseNet121(spatial_dims=3, 
#                                         in_channels=3, 
#                                         out_channels=2)

# model1 = monai.networks.nets.DenseNet121(spatial_dims=3, 
#                                         in_channels=3, 
#                                         out_channels=2)

# model2 = monai.networks.nets.DenseNet121(spatial_dims=3, 
#                                         in_channels=3, 
#                                         out_channels=2)
# model3 = monai.networks.nets.DenseNet121(spatial_dims=3, 
#                                         in_channels=3, 
#                                         out_channels=2)
# def get_model_DesnNet201fold0(PATH):
#   model = monai.networks.nets.DenseNet121(spatial_dims=3, 
#                                         in_channels=3, 
#                                         out_channels=2)
#   model =nn.DataParallel(model)
#   model.to(device)
#   model.load_state_dict(torch.load(PATH, map_location=torch.device(device)))
#             #self.model =nn.DataParallel(smodel)
#   model.eval()
#   return model

# def get_model_DesnNet201fold1(PATH):
#   model1 = monai.networks.nets.DenseNet121(spatial_dims=3, 
#                                         in_channels=3, 
#                                         out_channels=2)
#   model1 =nn.DataParallel(model1)
#   model1.to(device)
            
#   model1.load_state_dict(torch.load(PATH, map_location=torch.device(device)))
#   #model1 =nn.DataParallel(model1)
#   model1.eval()
#   return model1
# def get_model_DesnNet201fold2(PATH):
#   model2 = monai.networks.nets.DenseNet121(spatial_dims=3, 
#                                         in_channels=3, 
#                                         out_channels=2)
#   model2 =nn.DataParallel(model2)
#   model2.to(device)
            
#   model2.load_state_dict(torch.load(PATH, map_location=torch.device(device)))
#   #model2 =nn.DataParallel(model2)
#   model2.eval()
#   return model2      
        
# def get_model_DesnNet201fold3(PATH):
#   model3 = monai.networks.nets.DenseNet121(spatial_dims=3, 
#                                         in_channels=3, 
#                                         out_channels=2)
#   model3 =nn.DataParallel(model3)
#   model3.to(device)
#   model3.load_state_dict(torch.load(PATH, map_location=torch.device(device)))
#   #model3 =nn.DataParallel(model3)
#   model3.eval()
#   return model3

# class EnsembledModel():
#   def __init__(self, model_paths):
#     super().__init__()
#     self.num_models = len(model_paths)
#     self.leafmodel1 = get_model_DesnNet201fold0(model_paths[0])
#     self.leafmodel2 = get_model_DesnNet201fold1(model_paths[1])
#     self.leafmodel3 = get_model_DesnNet201fold2(model_paths[2])
#     #self.leafmodel4 = get_model_DesnNet201fold3(model_paths[3])
#   def predict(self, x):
#     with torch.no_grad():
#       l1 = self.leafmodel1(x)
#       l2 = self.leafmodel2(x)
#       l3=self.leafmodel3(x)
#       #l4=self.leafmodel4(x)
#       pred = (l1+l2+l3) / (3)
#       return pred

# model_paths = [
#     './algorithm/model_densnet3dfold0.pth',
#     './algorithm/model_densnet3dfold1.pth',
#     './algorithm/model_densnet3dfold2.pth',
#     #'/content/drive/MyDrive/stoic2021-baseline-main/model_densnet3dfold3.pth',
#     ]
#model= EnsembledModel(model_paths)


from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)

from utils import MultiClassAlgorithm, unpack_single_output, device,to_input_format
from algorithm.preprocess import preprocess
#from algorithm.i3d.i3dpt import I3D
# def to_input_format1(input_image):
#     input_image = torch.Tensor(input_image)
#     #input_image=torch.unsqueeze(input_image,axis=0)
#     expdim=input_image.expand(3,input_image.shape[0],input_image.shape[1],input_image.shape[2])
#     input_image = expdim.unsqueeze(axis=0)
#     return input_image

def to_input_format1(input_image):
    input_image = torch.Tensor(input_image)
    #input_image=torch.unsqueeze(input_image,axis=0)
    expdim=input_image.expand(3,input_image.shape[0],input_image.shape[1],input_image.shape[2])
    input_image = expdim.unsqueeze(axis=0)
    return input_image

COVID_OUTPUT_NAME = Path("probability-covid-19")
SEVERE_OUTPUT_NAME = Path("probability-severe-covid-19")


class StoicAlgorithm(MultiClassAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
            input_path=Path("/input/images/ct/"),
            output_path=Path("/output/")
        )

        # load model
        #self.model =nn.DataParallel(model)
        #self.model = self.model.to(device)
        #self.model.load_state_dict(torch.load('./algorithm/model_resnet3d.pth', map_location=torch.device(device)))
        #self.model = self.model.eval()
        def get_model_DesnNet201fold0():
            model = monai.networks.nets.DenseNet121(spatial_dims=3, 
                                        in_channels=3, 
                                        out_channels=2)
            model =nn.DataParallel(model)
            model.to(device)
            
            model.load_state_dict(torch.load('./algorithm/model_densnet3dfold0.pth', map_location=torch.device(device)))
            #model1 =nn.DataParallel(model1)
            model.eval()
            return model
        
        def get_model_DesnNet201fold1():
            model1 = monai.networks.nets.DenseNet121(spatial_dims=3, 
                                        in_channels=3, 
                                        out_channels=2)
            model1 =nn.DataParallel(model1)
            model1.to(device)
            
            model1.load_state_dict(torch.load('./algorithm/model_densnet3dfold1.pth', map_location=torch.device(device)))
            #model1 =nn.DataParallel(model1)
            model1.eval()
            return model1
        
        def get_model_DesnNet201fold2():
            model2 = monai.networks.nets.DenseNet121(spatial_dims=3, 
                                        in_channels=3, 
                                        out_channels=2)
            model2 =nn.DataParallel(model2)
            model2.to(device)
            
            model2.load_state_dict(torch.load('./algorithm/model_densnet3dfold2.pth', map_location=torch.device(device)))
            #model2 =nn.DataParallel(model2)
            model2.eval()
            return model2  
        
        class EnsembledModel():
            def __init__(self,):
                super().__init__()
                self.num_models = 3
                self.leafmodel1 = get_model_DesnNet201fold0()
                self.leafmodel2 = get_model_DesnNet201fold1()
                self.leafmodel3 = get_model_DesnNet201fold2()
                #self.leafmodel4 = get_model_DesnNet201fold3(model_paths[3])
            def predict(self, x):
                with torch.no_grad():
                    l1 = self.leafmodel1(x)
                    l2 = self.leafmodel2(x)
                    #l3=self.leafmodel3(x)
                    #l4=self.leafmodel4(x)
                    pred = (l1+l2) / (self.num_models)
                    return pred
        
        
        self.model=EnsembledModel()
        

    def predict(self, *, input_image: SimpleITK.Image) -> Dict:
        # pre-processing
        input_image = preprocess(input_image)
        input_image = to_input_format1(input_image)

        # run model
        #with torch.no_grad():
        #self.model= self.EnsembledModel()
        output = torch.sigmoid(self.model.predict(input_image))
        prob_covid, prob_severe = unpack_single_output(output)
        #print(prob_covid)

        return {
            COVID_OUTPUT_NAME: prob_covid,
            SEVERE_OUTPUT_NAME: prob_severe
        }


if __name__ == "__main__":
    StoicAlgorithm().process()