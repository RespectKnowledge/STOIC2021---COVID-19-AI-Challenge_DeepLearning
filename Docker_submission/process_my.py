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
model = monai.networks.nets.DenseNet121(spatial_dims=3, 
                                        in_channels=3, 
                                        out_channels=2)

model1 = monai.networks.nets.DenseNet121(spatial_dims=3, 
                                        in_channels=3, 
                                        out_channels=2)

model2 = monai.networks.nets.DenseNet121(spatial_dims=3, 
                                        in_channels=3, 
                                        out_channels=2)
model3 = monai.networks.nets.DenseNet121(spatial_dims=3, 
                                        in_channels=3, 
                                        out_channels=2)


# class EnsembledModel():

#     def __init__(self, model_paths):
#         super().__init__()
#         self.num_models = len(model_paths)

#         self.leafmodel1 = get_model_DesnNet201(model_paths[0])
#         #self.leafmodel2 = get_model_Effib3(model_paths[1])
#         #self.leafmodel3 = get_model_DesnNet201nn(model_paths[2])
#         #self.leafmodel4 = get_model_DesnNet201n(model_paths[3])
#         #self.leafmodel5 = get_model_transformer(model_paths[4])
        

#     def predict(self, x):
#         with torch.no_grad():
#             l1 = self.leafmodel1(x)
#             #l2 = self.leafmodel2(x)
#             #l3=self.leafmodel3(x)
#             #l4=self.leafmodel4(x)
#             #l5=self.leafmodel5(x)
#             #b4_e1 = self.effb4_model1(x)
#             pred = (l1) / (self.num_models)
#             #pred = (l1+l2+l4) 
#             #pred = l4

#             return pred

#model = monai.networks.nets.resnet10(spatial_dims=3, n_input_channels=1, n_classes=2)

from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)

from utils import MultiClassAlgorithm, unpack_single_output, device
from algorithm.preprocess import preprocess
#from algorithm.i3d.i3dpt import I3D
def to_input_format1(input_image):
    input_image = torch.Tensor(input_image)
    #input_image=torch.unsqueeze(input_image,axis=0)
    #expdim=input_image.expand(3,input_image.shape[0],input_image.shape[1],input_image.shape[2])
    input_image = input_image.unsqueeze(axis=0)
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
            input_path=Path("C:\\Users\\Administrateur\\Desktop\\airgochallenegs2022\\stoic2021-baseline-main\\my_testdocker\\test\\images/ct/"),
            output_path=Path("C:\\Users\\Administrateur\\Desktop\\airgochallenegs2022\\stoic2021-baseline-main\\my_testdocker\\outpute/")
        )
        #PATH='C:\\Users\\Administrateur\\Desktop\\airgochallenegs2022\\stoic2021-baseline-main\\my_testdocker\\model_resnet3d.pth', map_location=torch.device(device)
        def get_model_DesnNet201fold0():
            self.model =nn.DataParallel(model)
            #checkpoint = torch.load(PATH)
            model.load_state_dict(torch.load('C:\\Users\\Administrateur\\Desktop\\airgochallenegs2022\\stoic2021-baseline-main\\stoic2021-baseline-main1\\model_densnet3dfold0.pth', map_location=torch.device(device)))
            model.eval()
            return model.to(device)
        
        def get_model_DesnNet201fold1():
            self.model1 =nn.DataParallel(model1)
            #checkpoint = torch.load(PATH)
            model1.load_state_dict(torch.load('C:\\Users\\Administrateur\\Desktop\\airgochallenegs2022\\stoic2021-baseline-main\\stoic2021-baseline-main1\\model_densnet3dfold1.pth', map_location=torch.device(device)))
            model1.eval()
            return model1.to(device)
        
        def get_model_DesnNet201fold2():
            self.model2 =nn.DataParallel(model2)
            #checkpoint = torch.load(PATH)
            model2.load_state_dict(torch.load('C:\\Users\\Administrateur\\Desktop\\airgochallenegs2022\\stoic2021-baseline-main\\stoic2021-baseline-main1\\model_densnet3dfold2.pth', map_location=torch.device(device)))
            model2.eval()
            return model2.to(device)
        
        def get_model_DesnNet201fold3():
            self.model3 =nn.DataParallel(model3)
            #checkpoint = torch.load(PATH)
            model3.load_state_dict(torch.load('C:\\Users\\Administrateur\\Desktop\\airgochallenegs2022\\stoic2021-baseline-main\\stoic2021-baseline-main1\\model_densnet3dfold3.pth', map_location=torch.device(device)))
            model3.eval()
            return model3.to(device)
        
        class EnsembledModel():
            def __init__(self, model_paths=None):
                super().__init__()
                self.num_models = len(model_paths)

            self.leafmodel1 = get_model_DesnNet201fold0()
            self.leafmodel2 = get_model_DesnNet201fold1()
            self.leafmodel3 = get_model_DesnNet201fold2()
            self.leafmodel4 = get_model_DesnNet201fold3()
            #self.leafmodel2 = get_model_Effib3(model_paths[1])
            #self.leafmodel3 = get_model_DesnNet201nn(model_paths[2])
            #self.leafmodel4 = get_model_DesnNet201n(model_paths[3])
            #self.leafmodel5 = get_model_transformer(model_paths[4])
        

            def predict(self, x):
                with torch.no_grad():
                    l1 = self.leafmodel1(x)
                    l2 = self.leafmodel2(x)
                    l3=self.leafmodel3(x)
                    l4=self.leafmodel4(x)
                    #l5=self.leafmodel5(x)
                    #b4_e1 = self.effb4_model1(x)
                    pred = (l1+l2+l3+l4) / (4)
                    #pred = (l1+l2+l4) 
                    #pred = l4

                return pred
        self.model_e = EnsembledModel()

        # # load model
        # self.model =nn.DataParallel(model)
        # self.model = self.model.to(device)
        # #self.model.load_state_dict(torch.load()
        # self.model = self.model.eval()
        # #####
        # self.model1 =nn.DataParallel(model1)
        # self.model1 = self.model1.to(device)
        # self.model1.load_state_dict(torch.load('C:\\Users\\Administrateur\\Desktop\\airgochallenegs2022\\stoic2021-baseline-main\\my_testdocker\\model_resnet3d.pth', map_location=torch.device(device)))
        # self.model1 = self.model1.eval()
        
        # self.model2 =nn.DataParallel(model2)
        # self.model2 = self.model2.to(device)
        # self.model2.load_state_dict(torch.load('C:\\Users\\Administrateur\\Desktop\\airgochallenegs2022\\stoic2021-baseline-main\\my_testdocker\\model_resnet3d.pth', map_location=torch.device(device)))
        # self.model2 = self.model2.eval()
        
        # self.model3 =nn.DataParallel(model3)
        # self.model3 = self.model3.to(device)
        # self.model3.load_state_dict(torch.load('C:\\Users\\Administrateur\\Desktop\\airgochallenegs2022\\stoic2021-baseline-main\\my_testdocker\\model_resnet3d.pth', map_location=torch.device(device)))
        # self.model3 = self.model3.eval()
        
        
        

    def predict(self, *, input_image: SimpleITK.Image) -> Dict:
        # pre-processing
        input_image = preprocess(input_image)
        input_image = to_input_format1(input_image)

        # run model
        with torch.no_grad():
            output = torch.sigmoid(self.model_e(input_image))
        prob_covid, prob_severe = unpack_single_output(output)

        return {
            COVID_OUTPUT_NAME: prob_covid,
            SEVERE_OUTPUT_NAME: prob_severe
        }


if __name__ == "__main__":
    StoicAlgorithm().process()