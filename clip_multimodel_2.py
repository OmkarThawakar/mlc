#!/usr/bin/env python
# coding: utf-8


import os
import pandas as pd
from torchvision.io import read_image
from PIL import Image
import numpy as np



import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import json

from sklearn.preprocessing import MultiLabelBinarizer


import torch
import clip
from PIL import Image



import torch.nn as nn
import torchvision.transforms as T
from typing import Iterable
import torchvision
from torchvision.models._utils import IntermediateLayerGetter

from timm.utils import accuracy
from sklearn.metrics import accuracy_score


tag_list = [
    'Vegetarian',
'Vegan',
'Pescatarian',
'Paleo',
'Shellfish-Free',
'Nut-Free',
'Gluten-Free',
'Dairy-Free',
'American',
'Asian',
'Chinese',
'Indian',
'Italian',
'Japanese',
'Korean',
'Latin American',
'Mexican',
'Thai',
'Weeknight Dinner'
'Date Night',
'One-Pot',
'Bowl Meals',
'Soup',
'Salad',
'Pasta',
'Pizza',
'Stew',
'Breakfast',
'Brunch',
'Lunch',
'Dinner',
'Dessert',
'Appetizers',
'Main Dish',
'Side Dish',
'Snack',
'Drinks',
'Chicken',
'Beef',
'Salmon',
'Rice',
'Pork',
'Eggs',
'Tofu and Tempeh',
'Potatoes',
'Mushrooms',
'Beans & Legumes',
'Quick',
'Low-Carb',
'Keto',
'Budget-Friendly',
'Summer',
'Low-Calorie',
'Comfort Food',
'Baked Goods',
'Meals for Two',
'Cookies',
]


BATCH_SIZE,NUM_WORKERS = 1,1

device = "cuda" if torch.cuda.is_available() else "cpu"


### Custom Backbone Network for Image Encoder ###

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            # return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, x):
        xs = self.body(x)
        return xs

class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        norm_layer = FrozenBatchNorm2d
        if name == 'resnext101_64x4d':
            backbone = resnext101_64x4d(replace_stride_with_dilation=[False, False, dilation],
                pretrained=is_main_process(), norm_layer=norm_layer)
        else:
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=True, norm_layer=norm_layer)   #pretrained=is_main_process()
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class RecipeDataset(Dataset):
    def __init__(self, recipe_data_file, tags_file, img_dir, transform=None, target_transform=None):
        
        with open(recipe_data_file, 'r') as myfile:
            self.recipes = json.load(myfile)
            
        self.img_dir = img_dir
        
        # multi label binariser
        self.mlb = MultiLabelBinarizer()
        with open(tags_file, 'r') as myfile:
            self.tags = json.load(myfile)
        self.mlb.fit_transform([self.tags['tags']])
        self.tag_list = self.mlb.classes_
        
        self.transform = transform
        self.target_transform = target_transform
        
    def format_instrs(self, instrs):
        quantized_instrs = []
        instrs = instrs.split('\n')
        for inst in instrs :
            if len(inst.split(' ')) < 58 :
                quantized_instrs.append(inst)
            else:
                tmp = inst.split(' ')
                for i in range(0, len(tmp), 56):
                    quantized_instrs.append(' '.join(tmp[i:i+56]))

        return quantized_instrs

    def __len__(self):
        return len(self.recipes)

    def __getitem__(self, idx):
        
        # get the recipe data based on index
        recipe = self.recipes[idx]
        img_path = '{}/{}.png'.format(self.img_dir,recipe['id_'])
        image = Image.open((img_path)).convert('RGB')
        labels = np.array(self.mlb.transform([recipe['tags']]), dtype=np.float32)
        ingrs = recipe['recipe_ingredients']
        instrs = recipe['recipe_instructions'].split('\n') #self.format_instrs(recipe['recipe_instructions'])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(labels)
        return ingrs, instrs, image, labels


target_transform = T.Compose([
        T.ToTensor()
    ])

input_transform = T.Compose([
        T.Resize([256,256]),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])



dataset = RecipeDataset('../recipe_dataset/sidechef/my_recipes.json',
                        '../recipe_dataset/sidechef/my_tags.json', 
                        '../recipe_dataset/sidechef/images', 
                        transform = input_transform,
                        target_transform = target_transform
                       )



train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


print("Train Size : {}".format(len(train_dataset)))
print("Test Size : {}".format(len(test_dataset)))

sampler_train = torch.utils.data.RandomSampler(
            train_dataset
        )

sampler_test = torch.utils.data.RandomSampler(
            test_dataset
        )

data_loader_train = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler_train,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

data_loader_test = torch.utils.data.DataLoader(
        test_dataset, sampler=sampler_test,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

### for formatting ingredients and instructions from tuples #batch_size = 1 if needed
def format_data(x):
    out = []
    for item in x:
        out.append(item[0])
    return out


## Load clip model for encodings
clip_model, preprocess = clip.load("ViT-B/32", device=device)





class MultilabelClassifier(nn.Module):
    
    def __init__(self, inp_dims,n_classes):
        super().__init__()
        
        self.backbone = Backbone('resnet50', True, False, False)
        self.downsample = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1)        
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=inp_dims, out_features=512),
            nn.Linear(in_features=512, out_features=n_classes)
        )
        

    def forward(self, img_features, ing_encodings, inst_encodings):
        
        ing_encodings_mean = torch.mean(ing_encodings, axis=0).unsqueeze(dim=0)
        inst_encodings_mean = torch.mean(inst_encodings, axis=0).unsqueeze(dim=0)
        
        img_features = self.backbone(img_features)['0']
        img_features = self.downsample(img_features)
        img_features = torch.flatten(img_features).unsqueeze(dim=0)
        
        x = torch.cat([img_features, ing_encodings_mean, inst_encodings_mean], axis=1).float()
        
        x = self.classifier(x)
        x = torch.sigmoid(x)

        return x


embedding_dim = (16384 + 512 + 512)  # Image , Ing and Inst embedding dim

model = MultilabelClassifier(inp_dims=embedding_dim, n_classes=56)

model = model.to(device)


def criterion(loss_func,outputs,labels):
    loss = loss_func(outputs, labels)
    return loss


loss_func = nn.BCELoss()

lr_rate =0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)


def evaluate(model, loss_func, test_loader):
    
    model.eval()
    
    acc, loss = 0, 0
    
    print("Evaluate !!")
    
    for i, data in enumerate(test_loader):

        print("{}/{}".format(i, len(test_loader)), end='\r')

        ingrs, instrs, recipe_image, labels = data
        
        ingrs, instrs = format_data(ingrs), format_data(instrs)

        # labels = T.ToTensor()(labels).to(device)
        recipe_image = recipe_image.to(device)
        # labels = torch.tensor(labels, dtype=torch.float, device=device)
        labels = labels.to(device)
        labels = labels.squeeze(0)

        # image = preprocess(Image.open(recipe_image)).unsqueeze(0).to(device)
        ingrs_tokens = clip.tokenize(ingrs, truncate=True).to(device)
        instrs_tokens = clip.tokenize(instrs, truncate=True).to(device)

        with torch.no_grad():
            #recipe_image_encodings = clip_model.encode_image(image)
            ingrs_encodings = clip_model.encode_text(ingrs_tokens)
            instrs_encodings = clip_model.encode_text(instrs_tokens) 

        outputs = model(recipe_image, ingrs_encodings, instrs_encodings)
        loss += criterion(loss_func, outputs, labels).tolist()

        y_pred=[]
        for sample in outputs.cpu():
            y_pred.append([1 if i>=0.5 else 0 for i in sample ] )
        y_pred = np.array(y_pred)

        acc+= accuracy_score(y_pred, labels.cpu())
            
    accuracy = acc/len(test_loader)
    loss = loss/ len(test_loader)
    
    return accuracy, loss



def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer,
                    epoch: int):
    print("Training , Epoch : {}".format(epoch))
    
    total_iters = len(data_loader)
    
    for i, data in enumerate(data_loader):
        
        ingrs, instrs, recipe_image, labels = data
        
        ingrs, instrs = format_data(ingrs), format_data(instrs)

        #labels = T.ToTensor()(labels).to(device)
        recipe_image = recipe_image.to(device)
        #labels = torch.tensor(labels, dtype=torch.float, device=device)
        labels = labels.to(device)
        labels = labels.squeeze(0)

        #image = preprocess(Image.open(recipe_image)).unsqueeze(0).to(device)
        ingrs_tokens = clip.tokenize(ingrs, truncate=True).to(device)
        instrs_tokens = clip.tokenize(instrs, truncate=True).to(device)

        with torch.no_grad():
            #recipe_image_encodings = clip_model.encode_image(image)
            ingrs_encodings = clip_model.encode_text(ingrs_tokens)
            instrs_encodings = clip_model.encode_text(instrs_tokens) 

        outputs = model(recipe_image, ingrs_encodings, instrs_encodings)

        #loss = loss_func(outputs,labels)

        loss = criterion(loss_func,outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('progress : {}/{}, Loss : {}'.format(i, total_iters, loss), end='\r')

    print('='*50)
    torch.cuda.empty_cache()
    
    



EPOCHS = 300

max_acc = -1

for epoch in range(EPOCHS):
    
    train_one_epoch(model, data_loader_train, optimizer, epoch)
    
    # validate the model 
    
    acc, loss = evaluate(model, loss_func, data_loader_test)
    
    if acc > max_acc :
        state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                }
        torch.save(state, 'multimodel2/best_ckpt.pth')
        
        max_acc = acc
    
    print("Epoch : {} , Accuracy : {}, loss : {} ".format(epoch, max_acc, loss))

state = {
    'epoch': epoch,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
}
torch.save(state, 'multimodel2/epoch_{}_ckpt.pth'.format(epoch))
    
    
print('='*50)
print("Accuracy : {}".format(max_acc))
print("Finished !!!!!!")