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


device = "cuda" if torch.cuda.is_available() else "cpu"



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
        image = img_path #np.array(Image.open((img_path)))
        labels = np.array(self.mlb.transform([recipe['tags']]), dtype=np.float32)
        ingrs = recipe['recipe_ingredients']
        instrs = recipe['recipe_instructions'].split('\n') #self.format_instrs(recipe['recipe_instructions'])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return ingrs, instrs, image, labels


dataset = RecipeDataset('../recipe_dataset/sidechef/my_recipes.json',
                        '../recipe_dataset/sidechef/my_tags.json', 
                        '../recipe_dataset/sidechef/images')



train_size = int(0.80 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

print("Train Size : {}".format(len(train_dataset)))
print("Test Size : {}".format(len(test_dataset)))


## Load clip model for encodings
clip_model, preprocess = clip.load("ViT-B/32", device=device)



torch.set_grad_enabled(True) 



class MultilabelClassifier(nn.Module):
    
    def __init__(self, inp_dims,n_classes):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=inp_dims, out_features=512),
            nn.Linear(in_features=512, out_features=n_classes)
        )
        

    def forward(self, ing_encodings, inst_encodings):
        
        ing_encodings_mean = torch.mean(ing_encodings, axis=0).unsqueeze(dim=0)
        inst_encodings_mean = torch.mean(inst_encodings, axis=0).unsqueeze(dim=0)
        
        x = torch.cat([ing_encodings_mean, inst_encodings_mean], axis=1).float()
        
        x = self.classifier(x)
        x = torch.sigmoid(x)

        return x


model = MultilabelClassifier(inp_dims=512*2, n_classes=56)

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

        #labels = T.ToTensor()(labels).to(device)
        labels = torch.tensor(labels, dtype=torch.float, device=device)
        #print(labels)

        # image = preprocess(Image.open(recipe_image)).unsqueeze(0).to(device)
        ingrs_tokens = clip.tokenize(ingrs, truncate=True).to(device)
        instrs_tokens = clip.tokenize(instrs, truncate=True).to(device)

        with torch.no_grad():
            # recipe_image_encodings = clip_model.encode_image(image)
            ingrs_encodings = clip_model.encode_text(ingrs_tokens)
            instrs_encodings = clip_model.encode_text(instrs_tokens) 

            outputs = model(ingrs_encodings, instrs_encodings)
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

        #labels = T.ToTensor()(labels).to(device)
        labels = torch.tensor(labels, dtype=torch.float, device=device)

        # image = preprocess(Image.open(recipe_image)).unsqueeze(0).to(device)
        ingrs_tokens = clip.tokenize(ingrs, truncate=True).to(device)
        instrs_tokens = clip.tokenize(instrs, truncate=True).to(device)

        with torch.no_grad():
            # recipe_image_encodings = clip_model.encode_image(image)
            ingrs_encodings = clip_model.encode_text(ingrs_tokens)
            instrs_encodings = clip_model.encode_text(instrs_tokens) 

        outputs = model(ingrs_encodings, instrs_encodings)

        #loss = loss_func(outputs,labels)

        loss = criterion(loss_func,outputs, labels)
        #### loss.requires_grad = True
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('progress : {}/{}, Loss : {}'.format(i, total_iters, loss), end='\r')

    print('='*50)
    torch.cuda.empty_cache()
    



EPOCHS = 300

max_acc = -1

for epoch in range(EPOCHS):
    
    train_one_epoch(model,train_dataset,optimizer,epoch)
    
    # validate the model 
    
    acc, loss = evaluate(model, loss_func, test_dataset)
    
    if acc > max_acc :
        state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                }
        torch.save(state, 'nlp/best_ckpt.pth')
        
        max_acc = acc
    
    print("Epoch : {} , Accuracy : {}, loss : {} ".format(epoch, max_acc, loss))

state = {
    'epoch': epoch,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
}
torch.save(state, 'nlp/epoch_{}_ckpt.pth'.format(epoch))
    
    
print('='*50)
print("Accuracy : {}".format(max_acc))
print("Finished !!!!!!")